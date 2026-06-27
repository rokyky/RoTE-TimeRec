"""RoTE（旋转时间嵌入）- 多粒度时间编码器。

将时间戳编码为旋转位置嵌入风格的表示，
涵盖多个时间粒度（小时、天、周、月等）。

参考：
    - 旋转位置嵌入（Su et al. 2021）
    - 由粗到细的多层级时间表示
"""

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn


def _get_granularity_seconds(granularity: str) -> float:
    """将粒度名称转换为其周期（秒）。

    支持的粒度：second, minute, hour, day, week, month（30天）, year（365天）。
    """
    mapping = {
        'second': 1.0,
        'minute': 60.0,
        'hour': 3600.0,
        'day': 86400.0,
        'week': 604800.0,
        'month': 2592000.0,   # 30 days
        'year': 31536000.0,   # 365 days
    }
    if granularity not in mapping:
        raise ValueError(f"Unknown granularity '{granularity}'. "
                         f"Supported: {list(mapping.keys())}")
    return mapping[granularity]


class RoTEEncoder(nn.Module):
    """多粒度旋转时间嵌入编码器。

    对于每个时间戳，此编码器：
    1. 将时间戳按每个粒度周期归一化
    2. 在每个层级应用正弦编码（旋转风格）
    3. 组合由粗到细的表示

    输出可用作注意力分数的加性偏置，
    或作为位置信号添加到物品嵌入中。

    参数：
        hidden_dim: 模型隐藏维度（旋转对编码需要偶数）。
        granularities: 时间粒度名称列表，按由粗到细排序。
        theta_base: 频率计算基数（如 RoPE theta）。
        learnable: 若为 True，粒度缩放因子可学习。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        granularities: Optional[List[str]] = None,
        theta_base: float = 10000.0,
        learnable: bool = False,
    ):
        super().__init__()
        if granularities is None:
            granularities = ['hour', 'day', 'week']

        self.hidden_dim = hidden_dim
        self.granularities = granularities
        self.num_granularities = len(granularities)
        self.theta_base = theta_base

        # 验证：hidden_dim 必须为偶数以便旋转对编码
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim 必须为偶数以便旋转编码，当前为 {hidden_dim}"
            )

        # 将粒度周期注册为缓冲区（每周期秒数）
        periods = torch.tensor(
            [_get_granularity_seconds(g) for g in granularities],
            dtype=torch.float,
        )
        self.register_buffer('periods', periods)

        # 每个粒度分配 (hidden_dim // 2) // num_granularities 维度，
        # 将 hidden_dim 分布到各层级。
        # 剩余维度归入最细粒度。
        base_dims = hidden_dim // 2 // self.num_granularities
        remainder = (hidden_dim // 2) - base_dims * self.num_granularities

        dims_per_level = []
        for i in range(self.num_granularities):
            d = base_dims
            if i == self.num_granularities - 1:
                d += remainder
            dims_per_level.append(d * 2)  # 每个层级获得一对 sin/cos

        self.dims_per_level = dims_per_level
        total = sum(dims_per_level)
        # 填充到 hidden_dim（应正好相等）
        assert total == hidden_dim, f"Dimension mismatch: {total} vs {hidden_dim}"

        # 注册频率乘数（theta_base^(-2i/d)）用于每个层级
        freqs = []
        for level_idx, dim_count in enumerate(dims_per_level):
            half_d = dim_count // 2
            i = torch.arange(half_d, dtype=torch.float)
            theta = theta_base ** (-2.0 * i / half_d) if half_d > 0 else torch.tensor([])
            freqs.append(theta)
        self.register_buffer('freqs', torch.cat(freqs) if freqs else torch.tensor([]))

        # 可选：每个粒度的可学习缩放因子
        if learnable:
            self.scale = nn.Parameter(torch.ones(self.num_granularities))
        else:
            self.register_buffer('scale', torch.ones(self.num_granularities))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """将时间戳编码为 RoTE 表示。

        参数：
            timestamps: (batch, seq_len) 浮点张量，Unix 时间戳（秒）。

        返回：
            rote_emb: (batch, seq_len, hidden_dim) 浮点张量。
                      不包含 NaN 或 Inf。
        """
        device = timestamps.device
        batch, seq_len = timestamps.shape

        # 归一化时间戳：转换为每个粒度的 [0, 2*pi) 相位偏移。
        # 通过周期进行归一化。
        # timestamps: (B, L) -> (B, L, 1) * (G,) -> (B, L, G)
        periods = self.periods.to(device)          # (G,)
        phases = timestamps.unsqueeze(-1) / periods  # (B, L, G)

        # 构建旋转编码：对每个粒度生成 sin(phase * freq) 和 cos(phase * freq) 对。
        # 频率在每个维度上是固定的（如 RoPE）。
        #
        # phases: (B, L, G)
        # freqs:  (D/2,)  -- 所有粒度共享
        # 需要按维度将 freq 应用于 phase。
        #
        # 策略：将 G 维展平为 D/2 对。
        # 对于维度数为 d_l 的层级 l，有 d_l/2 个频率。
        # 计算外积：(B, L) x (d_l/2) -> (B, L, d_l/2)

        output_parts = []
        offset = 0
        for level_idx, dim_count in enumerate(self.dims_per_level):
            half_d = dim_count // 2
            if half_d == 0:
                continue

            phase_l = phases[:, :, level_idx:level_idx + 1]  # (B, L, 1)
            freq_l = self.freqs[offset:offset + half_d].to(device)  # (half_d,)

            # 计算角度 = phase * freq（广播）
            # phase 单位为周期，所以 angle = 2*pi * phase * freq
            angle = 2.0 * math.pi * phase_l * freq_l  # (B, L, half_d)

            sin_enc = torch.sin(angle)  # (B, L, half_d)
            cos_enc = torch.cos(angle)  # (B, L, half_d)

            # 交错排列 sin 和 cos：(B, L, half_d, 2) -> (B, L, dim_count)
            level_out = torch.stack([sin_enc, cos_enc], dim=-1)  # (B, L, half_d, 2)
            level_out = level_out.view(batch, seq_len, dim_count)  # (B, L, dim_count)

            # 应用粒度缩放
            level_out = level_out * self.scale[level_idx]

            output_parts.append(level_out)
            offset += half_d

        rote_emb = torch.cat(output_parts, dim=-1)  # (B, L, hidden_dim)

        # 安全性：确保无 NaN/Inf
        rote_emb = torch.nan_to_num(rote_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        return rote_emb

    def get_output_dim(self) -> int:
        return self.hidden_dim
