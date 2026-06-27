"""TiSASRec + RoTE：带旋转时间嵌入的时间间隔感知自注意力。

通过可选地添加 RoTE 多粒度时间嵌入来扩展 TiSASRec，
以及相对时间间隔偏置。支持各组件的消融实验。

当 timestamps 为 None 时，行为与标准 TiSASRec 相同（仅使用 time_deltas）。
当 use_relative_bias=False 时，禁用 TiSASRec 时间间隔偏置。
当 use_rote=False 时，禁用 RoTE 时间嵌入。
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import PointWiseFeedForward
from .tisasrec import discretize_time_delta
from .rote import RoTEEncoder


class TiSASRecRoTE(nn.Module):
    """带可选 RoTE 多粒度时间嵌入的 TiSASRec。

    支持通过独立切换相对时间偏置和 RoTE 进行消融，
    便于分析两者的互补性。

    参数：
        num_items: 物品数量（不含填充索引 0）。
        hidden_dim: 模型维度。
        num_layers: Transformer 层数。
        num_heads: 注意力头数（保持以兼容 API）。
        dropout: Dropout 率。
        max_len: 最大序列长度。
        time_bucket_defs: 时间差离散化的小时阈值。
        rote_granularities: RoTE 粒度名称列表。
        rote_theta_base: RoTE 频率计算基数。
        use_relative_bias: 是否使用 TiSASRec 时间间隔偏置。
        use_rote: 是否使用 RoTE 时间嵌入。
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        max_len: int = 50,
        time_bucket_defs: Optional[list] = None,
        rote_granularities: Optional[list] = None,
        rote_theta_base: float = 10000.0,
        use_relative_bias: bool = True,
        use_rote: bool = True,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.use_relative_bias = use_relative_bias
        self.use_rote = use_rote

        if time_bucket_defs is None:
            time_bucket_defs = [0, 1, 6, 24, 168, 720]
        self.time_bucket_defs = time_bucket_defs
        self.num_time_buckets = len(time_bucket_defs)

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # 相对时间间隔偏置（TiSASRec 风格）
        self.time_bias = nn.Embedding(self.num_time_buckets, 1)

        # RoTE 时间编码器
        if rote_granularities is None:
            rote_granularities = ['hour', 'day', 'week']
        self.rote_encoder = RoTEEncoder(
            hidden_dim=hidden_dim,
            granularities=rote_granularities,
            theta_base=rote_theta_base,
        )
        self.rote_proj = nn.Linear(hidden_dim, hidden_dim)

        self.q_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.ffn = nn.ModuleList([
            PointWiseFeedForward(hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norm2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 1.0 / math.sqrt(self.hidden_dim)
            module.weight.data.normal_(0, std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, seqs, positions, time_deltas, timestamps=None):
        """前向传播。

        参数：
            seqs: (batch, max_len) LongTensor 类型，物品索引。
            positions: (batch, max_len) LongTensor 类型，位置索引。
            time_deltas: (batch, max_len, max_len) FloatTensor 类型，成对时间差。
            timestamps: (batch, max_len) 浮点张量的 Unix 时间戳，或 None。
                        若为 None，则跳过 RoTE（如果启用）。

        返回：
            scores: (batch, num_items) 预测得分。
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb

        # 如果启用且提供了时间戳，添加 RoTE 时间嵌入
        if self.use_rote and timestamps is not None:
            timestamps = timestamps.to(device=device, dtype=item_emb.dtype)
            rote_emb = self.rote_encoder(timestamps)
            rote_emb = self.rote_proj(rote_emb)
            x = x + rote_emb

        x = self.dropout(x)

        # 如果启用，对相对偏置的时间差进行离散化
        if self.use_relative_bias:
            time_bucket_idxs = discretize_time_delta(
                time_deltas, self.time_bucket_defs
            ).to(device)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        pad_mask = (seqs != 0).unsqueeze(1).expand(-1, seq_len, -1)

        for i in range(len(self.q_proj)):
            residual = x

            q = self.q_proj[i](x)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)

            # 如果启用，添加相对时间偏置（TiSASRec 风格）
            if self.use_relative_bias:
                time_bias = self.time_bias(time_bucket_idxs).squeeze(-1)
                attn = attn + time_bias

            attn = attn.masked_fill(causal_mask | ~pad_mask, -1e9)
            attn_weights = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = self.dropout(attn_out)

            x = self.layer_norm1[i](residual + attn_out)

            residual = x
            x = self.ffn[i](x)
            x = self.layer_norm2[i](residual + x)

        last = x[:, -1, :]
        scores = torch.matmul(last, self.item_emb.weight.t())

        return scores
