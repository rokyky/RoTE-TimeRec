import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import PointWiseFeedForward


def get_time_buckets(bucket_defs):
    """将基于小时的桶定义转换为时间差桶。
    
    参数：
        bucket_defs: 小时阈值列表，例如 [0, 1, 6, 24, 168, 720]
    
    返回：
        time_buckets: (upper_bound_hours, bucket_id) 列表
    """
    buckets = []
    for i, threshold in enumerate(bucket_defs[:-1]):
        buckets.append((bucket_defs[i], bucket_defs[i + 1], i))
    # 最后一个桶捕获所有超出范围的值
    buckets.append((bucket_defs[-1], float('inf'), len(bucket_defs) - 1))
    return buckets


def discretize_time_delta(time_deltas, bucket_defs, max_delta=1e7):
    """将时间差离散化为桶索引。
    
    参数：
        time_deltas: 时间差的 np.array 或 torch.Tensor（以秒为单位）
        bucket_defs: 小时阈值列表
        max_delta: 极端值的上限
    
    返回：
        bucket_indices: 与 time_deltas 相同形状
    """
    if isinstance(time_deltas, torch.Tensor):
        device = time_deltas.device
        time_deltas = time_deltas.cpu().numpy()
    else:
        device = None

    time_deltas = np.clip(time_deltas / 3600.0, 0, max_delta)  # 转换为小时
    bucket_idxs = np.zeros_like(time_deltas, dtype=np.int64)
    for i, thresh in enumerate(bucket_defs):
        if i == len(bucket_defs) - 1:
            bucket_idxs[time_deltas >= thresh] = i
        else:
            next_thresh = bucket_defs[i + 1]
            mask = (time_deltas >= thresh) & (time_deltas < next_thresh)
            bucket_idxs[mask] = i

    if device is not None:
        bucket_idxs = torch.from_numpy(bucket_idxs).to(device)
    return bucket_idxs


class TiSASRec(nn.Module):
    """TiSASRec：时间间隔感知的自注意力序列推荐模型。
    
    原始论文：Li et al. (2020) "TiSASRec: Time Interval Aware Self-Attention
                for Sequential Recommendation"
    
    架构：
        - 物品Embedding + 位置编码
        - 带相对时间间隔偏置的自注意力
    """

    def __init__(self, num_items, hidden_dim=64, num_layers=2, num_heads=1,
                 dropout=0.2, max_len=50, time_bucket_defs=None):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        if time_bucket_defs is None:
            time_bucket_defs = [0, 1, 6, 24, 168, 720]  # 小时
        self.time_bucket_defs = time_bucket_defs
        self.num_time_buckets = len(time_bucket_defs)

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # 时间间隔偏置：每个桶的标量偏置（可学习）
        self.time_bias = nn.Embedding(self.num_time_buckets, 1)

        # 投影层（每层独立）
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

    def forward(self, seqs, positions, time_deltas):
        """前向传播。
        
        参数：
            seqs: (batch, max_len) LongTensor 类型，物品索引
            positions: (batch, max_len) LongTensor 类型，位置索引
            time_deltas: (batch, max_len, max_len) FloatTensor 类型，成对时间差
        
        返回：
            scores: (batch, num_items) 预测得分
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        # Embedding
        item_emb = self.item_emb(seqs)  # (B, L, D)
        pos_emb = self.pos_emb(positions)  # (B, L, D)
        x = self.dropout(item_emb + pos_emb)

        # 将时间差离散化为桶索引
        time_bucket_idxs = discretize_time_delta(
            time_deltas, self.time_bucket_defs
        ).to(device)  # (B, L, L)

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, L)

        # 有效位置掩码（非填充）
        pad_mask = (seqs != 0).unsqueeze(1).expand(-1, seq_len, -1)  # (B, L, L)

        for i in range(len(self.q_proj)):
            residual = x

            # 自注意力
            q = self.q_proj[i](x)  # (B, L, D)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)  # (B, L, L)

            # 时间偏置
            time_bias = self.time_bias(time_bucket_idxs).squeeze(-1)  # (B, L, L)
            attn = attn + time_bias

            # 掩码
            # 掩盖无效位置（填充）
            attn = attn.masked_fill(causal_mask | ~pad_mask, -1e9)

            attn_weights = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn_weights, v)  # (B, L, D)
            attn_out = self.dropout(attn_out)

            x = self.layer_norm1[i](residual + attn_out)

            # FFN
            residual = x
            x = self.ffn[i](x)
            x = self.layer_norm2[i](residual + x)

        # 取最后一个位置的输出
        last = x[:, -1, :]  # (B, D)

        scores = torch.matmul(last, self.item_emb.weight.t())

        return scores
