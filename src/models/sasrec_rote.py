"""SASRec + RoTE：带多粒度旋转时间嵌入的自注意力。

通过将 RoTE 时间嵌入添加到物品+位置表示来扩展 SASRec。
当未提供时间戳时，退化为标准 SASRec。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import PointWiseFeedForward
from .rote import RoTEEncoder


class SASRecRoTE(nn.Module):
    """带 RoTE 多粒度时间嵌入的 SASRec。

    在注意力层之前，将旋转时间嵌入添加到物品+位置求和中。
    当 timestamps 为 None 时，行为与 SASRec 完全相同。

    参数：
        num_items: 物品数量（不含填充索引 0）。
        hidden_dim: 模型维度。
        num_layers: Transformer 层数。
        num_heads: 注意力头数（当前为 1，保留以兼容 API）。
        dropout: Dropout 率。
        max_len: 最大序列长度。
        rote_granularities: RoTE 粒度名称列表。
        rote_theta_base: RoTE 频率计算基数。
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        max_len: int = 50,
        rote_granularities: Optional[list] = None,
        rote_theta_base: float = 10000.0,
    ):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_layers = num_layers

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        # RoTE time encoder
        if rote_granularities is None:
            rote_granularities = ['hour', 'day', 'week']
        self.rote_encoder = RoTEEncoder(
            hidden_dim=hidden_dim,
            granularities=rote_granularities,
            theta_base=rote_theta_base,
        )

        # Linear projection to combine RoTE features (optional,
        # but useful for mixing time info before attention)
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

    def forward(self, seqs, positions, timestamps=None):
        """前向传播。

        参数：
            seqs: (batch, max_len) LongTensor 类型，物品索引。
            positions: (batch, max_len) LongTensor 类型，位置索引。
            timestamps: (batch, max_len) 浮点张量的 Unix 时间戳，或 None。
                        若为 None，行为与标准 SASRec 相同。

        返回：
            scores: (batch, num_items) 预测得分。
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb

        # 如果提供了时间戳，添加 RoTE 时间嵌入
        if timestamps is not None:
            timestamps = timestamps.to(device=device, dtype=item_emb.dtype)
            rote_emb = self.rote_encoder(timestamps)  # (B, L, D)
            rote_emb = self.rote_proj(rote_emb)
            x = x + rote_emb

        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        pad_mask = (seqs != 0).unsqueeze(1).expand(-1, seq_len, -1)

        for i in range(self.num_layers):
            residual = x

            q = self.q_proj[i](x)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
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
