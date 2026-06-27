"""SASRec + RoTE: Self-Attention with multi-granularity rotary time embeddings.

Extends SASRec by adding RoTE time embeddings to the item+position
representation. When timestamps are not provided, degrades to regular SASRec.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import PointWiseFeedForward
from .rote import RoTEEncoder


class SASRecRoTE(nn.Module):
    """SASRec with RoTE multi-granularity time embeddings.

    Adds rotary time embeddings to the item+position sum before the
    attention layers. When timestamps is None, behaves exactly like SASRec.

    Args:
        num_items: Number of items (plus padding index 0).
        hidden_dim: Model dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (currently 1, kept for API compat).
        dropout: Dropout rate.
        max_len: Maximum sequence length.
        rote_granularities: List of granularity names for RoTE.
        rote_theta_base: Base for RoTE frequency computation.
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
        """Forward pass.

        Args:
            seqs: (batch, max_len) LongTensor, item indices.
            positions: (batch, max_len) LongTensor, position indices.
            timestamps: (batch, max_len) float tensor of Unix timestamps, or None.
                        When None, behaves like regular SASRec.

        Returns:
            scores: (batch, num_items) prediction scores.
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb

        # Add RoTE time embeddings if timestamps are provided
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
