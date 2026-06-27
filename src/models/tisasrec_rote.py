"""TiSASRec + RoTE: Time Interval Aware Self-Attention with rotary time embeddings.

Extends TiSASRec by optionally adding RoTE multi-granularity time embeddings
alongside the relative time interval bias. Supports ablation of each component.

When timestamps is None, behaves like regular TiSASRec (only uses time_deltas).
When use_relative_bias=False, disables the TiSASRec time interval bias.
When use_rote=False, disables the RoTE time embeddings.
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
    """TiSASRec with optional RoTE multi-granularity time embeddings.

    Supports ablation by toggling relative time bias and RoTE independently,
    allowing analysis of their complementarity.

    Args:
        num_items: Number of items (plus padding index 0).
        hidden_dim: Model dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (kept for API compat).
        dropout: Dropout rate.
        max_len: Maximum sequence length.
        time_bucket_defs: Hour thresholds for discretizing time deltas.
        rote_granularities: List of granularity names for RoTE.
        rote_theta_base: Base for RoTE frequency computation.
        use_relative_bias: Whether to use TiSASRec time interval bias.
        use_rote: Whether to use RoTE time embeddings.
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

        # Relative time interval bias (TiSASRec style)
        self.time_bias = nn.Embedding(self.num_time_buckets, 1)

        # RoTE time encoder
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
        """Forward pass.

        Args:
            seqs: (batch, max_len) LongTensor, item indices.
            positions: (batch, max_len) LongTensor, position indices.
            time_deltas: (batch, max_len, max_len) FloatTensor, pairwise time diffs.
            timestamps: (batch, max_len) float tensor of Unix timestamps, or None.
                        When None, RoTE is skipped (if enabled).

        Returns:
            scores: (batch, num_items) prediction scores.
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = item_emb + pos_emb

        # Add RoTE time embeddings if enabled and timestamps provided
        if self.use_rote and timestamps is not None:
            timestamps = timestamps.to(device=device, dtype=item_emb.dtype)
            rote_emb = self.rote_encoder(timestamps)
            rote_emb = self.rote_proj(rote_emb)
            x = x + rote_emb

        x = self.dropout(x)

        # Discretize time deltas for relative bias (if enabled)
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

            # Add relative time bias (TiSASRec style) if enabled
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
