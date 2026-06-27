"""RoTE (Rotary Time Embedding) - Multi-granularity time encoder.

Encodes timestamps into rotary-position-embedding-style representations
across multiple time granularities (hour, day, week, month, etc.).

Reference:
    - Rotary Position Embedding (Su et al. 2021)
    - coarse-to-fine multi-level time representation
"""

import math
from typing import List, Optional, Union

import torch
import torch.nn as nn


def _get_granularity_seconds(granularity: str) -> float:
    """Convert a granularity name to its period in seconds.

    Supported: second, minute, hour, day, week, month (30d), year (365d).
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
    """Multi-granularity Rotary Time Embedding encoder.

    For each timestamp, this encoder:
    1. Normalizes the timestamp by each granularity period
    2. Applies sinusoidal encoding (rotary-style) at each level
    3. Combines coarse-to-fine representations

    The output can be used as an additive bias to attention scores or
    as a positional signal added to item embeddings.

    Args:
        hidden_dim: Model hidden dimension (must be even for rotary pairs).
        granularities: List of time granularity names, ordered coarse-to-fine.
        theta_base: Base for frequency computation (like RoPE theta).
        learnable: If True, make granularity scaling factors learnable.
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

        # Validate: hidden_dim must be even for rotary pair encoding
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be even for rotary encoding, got {hidden_dim}"
            )

        # Register granularity periods as buffers (seconds per cycle)
        periods = torch.tensor(
            [_get_granularity_seconds(g) for g in granularities],
            dtype=torch.float,
        )
        self.register_buffer('periods', periods)

        # For each granularity, we allocate (hidden_dim // 2) // num_granularities
        # dimensions, distributing the hidden_dim across levels.
        # Remaining dimensions go to the finest level.
        base_dims = hidden_dim // 2 // self.num_granularities
        remainder = (hidden_dim // 2) - base_dims * self.num_granularities

        dims_per_level = []
        for i in range(self.num_granularities):
            d = base_dims
            if i == self.num_granularities - 1:
                d += remainder
            dims_per_level.append(d * 2)  # each level gets both sin/cos pairs

        self.dims_per_level = dims_per_level
        total = sum(dims_per_level)
        # Pad to hidden_dim if needed (should be exact)
        assert total == hidden_dim, f"Dimension mismatch: {total} vs {hidden_dim}"

        # Register frequency multipliers (theta_base^(-2i/d)) for each level
        freqs = []
        for level_idx, dim_count in enumerate(dims_per_level):
            half_d = dim_count // 2
            i = torch.arange(half_d, dtype=torch.float)
            theta = theta_base ** (-2.0 * i / half_d) if half_d > 0 else torch.tensor([])
            freqs.append(theta)
        self.register_buffer('freqs', torch.cat(freqs) if freqs else torch.tensor([]))

        # Optional learnable scaling per granularity
        if learnable:
            self.scale = nn.Parameter(torch.ones(self.num_granularities))
        else:
            self.register_buffer('scale', torch.ones(self.num_granularities))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps into RoTE representations.

        Args:
            timestamps: (batch, seq_len) float tensor of Unix timestamps (seconds).

        Returns:
            rote_emb: (batch, seq_len, hidden_dim) float tensor.
                      No NaN or Inf values.
        """
        device = timestamps.device
        batch, seq_len = timestamps.shape

        # Normalize timestamps: convert to a phase offset in [0, 2*pi)
        # per granularity.  We normalize by the period.
        # timestamps: (B, L) -> (B, L, 1) * (G,) -> (B, L, G)
        periods = self.periods.to(device)          # (G,)
        phases = timestamps.unsqueeze(-1) / periods  # (B, L, G)

        # Build the rotary encoding: for each granularity, we produce
        # sin(phase * freq) and cos(phase * freq) pairs.
        # The frequencies are fixed per dimension (like RoPE).
        #
        # phases: (B, L, G)
        # freqs:  (D/2,)  -- shared across all granularities
        # We need to apply freq to phase per dimension.
        #
        # Strategy: flatten the G dims into D/2 pairs.
        # For each level l with dim_count d_l, we have d_l/2 frequencies.
        # We compute the outer product: (B, L) x (d_l/2) -> (B, L, d_l/2)

        output_parts = []
        offset = 0
        for level_idx, dim_count in enumerate(self.dims_per_level):
            half_d = dim_count // 2
            if half_d == 0:
                continue

            phase_l = phases[:, :, level_idx:level_idx + 1]  # (B, L, 1)
            freq_l = self.freqs[offset:offset + half_d].to(device)  # (half_d,)

            # Compute angle = phase * freq  (broadcast)
            # phase is in cycles, so angle = 2*pi * phase * freq
            angle = 2.0 * math.pi * phase_l * freq_l  # (B, L, half_d)

            sin_enc = torch.sin(angle)  # (B, L, half_d)
            cos_enc = torch.cos(angle)  # (B, L, half_d)

            # Interleave sin and cos: (B, L, half_d, 2) -> (B, L, dim_count)
            level_out = torch.stack([sin_enc, cos_enc], dim=-1)  # (B, L, half_d, 2)
            level_out = level_out.view(batch, seq_len, dim_count)  # (B, L, dim_count)

            # Apply granularity scaling
            level_out = level_out * self.scale[level_idx]

            output_parts.append(level_out)
            offset += half_d

        rote_emb = torch.cat(output_parts, dim=-1)  # (B, L, hidden_dim)

        # Safety: ensure no NaN/Inf
        rote_emb = torch.nan_to_num(rote_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        return rote_emb

    def get_output_dim(self) -> int:
        return self.hidden_dim
