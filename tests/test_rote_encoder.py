"""Test RoTE Encoder: shape, determinism, no NaN/Inf, multi-granularity.

Usage: pytest tests/test_rote_encoder.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.rote import RoTEEncoder, _get_granularity_seconds


class TestRoTEEncoder:
    """Tests for the RoTE multi-granularity time encoder."""

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def seq_len(self):
        return 10

    @pytest.fixture
    def encoder(self, hidden_dim):
        return RoTEEncoder(
            hidden_dim=hidden_dim,
            granularities=['hour', 'day', 'week'],
            theta_base=10000.0,
        )

    def test_output_shape(self, encoder, batch_size, seq_len, hidden_dim):
        """Output should be (batch, seq_len, hidden_dim)."""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert out.shape == (batch_size, seq_len, hidden_dim), \
            f"Expected ({batch_size}, {seq_len}, {hidden_dim}), got {out.shape}"

    def test_output_deterministic(self, encoder, batch_size, seq_len):
        """Same input must produce identical output."""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out1 = encoder(timestamps)
        out2 = encoder(timestamps)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "RoTE encoder is not deterministic"

    def test_no_nan_inf(self, encoder, batch_size, seq_len):
        """Output must not contain NaN or Inf."""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_zero_timestamps(self, encoder, batch_size, seq_len):
        """Zero timestamps should produce valid output (not crash)."""
        timestamps = torch.zeros(batch_size, seq_len)
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Zero timestamps produced NaN"
        assert not torch.isinf(out).any(), "Zero timestamps produced Inf"

    def test_mixed_timestamps(self, encoder, batch_size, seq_len):
        """Mix of valid and zero timestamps (simulating padding)."""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        timestamps[:, :3] = 0.0  # simulate padding
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Mixed timestamps produced NaN"

    def test_different_granularities_produce_different_output(self, hidden_dim, batch_size, seq_len):
        """Different granularity lists should produce different outputs."""
        enc1 = RoTEEncoder(hidden_dim=hidden_dim, granularities=['hour'])
        enc2 = RoTEEncoder(hidden_dim=hidden_dim, granularities=['hour', 'day', 'week'])
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out1 = enc1(timestamps)
        out2 = enc2(timestamps)
        # Different granularities should produce different representations
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "Different granularities should produce different outputs"

    def test_single_granularity(self, hidden_dim, batch_size, seq_len):
        """Single granularity should work."""
        encoder = RoTEEncoder(hidden_dim=hidden_dim, granularities=['day'])
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert out.shape == (batch_size, seq_len, hidden_dim)

    def test_output_varies_with_timestamp(self, encoder, batch_size, seq_len):
        """Different timestamps should produce different encodings."""
        ts1 = torch.zeros(batch_size, seq_len)
        ts2 = torch.ones(batch_size, seq_len) * 86400  # 1 day difference
        out1 = encoder(ts1)
        out2 = encoder(ts2)
        diff = (out1 - out2).abs().mean().item()
        assert diff > 0.0, "Different timestamps should produce different encodings"

    def test_granularity_periods(self):
        """Test granularity string to seconds conversion."""
        assert _get_granularity_seconds('second') == 1.0
        assert _get_granularity_seconds('minute') == 60.0
        assert _get_granularity_seconds('hour') == 3600.0
        assert _get_granularity_seconds('day') == 86400.0
        assert _get_granularity_seconds('week') == 604800.0

    def test_invalid_granularity(self):
        """Invalid granularity name should raise ValueError."""
        with pytest.raises(ValueError):
            _get_granularity_seconds('decade')

    def test_odd_hidden_dim_raises(self):
        """Odd hidden_dim should raise ValueError (rotary needs even pairs)."""
        with pytest.raises(ValueError):
            RoTEEncoder(hidden_dim=63)

    def test_device_transfer(self, encoder, batch_size, seq_len):
        """Encoder should work on CPU and respect device of input."""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        cpu_out = encoder(timestamps)
        assert cpu_out.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
