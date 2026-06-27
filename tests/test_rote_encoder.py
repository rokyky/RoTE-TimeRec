"""测试 RoTE 编码器：形状、确定性、无 NaN/Inf、多粒度。

用法：pytest tests/test_rote_encoder.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.rote import RoTEEncoder, _get_granularity_seconds


class TestRoTEEncoder:
    """RoTE 多粒度时间编码器的测试。"""

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
        """输出应为 (batch, seq_len, hidden_dim)。"""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert out.shape == (batch_size, seq_len, hidden_dim), \
            f"Expected ({batch_size}, {seq_len}, {hidden_dim}), got {out.shape}"

    def test_output_deterministic(self, encoder, batch_size, seq_len):
        """相同输入必须产生相同输出。"""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out1 = encoder(timestamps)
        out2 = encoder(timestamps)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "RoTE encoder is not deterministic"

    def test_no_nan_inf(self, encoder, batch_size, seq_len):
        """输出不能包含 NaN 或 Inf。"""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_zero_timestamps(self, encoder, batch_size, seq_len):
        """零时间戳应产生有效输出（不崩溃）。"""
        timestamps = torch.zeros(batch_size, seq_len)
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Zero timestamps produced NaN"
        assert not torch.isinf(out).any(), "Zero timestamps produced Inf"

    def test_mixed_timestamps(self, encoder, batch_size, seq_len):
        """有效和零时间戳的混合（模拟填充）。"""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        timestamps[:, :3] = 0.0  # 模拟填充
        out = encoder(timestamps)
        assert not torch.isnan(out).any(), "Mixed timestamps produced NaN"

    def test_different_granularities_produce_different_output(self, hidden_dim, batch_size, seq_len):
        """不同的粒度列表应产生不同的输出。"""
        enc1 = RoTEEncoder(hidden_dim=hidden_dim, granularities=['hour'])
        enc2 = RoTEEncoder(hidden_dim=hidden_dim, granularities=['hour', 'day', 'week'])
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out1 = enc1(timestamps)
        out2 = enc2(timestamps)
        # 不同的粒度应产生不同的表示
        assert not torch.allclose(out1, out2, atol=1e-3), \
            "Different granularities should produce different outputs"

    def test_single_granularity(self, hidden_dim, batch_size, seq_len):
        """单个粒度也应能工作。"""
        encoder = RoTEEncoder(hidden_dim=hidden_dim, granularities=['day'])
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        out = encoder(timestamps)
        assert out.shape == (batch_size, seq_len, hidden_dim)

    def test_output_varies_with_timestamp(self, encoder, batch_size, seq_len):
        """不同的时间戳应产生不同的编码。"""
        ts1 = torch.zeros(batch_size, seq_len)
        ts2 = torch.ones(batch_size, seq_len) * 86400  # 1 天差异
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
        """无效的粒度名称应抛出 ValueError。"""
        with pytest.raises(ValueError):
            _get_granularity_seconds('decade')

    def test_odd_hidden_dim_raises(self):
        """奇数 hidden_dim 应抛出 ValueError（旋转编码需要偶数对）。"""
        with pytest.raises(ValueError):
            RoTEEncoder(hidden_dim=63)

    def test_device_transfer(self, encoder, batch_size, seq_len):
        """编码器应在 CPU 上工作并遵循输入设备。"""
        timestamps = torch.rand(batch_size, seq_len) * 1e9
        cpu_out = encoder(timestamps)
        assert cpu_out.device.type == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
