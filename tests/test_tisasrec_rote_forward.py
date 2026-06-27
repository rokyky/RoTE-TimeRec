"""测试 TiSASRec+RoTE 前向传播：消融开关、组合偏置+RoTE、形状。

用法：pytest tests/test_tisasrec_rote_forward.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.tisasrec_rote import TiSASRecRoTE


def make_model(num_items=100, hidden_dim=64, max_len=20, **kwargs):
    return TiSASRecRoTE(
        num_items=num_items,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=1,
        dropout=0.1,
        max_len=max_len,
        time_bucket_defs=[0, 1, 6, 24, 168, 720],
        **kwargs,
    )


def score_dim(num_items):
    """得分输出维度 = num_items + 1（含填充索引 0）。"""
    return num_items + 1


class TestTiSASRecRoTEForward:
    """TiSASRec + RoTE 模型的测试。"""

    def test_full_model_forward(self):
        """TiSASRec 偏置 + RoTE 启用：输出有限得分。"""
        model = make_model(use_relative_bias=True, use_rote=True)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000  # 时间差（秒）
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_bias_only_ablation(self):
        """仅 TiSASRec 偏置（RoTE 禁用）。"""
        model = make_model(use_relative_bias=True, use_rote=False)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_rote_only_ablation(self):
        """仅 RoTE（TiSASRec 偏置禁用）。"""
        model = make_model(use_relative_bias=False, use_rote=True)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_neither_ablation(self):
        """既无偏置也无 RoTE：纯位置注意力。"""
        model = make_model(use_relative_bias=False, use_rote=False)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_no_timestamps_fallback(self):
        """当 timestamps=None 时，RoTE 应优雅跳过。"""
        model = make_model(use_relative_bias=True, use_rote=True)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        scores = model(seqs, pos, td, timestamps=None)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_float64_timestamps_supported(self):
        """切分协议产生的 Float64 时间戳应被接受。"""
        model = make_model(use_relative_bias=True, use_rote=True)
        B, L = 2, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L)
        ts = torch.rand(B, L, dtype=torch.float64) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_ablation_produces_different_outputs(self):
        """不同的消融设置应产生不同的得分。"""
        model_full = make_model(use_relative_bias=True, use_rote=True)
        model_bias = make_model(use_relative_bias=True, use_rote=False)
        model_rote = make_model(use_relative_bias=False, use_rote=True)
        model_neither = make_model(use_relative_bias=False, use_rote=False)

        B, L = 2, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9

        with torch.no_grad():
            s_full = model_full(seqs, pos, td, timestamps=ts)
            s_bias = model_bias(seqs, pos, td, timestamps=ts)
            s_rote = model_rote(seqs, pos, td, timestamps=ts)
            s_neither = model_neither(seqs, pos, td, timestamps=ts)

        # 所有组合应各不相同（至少略有差异）
        pairs = [
            (s_full, s_bias, "full vs bias-only"),
            (s_full, s_rote, "full vs rote-only"),
            (s_full, s_neither, "full vs neither"),
            (s_bias, s_rote, "bias-only vs rote-only"),
        ]
        for a, b, label in pairs:
            diff = (a - b).abs().mean().item()
            assert diff > 0.0, f"{label} should differ but diff=0"

    def test_padding_handling(self):
        """填充序列正常工作，不崩溃。"""
        model = make_model()
        B, L = 2, 20
        seqs = torch.randint(1, 100, (B, L))
        seqs[0, :5] = 0
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert not torch.isnan(scores).any()

    def test_deterministic_eval(self):
        """评估模式前向传播应具有确定性。"""
        model = make_model()
        model.eval()
        B, L = 2, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        ts = torch.rand(B, L) * 1e9
        with torch.no_grad():
            out1 = model(seqs, pos, td, timestamps=ts)
            out2 = model(seqs, pos, td, timestamps=ts)
        assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
