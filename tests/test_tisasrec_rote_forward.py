"""Test TiSASRec+RoTE forward: ablation switches, combined bias+rote, shapes.

Usage: pytest tests/test_tisasrec_rote_forward.py -v
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
    """Score output dim = num_items + 1 (includes padding idx 0)."""
    return num_items + 1


class TestTiSASRecRoTEForward:
    """Tests for TiSASRec + RoTE model."""

    def test_full_model_forward(self):
        """TiSASRec bias + RoTE enabled: output finite scores."""
        model = make_model(use_relative_bias=True, use_rote=True)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000  # time deltas in seconds
        ts = torch.rand(B, L) * 1e9
        scores = model(seqs, pos, td, timestamps=ts)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_bias_only_ablation(self):
        """TiSASRec bias only (RoTE disabled)."""
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
        """RoTE only (TiSASRec bias disabled)."""
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
        """Neither bias nor RoTE: pure position-only attention."""
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
        """When timestamps=None, RoTE should be skipped gracefully."""
        model = make_model(use_relative_bias=True, use_rote=True)
        B, L = 4, 20
        seqs = torch.randint(1, 100, (B, L))
        pos = torch.arange(L).unsqueeze(0).expand(B, -1)
        td = torch.rand(B, L, L) * 100000
        scores = model(seqs, pos, td, timestamps=None)
        assert scores.shape == (B, score_dim(100))
        assert not torch.isnan(scores).any()

    def test_float64_timestamps_supported(self):
        """Float64 timestamps from split protocols should be accepted."""
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
        """Different ablation settings should produce different scores."""
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

        # All should differ from each other (at least somewhat)
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
        """Padded sequences work without crash."""
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
        """Eval mode forward should be deterministic."""
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
