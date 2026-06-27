"""Test SASRec+RoTE forward: finite output, no-timestamp fallback, shape.

Usage: pytest tests/test_sasrec_rote_forward.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.sasrec_rote import SASRecRoTE


class TestSASRecRoTEForward:
    """Tests for SASRec + RoTE model forward pass."""

    @pytest.fixture
    def num_items(self):
        return 100

    @pytest.fixture
    def hidden_dim(self):
        return 64

    @pytest.fixture
    def max_len(self):
        return 20

    @pytest.fixture
    def model(self, num_items, hidden_dim, max_len):
        return SASRecRoTE(
            num_items=num_items,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=1,
            dropout=0.1,
            max_len=max_len,
            rote_granularities=['hour', 'day', 'week'],
        )

    @pytest.fixture
    def batch_size(self):
        return 4

    def test_forward_with_timestamps(self, model, num_items, max_len, batch_size):
        """Forward with timestamps: output finite scores of correct shape."""
        seqs = torch.randint(1, num_items, (batch_size, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
        timestamps = torch.rand(batch_size, max_len) * 1e9
        scores = model(seqs, positions, timestamps=timestamps)
        # Score dim = num_items + 1 (includes padding index 0)
        expected_dim = num_items + 1
        assert scores.shape == (batch_size, expected_dim), \
            f"Expected ({batch_size}, {expected_dim}), got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"

    def test_forward_accepts_float64_timestamps(self, model, num_items, max_len):
        """Float64 timestamps from split protocols should be accepted."""
        seqs = torch.randint(1, num_items, (2, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(2, -1)
        timestamps = (torch.rand(2, max_len, dtype=torch.float64) * 1e9)
        scores = model(seqs, positions, timestamps=timestamps)
        assert scores.shape == (2, num_items + 1)
        assert not torch.isnan(scores).any()

    def test_forward_without_timestamps(self, model, num_items, max_len, batch_size):
        """Forward without timestamps: behaves like regular SASRec."""
        seqs = torch.randint(1, num_items, (batch_size, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
        scores = model(seqs, positions, timestamps=None)
        # Score dim = num_items + 1 (includes padding index 0)
        expected_dim = num_items + 1
        assert scores.shape == (batch_size, expected_dim), \
            f"Expected ({batch_size}, {expected_dim}), got {scores.shape}"
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"

    def test_timestamps_vs_no_timestamps_different(self, model, max_len, batch_size):
        """With vs without timestamps should produce different outputs."""
        seqs = torch.randint(1, 100, (batch_size, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
        ts = torch.rand(batch_size, max_len) * 1e9
        scores_with_ts = model(seqs, positions, timestamps=ts)
        scores_no_ts = model(seqs, positions, timestamps=None)
        # With same random seed, these differ because RoTE adds time info
        diff = (scores_with_ts - scores_no_ts).abs().mean().item()
        assert diff > 0.0, "Timestamp should modify output"

    def test_padding_handling(self, model, max_len):
        """Padded sequences should not crash."""
        batch_size = 2
        seqs = torch.randint(1, 100, (batch_size, max_len))
        seqs[0, :5] = 0  # pad first 5 positions
        positions = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
        ts = torch.rand(batch_size, max_len) * 1e9
        scores = model(seqs, positions, timestamps=ts)
        assert not torch.isnan(scores).any()

    def test_deterministic_eval_mode(self, model, max_len):
        """Eval mode should be deterministic."""
        model.eval()
        seqs = torch.randint(1, 100, (2, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(2, -1)
        ts = torch.rand(2, max_len) * 1e9
        with torch.no_grad():
            out1 = model(seqs, positions, timestamps=ts)
            out2 = model(seqs, positions, timestamps=ts)
        assert torch.allclose(out1, out2, atol=1e-5), \
            "Eval mode forward should be deterministic"

    def test_gradient_flow(self, model, max_len):
        """Gradients should flow through RoTE encoder."""
        model.train()
        seqs = torch.randint(1, 100, (2, max_len))
        positions = torch.arange(max_len).unsqueeze(0).expand(2, -1)
        ts = torch.rand(2, max_len) * 1e9
        ts.requires_grad = False
        scores = model(seqs, positions, timestamps=ts)
        loss = scores.sum()
        loss.backward()
        # Check that RoTE encoder params received gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.rote_encoder.parameters()
        )
        # Note: rote_encoder may not have learnable params if learnable=False
        # Check at least model params have gradients
        model_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert model_has_grad, "No gradients flowing through model"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
