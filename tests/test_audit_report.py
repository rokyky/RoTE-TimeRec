"""Test audit report: aggregate, slice, runtime fields, JSON export.

Usage: pytest tests/test_audit_report.py -v
"""

import json
import pytest
import torch
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.audit import (
    SplitProtocolAuditor,
    compute_runtime_metrics,
    compute_hard_slice_metrics,
)
from src.eval.metrics import _model_forward

MAX_LEN = 20  # consistent max_len across model, split, and data


def make_data(n_users=10, min_len=5, max_len=15, seed=42):
    """Generate synthetic data for audit testing."""
    import random
    random.seed(seed)
    sequences = {}
    timestamps = {}
    item_freq = {}
    item_categories = {}
    base_time = 1_700_000_000
    for uid in range(n_users):
        length = random.randint(min_len, max_len)
        seq = [random.randint(1, 50) for _ in range(length)]
        sequences[uid] = seq
        ts = [base_time + i * 3600 * random.uniform(1, 48) for i in range(length)]
        timestamps[uid] = ts
        for item in seq:
            item_freq[item] = item_freq.get(item, 0) + 1
            if item not in item_categories:
                item_categories[item] = random.randint(1, 4)
    return sequences, timestamps, item_freq, item_categories


def _make_sasrec_model():
    from src.models.sasrec import SASRec
    return SASRec(num_items=50, hidden_dim=32, num_layers=1, max_len=MAX_LEN)


class TestAuditReport:
    """Tests for the audit system output structure."""

    def test_audit_has_aggregate_field(self):
        """Audit output must contain 'aggregate' key."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5, 10])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out', 'no_sss'],
            user_timestamps=ts,
            item_freq=freq,
            item_categories=cats,
            max_len=MAX_LEN,
            compute_hard_slices=True,
            compute_runtime=True,
        )
        assert 'aggregate' in results
        assert 'slices' in results
        assert 'runtime' in results
        assert 'config' in results

    def test_aggregate_has_all_protocols(self):
        """All requested protocols should appear in aggregate."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out', 'no_sss', 'sliding_window_sss', 'prefix_target_sss'],
            user_timestamps=ts,
            max_len=MAX_LEN,
            compute_hard_slices=False,
            compute_runtime=False,
        )
        for protocol in ['leave_one_out', 'no_sss', 'sliding_window_sss', 'prefix_target_sss']:
            assert protocol in results['aggregate'], \
                f"Missing protocol '{protocol}' in aggregate"

    def test_aggregate_metrics_have_correct_keys(self):
        """Each protocol should have recall@k and ndcg@k."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5, 10])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out'],
            user_timestamps=ts,
            max_len=MAX_LEN,
            compute_hard_slices=False,
            compute_runtime=False,
        )
        metrics = results['aggregate']['leave_one_out']
        for k in [1, 5, 10]:
            assert f'recall@{k}' in metrics
            assert f'ndcg@{k}' in metrics

    def test_hard_slices_present(self):
        """Hard-slice evaluation should produce slice results."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out'],
            user_timestamps=ts,
            item_freq=freq,
            item_categories=cats,
            max_len=MAX_LEN,
            compute_hard_slices=True,
            compute_runtime=False,
        )
        slices = results['slices'].get('leave_one_out', {})
        slice_names = list(slices.keys())
        assert len(slice_names) > 0, "No hard slices produced"

    def test_runtime_metrics_structure(self):
        """Runtime metrics should have latency, memory, params fields."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out'],
            user_timestamps=ts,
            max_len=MAX_LEN,
            compute_hard_slices=False,
            compute_runtime=True,
        )
        runtime = results['runtime']
        assert runtime is not None
        assert 'avg_latency_ms' in runtime
        assert 'p95_latency_ms' in runtime
        assert 'param_count' in runtime
        assert 'peak_memory_mb' in runtime

    def test_json_export_valid(self):
        """JSON export should produce valid, parseable JSON."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out'],
            user_timestamps=ts,
            max_len=MAX_LEN,
            compute_hard_slices=False,
            compute_runtime=False,
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            auditor.export_json(results, f.name)
            json_path = f.name

        with open(json_path) as f:
            loaded = json.load(f)

        assert 'aggregate' in loaded
        assert 'config' in loaded
        assert isinstance(loaded['aggregate'], dict)
        os.unlink(json_path)

    def test_csv_export(self):
        """CSV table export should create files."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data()
        auditor = SplitProtocolAuditor(model, device='cpu', ks=[1, 5])
        results = auditor.audit(
            seqs,
            split_protocols=['leave_one_out'],
            user_timestamps=ts,
            max_len=MAX_LEN,
            compute_hard_slices=True,
            compute_runtime=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'test_audit')
            paths = auditor.export_tables(results, prefix=prefix)
            assert 'aggregate_csv' in paths
            assert os.path.exists(paths['aggregate_csv'])


class TestComputeRuntimeMetrics:
    """Tests for runtime metrics computation."""

    def test_returns_expected_keys(self):
        """Should return all expected runtime keys."""
        model = _make_sasrec_model()
        hist = torch.randint(1, 50, (2, MAX_LEN))
        pos = torch.arange(MAX_LEN).unsqueeze(0).expand(2, -1)
        # Use _model_forward directly with model as first arg
        sample_inputs = {
            'model': model,
            'hist': hist,
            'pos': pos,
            'device': 'cpu',
        }
        metrics = compute_runtime_metrics(model, sample_inputs, device='cpu', num_warmup=2, num_runs=5)
        assert 'avg_latency_ms' in metrics
        assert 'p95_latency_ms' in metrics
        assert 'param_count' in metrics
        assert metrics['param_count'] > 0


class TestHardSliceMetrics:
    """Tests for hard-slice evaluation."""

    def test_slice_output_structure(self):
        """Should produce per-slice metrics."""
        model = _make_sasrec_model()
        seqs, ts, freq, cats = make_data(n_users=10)
        from src.data.split_protocols import split_leave_one_out
        samples = split_leave_one_out(seqs, max_len=MAX_LEN, user_timestamps=ts)
        slices = compute_hard_slice_metrics(
            model, samples, 'cpu',
            item_freq=freq, item_categories=cats,
            ks=[1, 5],
        )
        assert len(slices) > 0
        for slice_name, metrics in slices.items():
            assert isinstance(metrics, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
