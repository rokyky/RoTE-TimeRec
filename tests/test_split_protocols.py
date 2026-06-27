"""Test sequence split protocols: sample counts, timestamp preservation, shapes.

Usage: pytest tests/test_split_protocols.py -v
"""

import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.split_protocols import (
    split_leave_one_out,
    split_no_sss,
    split_sliding_window_sss,
    split_prefix_target_sss,
    apply_split,
    SPLIT_REGISTRY,
)


def make_sequences(n_users=10, min_len=5, max_len=15, seed=42):
    """Generate synthetic user sequences with timestamps."""
    import random
    random.seed(seed)
    sequences = {}
    timestamps = {}
    base_time = 1_700_000_000
    for uid in range(n_users):
        length = random.randint(min_len, max_len)
        seq = [random.randint(1, 100) for _ in range(length)]
        sequences[uid] = seq
        ts = [base_time + i * 3600 * random.uniform(1, 48) for i in range(length)]
        timestamps[uid] = ts
    return sequences, timestamps


class TestLeaveOneOut:
    """Tests for leave-one-out split protocol."""

    def test_sample_count(self):
        """One sample per user with >= 2 interactions."""
        seqs, ts = make_sequences(n_users=10)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        # All users have >= 5 items, so all 10 should produce samples
        assert len(samples) == 10

    def test_sample_structure(self):
        """Each sample has 5 elements: (item_seq, ts_seq, target_item, target_ts, uid)."""
        seqs, ts = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            assert len(sample) == 5
            assert isinstance(sample[0], torch.LongTensor)     # item_seq
            assert isinstance(sample[1], torch.Tensor)          # ts_seq
            assert torch.is_floating_point(sample[1])
            assert sample[1].dtype == torch.float64
            assert isinstance(sample[2], torch.LongTensor)      # target_item
            assert isinstance(sample[3], torch.Tensor)          # target_ts
            assert torch.is_floating_point(sample[3])
            assert sample[3].dtype == torch.float64
            assert isinstance(sample[4], torch.LongTensor)      # user_id

    def test_target_is_last_item(self):
        """Target should be the last item in user sequence."""
        seqs, ts = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            uid = sample[4].item()
            target = sample[2].item()
            assert target == seqs[uid][-1], \
                f"Target {target} != last item {seqs[uid][-1]}"

    def test_timestamp_preserved(self):
        """Target timestamp should match last interaction time."""
        seqs, ts = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            uid = sample[4].item()
            target_ts = sample[3].item()
            expected_ts = ts[uid][-1]
            assert abs(target_ts - expected_ts) < 0.01, \
                f"Target ts {target_ts} != expected {expected_ts}"

    def test_no_timestamps(self):
        """Should work without timestamps (zero-filled)."""
        seqs, _ = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=None)
        for sample in samples:
            assert sample[3].item() == 0.0  # target_ts = 0


class TestNoSSS:
    """Tests for no-sub-sequence-split protocol."""

    def test_one_sample_per_user(self):
        """One sample per user, no augmentation."""
        seqs, ts = make_sequences(n_users=10)
        samples = split_no_sss(seqs, max_len=20, user_timestamps=ts)
        assert len(samples) == 10

    def test_truncation(self):
        """Sequences longer than max_len are truncated."""
        seqs = {0: list(range(1, 51))}  # 50 items
        ts = {0: [float(i) * 3600 for i in range(50)]}
        samples = split_no_sss(seqs, max_len=10, user_timestamps=ts)
        item_seq = samples[0][0]
        non_pad = item_seq[item_seq != 0]
        assert len(non_pad) <= 9  # history is max_len-1


class TestSlidingWindowSSS:
    """Tests for sliding-window SSS protocol."""

    def test_sample_count(self):
        """Should produce multiple samples per user."""
        seqs, ts = make_sequences(n_users=5, min_len=10, max_len=15)
        window_size = 5
        samples = split_sliding_window_sss(
            seqs, max_len=20, window_size=window_size, user_timestamps=ts,
        )
        # Each user with seq_len L produces max(L - window_size + 1) samples
        # With min_len=10, window_size=5, at least 6 samples per user, so >= 30 total
        assert len(samples) >= 30

    def test_window_size_validation(self):
        """window_size < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            split_sliding_window_sss({0: [1, 2, 3]}, window_size=1)


class TestPrefixTargetSSS:
    """Tests for prefix-target SSS protocol."""

    def test_sample_count(self):
        """Multiple prefix-target pairs per user."""
        seqs, ts = make_sequences(n_users=5, min_len=10, max_len=10)
        prefix_min = 3
        samples = split_prefix_target_sss(
            seqs, max_len=20, prefix_min_len=prefix_min, user_timestamps=ts,
        )
        # Each user with 10 items produces (10 - 3) = 7 samples
        assert len(samples) == 5 * 7

    def test_prefix_min_validation(self):
        """prefix_min_len < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            split_prefix_target_sss({0: [1, 2, 3]}, prefix_min_len=0)

    def test_history_before_target(self):
        """All history items should come before target in original sequence."""
        seqs = {0: list(range(1, 11))}
        samples = split_prefix_target_sss(seqs, max_len=20, prefix_min_len=2)
        for sample in samples:
            item_seq = sample[0]
            target = sample[2].item()
            non_pad = item_seq[item_seq != 0].tolist()
            # All history items should be < target (since items are sequential 1..10)
            for h in non_pad:
                assert h < target, f"History item {h} >= target {target}"


class TestApplySplit:
    """Tests for the apply_split dispatcher."""

    def test_registered_protocols(self):
        """All 4 protocols should be registered."""
        assert set(SPLIT_REGISTRY.keys()) == {
            'leave_one_out', 'no_sss', 'sliding_window_sss', 'prefix_target_sss',
        }

    def test_unknown_protocol_raises(self):
        """Unknown protocol should raise ValueError."""
        with pytest.raises(ValueError):
            apply_split('nonexistent', {0: [1, 2, 3]})

    def test_apply_leave_one_out(self):
        """apply_split dispatches correctly."""
        seqs = {0: [1, 2, 3, 4, 5], 1: [10, 20, 30]}
        samples = apply_split('leave_one_out', seqs, max_len=10)
        assert len(samples) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
