"""Test target leakage detection across all split protocols.

Verifies that the target item NEVER appears in the history prefix
for any split protocol. This is critical for evaluation validity.

Usage: pytest tests/test_no_target_leakage.py -v
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
    check_target_leakage,
)


def make_sequences(n_users=10, min_len=5, max_len=15, seed=42):
    """Generate synthetic user sequences."""
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


class TestTargetLeakageBasic:
    """Basic leakage check function tests."""

    def test_no_leakage_clean_case(self):
        """Target not in history: returns False."""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(6, dtype=torch.long)
        assert not check_target_leakage(item_seq, target)

    def test_leakage_detected(self):
        """Target IS in history: returns True."""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(3, dtype=torch.long)
        assert check_target_leakage(item_seq, target)

    def test_padded_sequence(self):
        """Padding (value 0) should not cause false positives."""
        item_seq = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
        target = torch.tensor(0, dtype=torch.long)
        # 0 is padding, so should not be considered in non-padding check
        # But check_target_leakage filters out 0... let me check:
        # Actually, it filters seq[seq != 0], so target=0 would never be in
        # the filtered list unless it appears as non-padding.
        # Target 0 should NOT leak if 0 only appears as padding
        assert not check_target_leakage(item_seq, target)

    def test_leakage_with_history_end_idx(self):
        """With explicit history_end_idx."""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(3, dtype=torch.long)
        # Check only positions [0, 1, 2) — should NOT find target=3 at idx 2
        assert not check_target_leakage(item_seq, target, history_end_idx=2)
        # Check positions [0, 1, 2, 3) — SHOULD find target=3 at idx 2
        assert check_target_leakage(item_seq, target, history_end_idx=3)

    def test_empty_sequence(self):
        """Empty sequence should not leak."""
        item_seq = torch.tensor([], dtype=torch.long)
        target = torch.tensor(1, dtype=torch.long)
        assert not check_target_leakage(item_seq, target)


class TestNoLeakageInSplitProtocols:
    """Verify no target leakage in any split protocol."""

    def test_leave_one_out_no_leakage(self):
        """Leave-one-out: target is last item, must not be in history."""
        seqs, ts = make_sequences(n_users=20)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            assert not check_target_leakage(sample[0], sample[2]), \
                f"Leakage in leave_one_out: uid={sample[4].item()}"

    def test_no_sss_no_leakage(self):
        """No SSS: target is last truncated item, must not be in history."""
        seqs, ts = make_sequences(n_users=20)
        samples = split_no_sss(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            assert not check_target_leakage(sample[0], sample[2]), \
                f"Leakage in no_sss: uid={sample[4].item()}"

    def test_sliding_window_no_leakage(self):
        """Sliding window: target never in its own window's history."""
        seqs, ts = make_sequences(n_users=20, min_len=8, max_len=15)
        samples = split_sliding_window_sss(
            seqs, max_len=20, window_size=5, user_timestamps=ts,
        )
        leakage_count = 0
        for sample in samples:
            if check_target_leakage(sample[0], sample[2]):
                leakage_count += 1
        assert leakage_count == 0, \
            f"Found {leakage_count} leakage cases in sliding_window_sss"

    def test_prefix_target_no_leakage(self):
        """Prefix-target: target is always after all history items."""
        seqs, ts = make_sequences(n_users=20, min_len=8, max_len=15)
        samples = split_prefix_target_sss(
            seqs, max_len=20, prefix_min_len=3, user_timestamps=ts,
        )
        leakage_count = 0
        for sample in samples:
            if check_target_leakage(sample[0], sample[2]):
                leakage_count += 1
        assert leakage_count == 0, \
            f"Found {leakage_count} leakage cases in prefix_target_sss"

    def test_all_protocols_no_leakage_comprehensive(self):
        """Run all protocols on the same data, verify zero leakage."""
        seqs, ts = make_sequences(n_users=30, min_len=5, max_len=20)
        protocols = {
            'leave_one_out': lambda: split_leave_one_out(seqs, max_len=15, user_timestamps=ts),
            'no_sss': lambda: split_no_sss(seqs, max_len=15, user_timestamps=ts),
            'sliding_window_sss': lambda: split_sliding_window_sss(seqs, max_len=15, window_size=5, user_timestamps=ts),
            'prefix_target_sss': lambda: split_prefix_target_sss(seqs, max_len=15, prefix_min_len=3, user_timestamps=ts),
        }
        for name, fn in protocols.items():
            samples = fn()
            leakages = sum(
                1 for s in samples if check_target_leakage(s[0], s[2])
            )
            assert leakages == 0, \
                f"Protocol '{name}' has {leakages}/{len(samples)} leakage cases"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
