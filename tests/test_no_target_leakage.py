"""测试所有切分协议的目标泄露检测。

验证目标物品绝不会出现在任何切分协议的历史前缀中。
这对评估有效性至关重要。

用法：pytest tests/test_no_target_leakage.py -v
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
    """生成合成用户序列。"""
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
    """基础泄露检测函数测试。"""

    def test_no_leakage_clean_case(self):
        """目标不在历史中：返回 False。"""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(6, dtype=torch.long)
        assert not check_target_leakage(item_seq, target)

    def test_leakage_detected(self):
        """目标在历史中：返回 True。"""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(3, dtype=torch.long)
        assert check_target_leakage(item_seq, target)

    def test_padded_sequence(self):
        """填充（值 0）不应导致误报。"""
        item_seq = torch.tensor([0, 0, 1, 2, 3], dtype=torch.long)
        target = torch.tensor(0, dtype=torch.long)
        # 0 是填充，所以不应通过非填充检查
        # 但实际上，check_target_leakage 过滤了 seq[seq != 0] 中的 0
        # 所以 target=0 永远不会出现在过滤后的列表中，除非作为非填充出现
        # Target 0 如果只作为填充出现，不应泄露
        assert not check_target_leakage(item_seq, target)

    def test_leakage_with_history_end_idx(self):
        """带显式 history_end_idx 的测试。"""
        item_seq = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target = torch.tensor(3, dtype=torch.long)
        # 只检查位置 [0, 1, 2) — 不应在 idx=2 找到 target=3
        assert not check_target_leakage(item_seq, target, history_end_idx=2)
        # 检查位置 [0, 1, 2, 3) — 应在 idx=2 找到 target=3
        assert check_target_leakage(item_seq, target, history_end_idx=3)

    def test_empty_sequence(self):
        """空序列不应泄露。"""
        item_seq = torch.tensor([], dtype=torch.long)
        target = torch.tensor(1, dtype=torch.long)
        assert not check_target_leakage(item_seq, target)


class TestNoLeakageInSplitProtocols:
    """验证任何切分协议均无目标泄露。"""

    def test_leave_one_out_no_leakage(self):
        """留一法：目标是最后物品，不能出现在历史中。"""
        seqs, ts = make_sequences(n_users=20)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            assert not check_target_leakage(sample[0], sample[2]), \
                f"Leakage in leave_one_out: uid={sample[4].item()}"

    def test_no_sss_no_leakage(self):
        """无 SSS：目标是截断后的最后物品，不能出现在历史中。"""
        seqs, ts = make_sequences(n_users=20)
        samples = split_no_sss(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            assert not check_target_leakage(sample[0], sample[2]), \
                f"Leakage in no_sss: uid={sample[4].item()}"

    def test_sliding_window_no_leakage(self):
        """滑动窗口：目标绝不在其窗口的历史中。"""
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
        """前缀-目标：目标总是在所有历史物品之后。"""
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
        """在相同数据上运行所有协议，验证零泄露。"""
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
