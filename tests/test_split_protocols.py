"""测试序列切分协议：样本数量、时间戳保留、形状。

用法：pytest tests/test_split_protocols.py -v
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
    """生成带时间戳的合成用户序列。"""
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
    """留一法切分协议的测试。"""

    def test_sample_count(self):
        """每个 >= 2 次交互的用户生成一个样本。"""
        seqs, ts = make_sequences(n_users=10)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        # 所有用户都有 >= 5 个物品，所以 10 个用户都应产生样本
        assert len(samples) == 10

    def test_sample_structure(self):
        """每个样本有 5 个元素：(item_seq, ts_seq, target_item, target_ts, uid)。"""
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
        """目标应是用户序列的最后一个物品。"""
        seqs, ts = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            uid = sample[4].item()
            target = sample[2].item()
            assert target == seqs[uid][-1], \
                f"Target {target} != last item {seqs[uid][-1]}"

    def test_timestamp_preserved(self):
        """目标时间戳应与最后交互时间一致。"""
        seqs, ts = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=ts)
        for sample in samples:
            uid = sample[4].item()
            target_ts = sample[3].item()
            expected_ts = ts[uid][-1]
            assert abs(target_ts - expected_ts) < 0.01, \
                f"Target ts {target_ts} != expected {expected_ts}"

    def test_no_timestamps(self):
        """无时间戳也应能工作（零填充）。"""
        seqs, _ = make_sequences(n_users=5)
        samples = split_leave_one_out(seqs, max_len=20, user_timestamps=None)
        for sample in samples:
            assert sample[3].item() == 0.0  # target_ts = 0


class TestNoSSS:
    """无子序列切分协议的测试。"""

    def test_one_sample_per_user(self):
        """每用户一个样本，无增强。"""
        seqs, ts = make_sequences(n_users=10)
        samples = split_no_sss(seqs, max_len=20, user_timestamps=ts)
        assert len(samples) == 10

    def test_truncation(self):
        """长度超过 max_len 的序列被截断。"""
        seqs = {0: list(range(1, 51))}  # 50 个物品
        ts = {0: [float(i) * 3600 for i in range(50)]}
        samples = split_no_sss(seqs, max_len=10, user_timestamps=ts)
        item_seq = samples[0][0]
        non_pad = item_seq[item_seq != 0]
        assert len(non_pad) <= 9  # 历史长度为 max_len-1


class TestSlidingWindowSSS:
    """滑动窗口 SSS 协议的测试。"""

    def test_sample_count(self):
        """每用户应生成多个样本。"""
        seqs, ts = make_sequences(n_users=5, min_len=10, max_len=15)
        window_size = 5
        samples = split_sliding_window_sss(
            seqs, max_len=20, window_size=window_size, user_timestamps=ts,
        )
        # 序列长度为 L 的每个用户产生 max(L - window_size + 1) 个样本
        # 最小长度=10，窗口=5，每用户至少 6 个样本，总共 >= 30
        assert len(samples) >= 30

    def test_window_size_validation(self):
        """window_size < 2 应抛出 ValueError。"""
        with pytest.raises(ValueError):
            split_sliding_window_sss({0: [1, 2, 3]}, window_size=1)


class TestPrefixTargetSSS:
    """前缀-目标 SSS 协议的测试。"""

    def test_sample_count(self):
        """每用户多个前缀-目标对。"""
        seqs, ts = make_sequences(n_users=5, min_len=10, max_len=10)
        prefix_min = 3
        samples = split_prefix_target_sss(
            seqs, max_len=20, prefix_min_len=prefix_min, user_timestamps=ts,
        )
        # 10 个物品的每个用户产生 (10 - 3) = 7 个样本
        assert len(samples) == 5 * 7

    def test_prefix_min_validation(self):
        """prefix_min_len < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError):
            split_prefix_target_sss({0: [1, 2, 3]}, prefix_min_len=0)

    def test_history_before_target(self):
        """所有历史物品应在原始序列中出现在目标之前。"""
        seqs = {0: list(range(1, 11))}
        samples = split_prefix_target_sss(seqs, max_len=20, prefix_min_len=2)
        for sample in samples:
            item_seq = sample[0]
            target = sample[2].item()
            non_pad = item_seq[item_seq != 0].tolist()
            # 所有历史物品应 < target（因为物品是顺序的 1..10）
            for h in non_pad:
                assert h < target, f"History item {h} >= target {target}"


class TestApplySplit:
    """apply_split 分发器的测试。"""

    def test_registered_protocols(self):
        """所有 4 个协议应被注册。"""
        assert set(SPLIT_REGISTRY.keys()) == {
            'leave_one_out', 'no_sss', 'sliding_window_sss', 'prefix_target_sss',
        }

    def test_unknown_protocol_raises(self):
        """未知协议应抛出 ValueError。"""
        with pytest.raises(ValueError):
            apply_split('nonexistent', {0: [1, 2, 3]})

    def test_apply_leave_one_out(self):
        """apply_split 正确分发。"""
        seqs = {0: [1, 2, 3, 4, 5], 1: [10, 20, 30]}
        samples = apply_split('leave_one_out', seqs, max_len=10)
        assert len(samples) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
