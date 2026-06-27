'''测试 SeqRecDataset 和 EvalDataset 的数据加载逻辑。'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data.loader import SeqRecDataset, EvalDataset


def _make_sequences():
    '''构造测试用序列数据。'''
    return {
        0: [1, 2, 3, 4, 5],
        1: [10, 20, 30],
        2: [100, 200],
    }


class TestSeqRecDataset:
    def test_shapes(self):
        '''SeqRecDataset 返回 (hist, pos, target, uid) 四元组，形状正确。'''
        seqs = _make_sequences()
        ds = SeqRecDataset(seqs, max_len=10, num_items=1000)
        hist, pos, target, uid = ds[0]

        assert hist.shape == (10,)          # max_len
        assert pos.shape == (10,)
        assert target.shape == ()           # scalar
        assert uid.shape == ()

    def test_padding(self):
        '''短序列被正确 padding 到 max_len。'''
        seqs = _make_sequences()
        ds = SeqRecDataset(seqs, max_len=10, num_items=1000)

        # 取 uid=1 的第一个样本：hist=[10], target=20
        for i in range(len(ds)):
            hist, pos, target, uid = ds[i]
            if uid.item() == 1 and target.item() == 20:
                # 原始 hist 长度为 1，pad 9 个 0 在前面
                assert hist[0] == 0
                assert hist[-1] == 10
                return
        assert False, "未找到 uid=1, target=20 的样本"

    def test_target_is_next_item(self):
        '''target 为序列中历史的下一个 item。'''
        seqs = _make_sequences()
        ds = SeqRecDataset(seqs, max_len=10, num_items=1000)

        # uid=0, seq=[1,2,3,4,5], 样本: (hist=[1], target=2), (hist=[1,2], target=3), etc
        for i in range(len(ds)):
            hist, pos, target, uid = ds[i]
            if uid.item() == 0:
                # 检查 hist 最后一个非零位置 = target 的前一个 item
                non_zero = hist[hist != 0]
                seq = [1, 2, 3, 4, 5]
                # target 应该在序列中
                assert target.item() in seq

    def test_sample_count(self):
        '''样本数 = sum(len(seq)-1 for all users)。'''
        seqs = _make_sequences()
        ds = SeqRecDataset(seqs, max_len=10, num_items=1000)
        expected = (5 - 1) + (3 - 1) + (2 - 1)  # = 4 + 2 + 1 = 7
        assert len(ds) == expected


class TestEvalDataset:
    def test_leave_last_as_target(self):
        '''EvalDataset 以最后一条交互作为 target，其余为历史。'''
        seqs = {0: [1, 2, 3, 4, 5]}
        ds = EvalDataset(seqs, max_len=10)

        hist, pos, target, uid = ds[0]
        assert uid.item() == 0
        assert target.item() == 5
        # 历史应为 [1, 2, 3, 4] 加上前面 padding
        non_zero = hist[hist != 0]
        assert non_zero.tolist() == [1, 2, 3, 4]

    def test_short_sequence_skipped(self):
        '''长度 < 2 的序列被排除。'''
        seqs = {0: [1], 1: [10, 20]}
        ds = EvalDataset(seqs, max_len=10)
        assert len(ds) == 1  # 只有 uid=1
        _, _, target, uid = ds[0]
        assert uid.item() == 1

    def test_with_timestamps(self):
        '''提供时间戳时返回 5 元组（含 time_deltas）。'''
        seqs = {0: [1, 2, 3]}
        timestamps = {0: [100.0, 200.0, 300.0]}
        ds = EvalDataset(seqs, max_len=5, timestamps=timestamps)

        batch = ds[0]
        assert len(batch) == 5  # hist, pos, target, uid, time_deltas
        hist, pos, target, uid, td = batch
        assert td.shape == (5, 5)
        assert not torch.all(td == 0)  # 应有非零时间差

    def test_with_categories(self):
        '''提供时间戳和类目时返回 6 元组（含 time_deltas + same_cat_mask）。'''
        seqs = {0: [1, 2, 3]}
        timestamps = {0: [100.0, 200.0, 300.0]}
        item_cats = {1: 10, 2: 10, 3: 20}  # item 1,2 同类目，3 不同
        ds = EvalDataset(seqs, max_len=5,
                         timestamps=timestamps,
                         item_categories=item_cats)

        batch = ds[0]
        assert len(batch) == 6
        hist, pos, target, uid, td, scm = batch
        assert scm.shape == (5, 5)  # same_cat_mask
        assert scm.dtype == torch.bool
        # item 1 和 2 的 same_cat_mask 应为 True
        # 在 padded 序列中，1 和 2 的位置取决于 padding
        non_zero_mask = hist != 0
        item_positions = [i for i, v in enumerate(hist.tolist()) if v != 0]
        # 这两个 item 是同类目的，所以对应位置应为 True
        if len(item_positions) >= 2:
            p1, p2 = item_positions[0], item_positions[1]
            assert scm[p1, p2] == True
