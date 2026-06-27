'''测试 recall_at_k / ndcg_at_k / mrr_at_k 边界情况。'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.metrics import recall_at_k, ndcg_at_k, mrr_at_k


class TestRecallAtK:
    def test_empty_gt(self):
        '''空 ground truth 返回 0.0。'''
        assert recall_at_k([1, 2, 3], [], 5) == 0.0

    def test_perfect(self):
        '''完全命中时返回 1.0。'''
        assert recall_at_k([1, 2, 3], {1}, 5) == 1.0

    def test_partial(self):
        '''部分命中时返回命中比例。'''
        assert recall_at_k([1, 2, 3], {1, 4, 5}, 5) == 1.0 / 3

    def test_k_truncation(self):
        '''k 限制只考虑前 k 个结果。'''
        # ranked=[1,2,3], gt={3}, k=2 时命中不到
        assert recall_at_k([1, 2, 3], {3}, 2) == 0.0
        assert recall_at_k([1, 2, 3], {3}, 3) == 1.0

    def test_single_hit(self):
        '''单个命中项返回命中数/|gt|。'''
        ranked = [1, 2, 3]
        # recall = hits / len(gt)
        # hits 通过 set 去重，gt 中 1 和 1 都是同一个 item
        # 所以 hits=1, len(gt)=2 → 0.5
        r = recall_at_k(ranked, [1, 1], 5)
        assert r == 0.5  # hits=1, len(gt)=2


class TestNDCGAtK:
    def test_empty_gt(self):
        assert ndcg_at_k([1, 2, 3], [], 5) == 0.0

    def test_perfect_order(self):
        '''正确排序比错误排序 NDCG 高。'''
        # ranked 中 gt item 位置越靠前，NDCG 越高
        ndcg_high = ndcg_at_k([1, 2, 3, 4, 5], {1}, 5)   # 目标在第一位
        ndcg_low = ndcg_at_k([5, 4, 3, 2, 1], {1}, 5)    # 目标在最后
        assert ndcg_high > ndcg_low

    def test_ndcg_range(self):
        '''NDCG 值在 [0, 1] 范围内。'''
        val = ndcg_at_k([1, 2, 3], {1, 4}, 5)
        assert 0.0 <= val <= 1.0


class TestMRRAtK:
    def test_first_position(self):
        '''命中第一位时返回 1.0。'''
        assert mrr_at_k([1, 2, 3], {1}, 5) == 1.0

    def test_second_position(self):
        '''命中第二位时返回 0.5。'''
        assert mrr_at_k([1, 2, 3], {2}, 5) == 0.5

    def test_no_hit(self):
        '''未命中时返回 0.0。'''
        assert mrr_at_k([1, 2, 3], {99}, 5) == 0.0

    def test_k_truncation(self):
        '''k 限制生效。'''
        assert mrr_at_k([1, 2, 3], {3}, 2) == 0.0  # k=2 截断了
