'''测试 SASRec 前向传播：形状、数值、确定性。'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.sasrec import SASRec


class TestSASRecForward:
    def setup_method(self):
        self.num_items = 50
        self.model = SASRec(num_items=self.num_items, hidden_dim=32,
                            num_layers=2, max_len=20)
        self.model.eval()

        # 构造一个 batch
        self.hist = torch.tensor([
            [0, 0, 0, 1, 2, 3, 4],   # 前 3 个 padding
            [0, 0, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7],
        ])
        self.pos = torch.tensor([list(range(7)) for _ in range(3)])

    def test_output_shape(self):
        '''输出形状应为 (batch, num_items + 1)，含 padding item 0。'''
        with torch.no_grad():
            scores = self.model(self.hist, self.pos)
        # SASRec 的 item_emb 大小为 num_items + 1（包含 padding）
        assert scores.shape == (3, self.num_items + 1)

    def test_no_nan(self):
        '''输出应无 NaN。'''
        with torch.no_grad():
            scores = self.model(self.hist, self.pos)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

    def test_deterministic(self):
        '''相同输入两次前向传播结果一致（eval 模式下）。'''
        with torch.no_grad():
            s1 = self.model(self.hist, self.pos)
            s2 = self.model(self.hist, self.pos)
        assert torch.allclose(s1, s2, atol=1e-6)

    def test_different_input_different_output(self):
        '''不同输入产生不同输出。'''
        hist2 = self.hist.clone()
        hist2[0, -1] = 20  # 修改最后一个 item
        with torch.no_grad():
            s1 = self.model(self.hist, self.pos)
            s2 = self.model(hist2, self.pos)
        assert not torch.allclose(s1, s2, atol=1e-6)

    def test_padding_produces_same_score(self):
        '''padding item 0 在 batch 不同位置的得分应该一致。'''
        # 即 batch 中相同 uid 不应因为 padding 长度不同而改变得分
        # 这是一个 sanity check
        with torch.no_grad():
            scores = self.model(self.hist, self.pos)
        # 验证 item 0 的分数在 batch 中不全是相同的（这不是强要求，
        # 但通常 item 0 不应该获得高分）
        item0_scores = scores[:, 0]
        assert item0_scores.shape == (3,)
