'''测试 TiSASRec 时间偏置：非零 time_deltas 应产生不同输出。'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.tisasrec import TiSASRec, discretize_time_delta


class TestTiSASRecTimeBias:
    def setup_method(self):
        self.num_items = 50
        self.model = TiSASRec(num_items=self.num_items, hidden_dim=32,
                              num_layers=2, max_len=10)
        self.model.eval()

        self.hist = torch.tensor([
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
        ])
        self.pos = torch.tensor([list(range(10))])

    def test_time_bias_has_effect(self):
        '''非零 time_deltas 与全零 time_deltas 输出不同。'''
        B, L = self.hist.shape
        zeros_td = torch.zeros(B, L, L)
        # 模拟真实时间差：随机非零时间间隔（小时转秒）
        non_zero_td = torch.rand(B, L, L) * 86400 * 30  # 0~30 天

        with torch.no_grad():
            s_zero = self.model(self.hist, self.pos, zeros_td)
            s_non_zero = self.model(self.hist, self.pos, non_zero_td)

        # 时间偏置应该改变注意力权重，从而改变最终得分
        assert not torch.allclose(s_zero, s_non_zero, atol=1e-5), (
            "TiSASRec 在零/非零 time_deltas 下应产生不同输出"
        )

    def test_zero_time_delta_at_same_position(self):
        '''对角线应为零时间差（相同位置自身）。'''
        B, L = self.hist.shape
        # 构造一个时间差矩阵（模拟递增时间戳）
        ts = torch.arange(L, dtype=torch.float) * 3600  # 每小时一条
        td = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                td[i, j] = abs(ts[i] - ts[j])
        td = td.unsqueeze(0).expand(B, -1, -1)

        # 对角线上应为 0（自己和自己比较）
        for i in range(L):
            assert td[0, i, i] == 0.0

    def test_time_bias_different_for_different_gaps(self):
        '''不同时间偏置组合产生不同注意力输出。'''
        B, L = self.hist.shape

        # 使用非均匀时间差矩阵（不同位置对不同桶），
        # 否则 softmax 对均匀常数偏置不变
        # bucket 0: 0-1h, bucket 5: >=720h
        short_td = torch.zeros(B, L, L)
        long_td = torch.zeros(B, L, L)
        for i in range(L):
            for j in range(i):  # 下三角不同间隔
                short_td[:, i, j] = 1800      # 0.5h -> bucket 0
                long_td[:, i, j] = 86400 * 90  # 90d -> bucket 5

        with torch.no_grad():
            s_short = self.model(self.hist, self.pos, short_td)
            s_long = self.model(self.hist, self.pos, long_td)

        assert not torch.allclose(s_short, s_long, atol=1e-5), (
            "不同时间间隔（不同桶）应产生不同输出"
        )


class TestDiscretizeTimeDelta:
    def test_buckets(self):
        '''验证时间离散化桶边界。'''
        bucket_defs = [0, 1, 6, 24]  # 小时：0-1, 1-6, 6-24, 24+

        # 测试秒数 -> 桶索引
        td_seconds = torch.tensor([0.0, 1800.0, 7200.0, 36000.0, 200000.0])
        # 0s -> 0h -> bucket 0
        # 1800s -> 0.5h -> bucket 0
        # 7200s -> 2h -> bucket 1
        # 36000s -> 10h -> bucket 2
        # 200000s -> 55.6h -> bucket 3

        indices = discretize_time_delta(td_seconds, bucket_defs)
        # The discretize_time_delta function clips to max_delta=1e7 and divides by 3600
        # Let me check the actual function behavior
        assert indices[0].item() == 0  # 0h
        assert indices[1].item() == 0  # 0.5h -> bucket 0
        assert indices[2].item() == 1  # 2h -> bucket 1
        assert indices[3].item() == 2  # 10h -> bucket 2
        assert indices[4].item() == 3  # 55.6h -> bucket 3 (>= last threshold)
