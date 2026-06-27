import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tisasrec import TiSASRec, discretize_time_delta


class TiSASRecCat(TiSASRec):
    """TiSASRec-Cat：类别条件化的相对时间注意力偏置。
    
    扩展 TiSASRec，添加类别条件化的时间偏置：
        bias(i,j) = time_bias(Δt) + cat_time_bias(same_cat, Δt)
    
    其中 same_cat ∈ {0,1} 表示物品 i 和 j 是否共享相同的叶子类别。
    cat_time_bias 学习两组独立的时间桶偏置：
    - 同类配对
    - 跨类配对
    
    这使得模型能够学习同类内与跨类间物品转换的不同时间动态。
    """

    def __init__(self, num_items, hidden_dim=64, num_layers=2, num_heads=1,
                 dropout=0.2, max_len=50, time_bucket_defs=None):
        super().__init__(num_items, hidden_dim, num_layers, num_heads,
                         dropout, max_len, time_bucket_defs)

        # 类别条件化的时间偏置：2（同/异）x 桶数
        self.cat_time_bias = nn.Embedding(
            self.num_time_buckets * 2, 1
        )

        self.apply(self._init_weights)

    def forward(self, seqs, positions, time_deltas, same_cat_mask):
        """带类别条件化时间偏置的前向传播。
        
        参数：
            seqs: (batch, max_len) LongTensor 类型，物品索引
            positions: (batch, max_len) LongTensor 类型，位置索引
            time_deltas: (batch, max_len, max_len) FloatTensor 类型，成对时间差（秒）
            same_cat_mask: (batch, max_len, max_len) BoolTensor 类型
                           True 表示物品 i 和 j 共享相同的叶子类别
        
        返回：
            scores: (batch, num_items) 预测得分
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = self.dropout(item_emb + pos_emb)

        # 将时间差离散化为桶索引
        time_bucket_idxs = discretize_time_delta(
            time_deltas, self.time_bucket_defs
        ).to(device)  # (B, L, L)

        # 类别条件化的桶索引
        # same_cat_mask: True -> 偏移 0, False -> 偏移 num_time_buckets
        cat_offset = (~same_cat_mask).long() * self.num_time_buckets  # (B, L, L)
        cat_time_bucket_idxs = time_bucket_idxs + cat_offset  # (B, L, L)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        pad_mask = (seqs != 0).unsqueeze(1).expand(-1, seq_len, -1)

        for i in range(len(self.q_proj)):
            residual = x

            q = self.q_proj[i](x)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)

            # 基础时间偏置（来自 TiSASRec）
            time_bias = self.time_bias(time_bucket_idxs).squeeze(-1)

            # 类别条件化的时间偏置
            cat_time_bias = self.cat_time_bias(cat_time_bucket_idxs).squeeze(-1)

            attn = attn + time_bias + cat_time_bias
            attn = attn.masked_fill(causal_mask | ~pad_mask, -1e9)
            attn_weights = F.softmax(attn, dim=-1)
            attn_out = torch.matmul(attn_weights, v)
            attn_out = self.dropout(attn_out)

            x = self.layer_norm1[i](residual + attn_out)

            residual = x
            x = self.ffn[i](x)
            x = self.layer_norm2[i](residual + x)

        last = x[:, -1, :]
        scores = torch.matmul(last, self.item_emb.weight.t())

        return scores
