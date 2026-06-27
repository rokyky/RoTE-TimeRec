import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class SASRec(nn.Module):
    """SASRec：自注意力序列推荐模型。
    
    Kang & McAuley (2018)
    使用自定义多头注意力（无时间偏置）以保持与 TiSASRec 的一致性。
    """

    def __init__(self, num_items, hidden_dim=64, num_layers=2, num_heads=1,
                 dropout=0.2, max_len=50):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        self.q_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.ffn = nn.ModuleList([
            PointWiseFeedForward(hidden_dim, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norm2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 1.0 / math.sqrt(self.hidden_dim)
            module.weight.data.normal_(0, std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, seqs, positions):
        """前向传播。
        
        参数：
            seqs: (batch, max_len) LongTensor 类型，物品索引
            positions: (batch, max_len) LongTensor 类型，位置索引
        
        返回：
            scores: (batch, num_items) 预测得分
        """
        device = seqs.device
        batch_size, seq_len = seqs.shape

        item_emb = self.item_emb(seqs)
        pos_emb = self.pos_emb(positions)
        x = self.dropout(item_emb + pos_emb)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        pad_mask = (seqs != 0).unsqueeze(1).expand(-1, seq_len, -1)

        for i in range(self.num_layers):
            residual = x

            q = self.q_proj[i](x)
            k = self.k_proj[i](x)
            v = self.v_proj[i](x)

            attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim)
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
