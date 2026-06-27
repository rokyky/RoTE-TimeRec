"""BPR-MF：Bayesian Personalized Ranking Matrix Factorization。

使用矩阵分解 + BPR pairwise loss 的经典协同过滤模型。
作为序列推荐项目中的传统学习型 baseline。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class BPRMF(nn.Module):
    """BPR Matrix Factorization 模型。

    为每个 user 和 item 学习一个隐因子向量，
    通过 user_emb · item_emb 的点积计算偏好得分。

    参数：
        num_users: int, 用户数量（0-based）
        num_items: int, 物品数量（不含 padding）
        hidden_dim: int, 隐因子维度
    """

    def __init__(self, num_users, num_items, hidden_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    def forward(self, user_ids):
        """获取指定用户的全部 item 得分。

        参数：
            user_ids: (B,) LongTensor, user indices

        返回：
            scores: (B, num_items+1), 每个候选 item 的得分
        """
        u_emb = self.user_emb(user_ids)  # (B, D)
        scores = torch.matmul(u_emb, self.item_emb.weight.t())  # (B, N+1)
        return scores

    def predict_pair(self, user_ids, item_ids):
        """预测指定 user-item pair 的得分。

        参数：
            user_ids: (B,) LongTensor
            item_ids: (B,) LongTensor

        返回：
            scores: (B,)
        """
        u_emb = self.user_emb(user_ids)
        i_emb = self.item_emb(item_ids)
        return (u_emb * i_emb).sum(dim=-1)


class BPRMFDataset(Dataset):
    """BPR-MF 训练数据集。

    从训练 DataFrame 构建 (user_id, pos_item_id) pair 列表。
    每个对代表一个 user 的 1 个正样本交互。
    """

    def __init__(self, train_df, user2idx, item2idx,
                 user_col='reviewerID', item_col='asin'):
        self.pairs = []
        for _, row in train_df.iterrows():
            uid = user2idx.get(row[user_col])
            iid = item2idx.get(row[item_col])
            if uid is not None and iid is not None and iid > 0:
                self.pairs.append((uid, iid))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user_id, item_id = self.pairs[idx]
        return user_id, item_id


def collate_bprmf(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    pos_items = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return users, pos_items


def train_epoch_bprmf(model, loader, optimizer, device, num_neg=100):
    """训练一个 epoch 的 BPR-MF。

    对每个 batch 的正样本对采样 num_neg 个负样本，
    计算 BPR pairwise loss: -log(sigmoid(pos - neg))。

    返回：
        avg_loss: float, 平均 batch loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for users, pos_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        batch_size = users.size(0)

        optimizer.zero_grad()

        # 正样本得分
        pos_scores = model.predict_pair(users, pos_items)  # (B,)

        # 负样本采样（在每个 batch 内随机）
        neg_items = torch.randint(
            1, model.num_items + 1, (batch_size, num_neg), device=device
        )  # (B, num_neg)

        # 扩展 user_ids 以匹配负样本维度
        users_exp = users.unsqueeze(-1).expand(-1, num_neg)  # (B, num_neg)
        neg_scores = model.predict_pair(users_exp, neg_items)  # (B, num_neg)

        # BPR loss: -log(sigmoid(pos - neg))
        pos_exp = pos_scores.unsqueeze(-1).expand(-1, num_neg)  # (B, num_neg)
        loss = -F.logsigmoid(pos_exp - neg_scores).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
