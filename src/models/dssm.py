"""DSSM：双塔向量召回模型。

User Tower（user_id → Embedding → MLP → L2 Normalize → user_emb）
Item Tower（item_id → Embedding → MLP → L2 Normalize → item_emb）

训练：in-batch negative sampled softmax + temperature。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def _mlp(dims: list, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class DSSM(nn.Module):
    """双塔向量召回模型。

    参数：
        num_users: 用户数量（0-based）
        num_items: 物品数量（不含 padding）
        hidden_dim: 隐向量维度
        mlp_dims: MLP 隐层维度列表。默认 [hidden_dim, hidden_dim] 即单隐层
        dropout: MLP dropout 率
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int = 64,
        mlp_dims: list | None = None,
        dropout: float = 0.0,
        no_mlp: bool = False,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim

        self.user_emb = nn.Embedding(num_users, hidden_dim)
        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)

        if no_mlp:
            self.user_mlp = nn.Identity()
            self.item_mlp = nn.Identity()
        else:
            mlp_dims = mlp_dims or [hidden_dim, hidden_dim]
            self.user_mlp = _mlp([hidden_dim] + mlp_dims + [hidden_dim], dropout)
            self.item_mlp = _mlp([hidden_dim] + mlp_dims + [hidden_dim], dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        u = self.user_mlp(u)
        return F.normalize(u, p=2, dim=-1)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        i = self.item_emb(item_ids)
        i = self.item_mlp(i)
        return F.normalize(i, p=2, dim=-1)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """返回 (B,) 逐对得分。"""
        u = self.encode_user(user_ids)   # (B, D)
        i = self.encode_item(item_ids)   # (B, D)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def get_all_item_embs(self) -> torch.Tensor:
        """返回 (num_items+1, D) 全部 item embedding（含 padding）。"""
        all_ids = torch.arange(self.num_items + 1, device=self.item_emb.weight.device)
        return self.encode_item(all_ids)


class DSSMDataset(Dataset):
    """DSSM 训练数据集。从序列交互中提取 (user_id, pos_item_id) pair。"""

    def __init__(self, sequences: dict):
        self.pairs = []
        for uid, seq in sequences.items():
            for item in seq:
                self.pairs.append((uid, item))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uid, iid = self.pairs[idx]
        return uid, iid


def collate_dssm(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return users, items


def train_epoch_softmax(
    model: DSSM,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float = 0.05,
) -> float:
    """In-batch negative sampled softmax 训练。

    batch 内其他样本的 item 作为当前 user 的负样本。
    适合 dense signal、大 batch 场景。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for users, pos_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        batch_size = users.size(0)

        optimizer.zero_grad()

        u_emb = model.encode_user(users)
        i_emb = model.encode_item(pos_items)

        logits = torch.mm(u_emb, i_emb.t()) / temperature
        labels = torch.arange(batch_size, device=device)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_epoch_bpr(
    model: DSSM,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_neg: int = 100,
) -> float:
    """BPR pairwise loss — 预计算 item embedding 表，大幅加速负采样。

    对每个 (user, pos_item) 对采样 num_neg 个负样本，
    BPR loss: -log(sigmoid(pos - neg))。
    """
    model.train()

    # 预计算全量 item embedding（经过 MLP），避免重复前向
    with torch.no_grad():
        all_i_mlp = model.item_mlp(model.item_emb.weight)  # (N+1, D)

    total_loss = 0.0
    num_batches = 0

    for users, pos_items in loader:
        users = users.to(device)
        pos_items = pos_items.to(device)
        batch_size = users.size(0)

        optimizer.zero_grad()

        # user embedding (B, D)
        u_emb = model.user_mlp(model.user_emb(users))
        # 正样本
        i_pos = all_i_mlp[pos_items]
        pos_scores = (u_emb * i_pos).sum(dim=-1)  # (B,)

        # 负样本：从预计算表中索引
        neg_ids = torch.randint(1, model.num_items + 1, (batch_size, num_neg), device=device)
        i_neg = all_i_mlp[neg_ids]  # (B, num_neg, D)
        neg_scores = (u_emb.unsqueeze(1) * i_neg).sum(dim=-1)  # (B, num_neg)

        # BPR loss
        pos_exp = pos_scores.unsqueeze(-1).expand(-1, num_neg)
        loss = -F.logsigmoid(pos_exp - neg_scores).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)
