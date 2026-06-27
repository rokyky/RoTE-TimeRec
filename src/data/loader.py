'''数据加载与预处理模块。

参考：
    - RecBole 数据处理管线
    - 带负采样的序列数据集
'''

import random
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SeqRecDataset(Dataset):
    '''带可选时间戳支持的训练数据集，适用于 RoTE 模型。

    当提供时间戳时，__getitem__ 返回 5 元素元组：
        (hist, pos, target, uid, ts_hist)
    否则返回 4 元素元组以保持向后兼容：
        (hist, pos, target, uid)
    '''

    def __init__(self,
                 sequences: Dict[int, List[int]],
                 max_len: int = 50,
                 num_items: int = 0,
                 neg_samples: int = 1,
                 item_popularity: Optional[Dict[int, int]] = None,
                 timestamps: Optional[Dict[int, List[float]]] = None):
        self.max_len = max_len
        self.num_items = num_items
        self.neg_samples = neg_samples
        self.item_popularity = item_popularity
        self.timestamps = timestamps or {}
        self.has_timestamps = bool(self.timestamps)
        self.samples = []
        for uid, seq in sequences.items():
            for i in range(1, len(seq)):
                self.samples.append((uid, seq[:i], seq[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        uid, hist, target = self.samples[idx]
        hist = hist[-self.max_len:]
        pad_len = self.max_len - len(hist)
        hist = [0] * pad_len + hist
        pos = list(range(0, len(hist)))

        result = [
            torch.tensor(hist, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long),
        ]

        # 如果可用，追加原始时间戳（用于 RoTE 模型）
        if self.has_timestamps:
            ts_all = self.timestamps.get(uid, [])
            # 将时间戳与历史前缀对齐
            hist_len = min(len(hist) - pad_len, len(ts_all))
            ts_hist = [0.0] * pad_len
            for j in range(hist_len):
                ts_hist.append(ts_all[j] if j < len(ts_all) else 0.0)
            ts_hist = ts_hist[-self.max_len:]  # 确保精确长度
            result.append(torch.tensor(ts_hist, dtype=torch.float))

        return tuple(result)


class EvalDataset(Dataset):
    '''带可选时间戳和类目支持的评估数据集。

    当提供时间戳和 item_categories 时，__getitem__ 返回
    6 元素元组：(hist, pos, target, uid, time_deltas, same_cat_mask)。
    否则返回 4 元素元组以保持向后兼容。
    '''

    def __init__(self,
                 sequences: Dict[int, List[int]],
                 max_len: int = 50,
                 timestamps: Optional[Dict[int, List[float]]] = None,
                 item_categories: Optional[Dict[int, int]] = None,
                 return_timestamps: bool = False):
        self.max_len = max_len
        self.users = []
        self.sequences = {}
        self.timestamps = timestamps or {}
        self.item_categories = item_categories or {}
        self.return_timestamps = return_timestamps
        self.has_timestamps = bool(self.timestamps)
        self.has_categories = bool(self.item_categories)

        for uid, seq in sequences.items():
            if len(seq) >= 2:
                self.users.append(uid)
                self.sequences[uid] = seq

        if not self.has_timestamps:
            logger.warning(
                "EvalDataset: no timestamps provided. TiSASRec/TiSASRec-Cat will "
                "use zero time_deltas, degrading to position-only attention."
            )
        if not self.has_categories:
            logger.warning(
                "EvalDataset: no item_categories provided. TiSASRec-Cat will "
                "use all-False same_cat_mask, disabling category-conditioned bias."
            )

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple:
        uid = self.users[idx]
        seq = self.sequences[uid]
        hist = seq[:-1]          # 除了最后一个，其余作为历史
        target = seq[-1]         # 最后一个作为目标

        # 截断和填充历史
        hist_items = hist[-self.max_len:]
        pad_len = self.max_len - len(hist_items)
        hist_padded = [0] * pad_len + hist_items
        pos = list(range(0, len(hist_padded)))

        result = [
            torch.tensor(hist_padded, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long),
        ]

        # 如果时间戳可用，构建 time_deltas 矩阵
        if self.has_timestamps:
            ts = self.timestamps.get(uid, [])
            # 将时间戳与历史物品对齐（填充前）
            if len(ts) >= len(hist):
                ts_hist = ts[:len(hist)][-self.max_len:]  # 取最后 max_len 个物品
                ts_hist = [0.0] * pad_len + ts_hist       # 前面填充
            else:
                ts_hist = [0.0] * self.max_len

            L = self.max_len
            td = torch.zeros(L, L, dtype=torch.float32)
            for i in range(L):
                for j in range(L):
                    td[i, j] = abs(ts_hist[i] - ts_hist[j])
            result.append(td)

            # RoTE 变体需要原始时间戳；保持 opt-in 方式，以便
            # 历史 EvalDataset 元组形状在默认情况下保持稳定。
            if self.return_timestamps:
                result.append(torch.tensor(ts_hist, dtype=torch.float32))

        # 如果类目可用，构建 same_cat_mask
        if self.has_categories:
            # 获取每个历史物品的类目
            cat_hist = []
            for item in hist_items:
                cat_hist.append(self.item_categories.get(item, -1))
            cat_hist = [-1] * pad_len + cat_hist  # 前面用 -1（不匹配）填充

            L = self.max_len
            scm = torch.zeros(L, L, dtype=torch.bool)
            for i in range(L):
                for j in range(L):
                    if cat_hist[i] >= 0 and cat_hist[j] >= 0 and cat_hist[i] == cat_hist[j]:
                        scm[i, j] = True
            result.append(scm)

        return tuple(result)
