'''Data loading and preprocessing module.

Reference:
    - RecBole data processing pipeline
    - Sequential dataset with negative sampling
'''

import random
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class SeqRecDataset(Dataset):
    def __init__(self,
                 sequences: Dict[int, List[int]],
                 max_len: int = 50,
                 num_items: int = 0,
                 neg_samples: int = 1,
                 item_popularity: Optional[Dict[int, int]] = None):
        self.max_len = max_len
        self.num_items = num_items
        self.neg_samples = neg_samples
        self.item_popularity = item_popularity
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
        return (torch.tensor(hist, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(uid, dtype=torch.long))

class EvalDataset(Dataset):
    def __init__(self,
                 sequences: Dict[int, List[int]],
                 max_len: int = 50):
        self.max_len = max_len
        self.users = []
        self.sequences = {}
        for uid, seq in sequences.items():
            if len(seq) >= 2:
                self.users.append(uid)
                self.sequences[uid] = seq

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> Tuple:
        uid = self.users[idx]
        seq = self.sequences[uid]
        hist = seq[:-1]
        target = seq[-1]
        hist = hist[-self.max_len:]
        pad_len = self.max_len - len(hist)
        hist = [0] * pad_len + hist
        pos = list(range(0, len(hist)))
        return (torch.tensor(hist, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(uid, dtype=torch.long))