import numpy as np


class PopularityModel:
    """基于流行度的推荐模型。

    根据训练集中的物品频率排序。
    作为下界 baseline（基线）。
    """

    def __init__(self, item2idx=None):
        self.pop_items = []
        self.item_freq = {}
        self.item2idx = item2idx or {}

    def fit(self, train_df, item_col='asin', item2idx=None):
        """统计训练数据中的物品频率。

        Args:
            train_df: 训练 DataFrame
            item_col: 物品 ID 列名
            item2idx: 可选，物品 ID → 索引映射。若提供则替代构造时的映射。
        """
        self.item_freq = train_df[item_col].value_counts().to_dict()
        self.pop_items = list(self.item_freq.keys())
        if item2idx is not None:
            self.item2idx = item2idx
        return self

    def set_item2idx(self, item2idx):
        """设置物品 ID 到索引的映射（用于 get_top_k）。"""
        self.item2idx = item2idx

    def predict(self, user_idx=None):
        """返回所有物品的流行度排名得分。

        返回：
            scores: 得分列表/数组，越高表示越流行
        """
        return None  # 得分通过 pop_items 顺序在外部处理

    def get_pop_items(self):
        """按流行度降序返回物品索引。"""
        return self.pop_items

    def get_top_k(self, k=10):
        """返回 top-k 流行物品的索引。"""
        if not self.item2idx:
            return []
        return [self.item2idx.get(item, 0) for item in self.pop_items[:k]
                if item in self.item2idx]
