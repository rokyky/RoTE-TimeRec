"""ItemCF：基于物品协同过滤的推荐 baseline。

通过训练集中的物品共现关系计算物品间相似度，
然后根据用户历史交互物品的相似物品进行推荐。

适用于作为非学习基线（无参数训练）。
"""

import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix


class ItemCF:
    """基于物品协同过滤的推荐模型。

    用法：
        model = ItemCF(num_items)
        model.fit(train_df)
        scores = model.predict(user_history_items)
    """

    def __init__(self, num_items):
        self.num_items = num_items
        self.sim_matrix = None  # (num_items+1, num_items+1) 稀疏矩阵，0 为 padding
        self.is_fitted = False

    def fit(self, train_df, item2idx=None, user_col='reviewerID', item_col='asin',
            min_sim_items=2):
        """在训练数据上拟合 ItemCF 模型。

        基于 co-occurrence 构建 item-item 相似度矩阵。
        相似度度量：cosine similarity over user co-occurrence vectors。

        参数：
            train_df: 训练 DataFrame，必须包含 user_col 和 item_col
            item2idx: dict, asin -> item index (1-based)，若提供则将原始 ID 映射为索引
            user_col: 用户 ID 列名
            item_col: 物品 ID 列名
            item_col: 物品 ID 列名
            min_sim_items: 相似度非零的物品对至少需要共现在多少用户中
        """
        # 构建 user-item 交互矩阵 (稀疏)
        users = train_df[user_col].unique()
        user2idx = {u: i for i, u in enumerate(users)}
        num_users = len(users)

        # 收集 interactions
        user_item_pairs = defaultdict(set)
        for _, row in train_df.iterrows():
            u = user2idx.get(row[user_col])
            if u is None:
                continue
            if item2idx is not None:
                i = item2idx.get(row[item_col], 0)
            else:
                i = row[item_col]
            if i > 0:
                user_item_pairs[u].add(i)

        # 构建 item co-occurrence 矩阵
        # 使用 scipy.sparse.lil_matrix 逐步构建
        cooc = lil_matrix((self.num_items + 1, self.num_items + 1), dtype=np.float32)

        for u, items in user_item_pairs.items():
            items_list = list(items)
            n = len(items_list)
            if n < 2:
                continue
            for i in range(n):
                ii = items_list[i]
                for j in range(i + 1, n):
                    jj = items_list[j]
                    cooc[ii, jj] += 1.0
                    cooc[jj, ii] += 1.0

        cooc_csr = cooc.tocsr()

        # 计算 item 频率（用于归一化）
        item_freq = np.zeros(self.num_items + 1, dtype=np.float32)
        for items in user_item_pairs.values():
            for i in items:
                item_freq[i] += 1.0

        # 计算 cosine similarity
        # sim(i,j) = cooc(i,j) / sqrt(freq(i) * freq(j))
        rows, cols = cooc_csr.nonzero()
        data = cooc_csr.data
        sim_data = np.array(data, dtype=np.float32)
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            norm = np.sqrt(item_freq[i] * item_freq[j])
            if norm > 0:
                sim_data[idx] = data[idx] / norm
            else:
                sim_data[idx] = 0.0

        # 过滤低相似度
        if min_sim_items > 1:
            keep = sim_data >= (1.0 / min_sim_items)
            sim_data = sim_data[keep]
            rows = rows[keep]
            cols = cols[keep]

        self.sim_matrix = csr_matrix(
            (sim_data, (rows, cols)),
            shape=(self.num_items + 1, self.num_items + 1),
            dtype=np.float32,
        )
        self.is_fitted = True
        return self

    def predict(self, user_history):
        """根据用户历史交互物品生成推荐得分。

        参数：
            user_history: list of int, 用户历史物品索引（1-based，0 为 padding）

        返回：
            scores: np.ndarray, shape (num_items+1,), 每个候选物品的得分
        """
        assert self.is_fitted, "ItemCF not fitted yet"

        scores = np.zeros(self.num_items + 1, dtype=np.float32)

        # 过滤 padding
        valid_items = [i for i in user_history if i > 0 and i <= self.num_items]
        if not valid_items:
            return scores

        # 聚合每个历史 item 的相似 item 得分
        for item in valid_items:
            row = self.sim_matrix[item].toarray().flatten()
            scores += row

        # 排除历史中已交互的 item
        for item in valid_items:
            scores[item] = 0.0

        return scores

    def predict_batch(self, user_histories):
        """批量预测。

        参数：
            user_histories: list of list of int, 每个用户的历史物品列表

        返回：
            all_scores: np.ndarray, shape (len(user_histories), num_items+1)
        """
        all_scores = np.zeros((len(user_histories), self.num_items + 1), dtype=np.float32)
        for i, history in enumerate(user_histories):
            all_scores[i] = self.predict(history)
        return all_scores
