"""粗排阶段：轻量模型剪枝候选集。

参考：
    - Feed_Recommender → 粗剪策略（从千级别剪到百级别）
    - EasyRec pre-ranking → 简单双塔 / 小 MLP
"""

import math
from typing import List
import numpy as np

from .base import Candidate, CandidateList, PipelineStage


class SimplePreRank(PipelineStage):
    """简易粗排：用少量统计特征 + 线性模型剪枝。"""

    def __init__(self, keep_k: int = 200):
        super().__init__("pre_rank_simple")
        self.keep_k = keep_k

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        """用统计特征给候选打一个粗排分，保留 top-K。"""
        result = CandidateList()
        for user_id, cands in candidates.items():
            scored: List[Candidate] = []
            for c in cands:
                # 粗排分 = 原始分 * pop_bonus
                pop_bonus = self._pop_bonus(c.item_id, context)
                c.score = c.score * pop_bonus
                scored.append(c)
            scored.sort(key=lambda x: x.score, reverse=True)
            for c in scored[:self.keep_k]:
                result.add(user_id, c)
        return result

    def _pop_bonus(self, item_id: int, context: dict) -> float:
        item_pop = context.get("item_popularity", {}).get(item_id, 1)
        return math.log(item_pop + 1) ** 0.5
