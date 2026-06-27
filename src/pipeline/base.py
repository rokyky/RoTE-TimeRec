"""Pipeline 基类与候选数据结构定义。

定义各阶段统一的 Candidate 数据结构和 PipelineStage 接口，
确保召回 → 粗排 → 精排 → 重排 各阶段数据流一致。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class Candidate:
    """候选物品的中间表示，在流水线各阶段之间传递。

    字段说明：
        user_id: 用户 ID
        item_id: 物品 ID
        score:   当前阶段对该候选的打分
        features: 特征字典（各阶段可附加特征）
        source:   来源阶段（如 "recall_pop", "rank_tisasrec_cat"）
    """
    user_id: int
    item_id: int
    score: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)
    source: str = ""


class CandidateList:
    """一批用户的候选列表，统一各阶段的输入/输出格式。"""

    def __init__(self, candidates: Optional[Dict[int, List[Candidate]]] = None):
        self._data: Dict[int, List[Candidate]] = candidates or {}

    def add(self, user_id: int, candidate: Candidate) -> None:
        self._data.setdefault(user_id, []).append(candidate)

    def extend(self, user_id: int, candidates: List[Candidate]) -> None:
        self._data.setdefault(user_id, []).extend(candidates)

    def get(self, user_id: int) -> List[Candidate]:
        return self._data.get(user_id, [])

    def items(self):
        return self._data.items()

    @property
    def user_ids(self):
        return list(self._data.keys())

    def __len__(self) -> int:
        return sum(len(cands) for cands in self._data.values())

    def merge(self, other: "CandidateList") -> "CandidateList":
        """合并另一个 CandidateList（用于多路召回合并）。"""
        merged = CandidateList()
        for uid, cands in self._data.items():
            merged._data[uid] = list(cands)
        for uid, cands in other._data.items():
            merged._data.setdefault(uid, []).extend(cands)
        return merged

    def truncate(self, k: int) -> "CandidateList":
        """每个用户截断到 top-K。"""
        truncated = CandidateList()
        for uid, cands in self._data.items():
            sorted_cands = sorted(cands, key=lambda c: c.score, reverse=True)
            truncated._data[uid] = sorted_cands[:k]
        return truncated

    def sort_by_score(self, user_id: int) -> List[Candidate]:
        return sorted(self._data.get(user_id, []), key=lambda c: c.score, reverse=True)


class PipelineStage:
    """流水线阶段的基类。

    子类必须实现 predict() 方法。
    """

    def __init__(self, name: str):
        self.name = name

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        """对候选列表进行打分/过滤/重排序。

        参数：
            candidates: 输入候选
            context:    上下文信息（用户特征、物品特征、配置等）

        返回：
            处理后的候选列表
        """
        raise NotImplementedError
