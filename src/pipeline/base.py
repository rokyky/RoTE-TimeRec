"""管道基类与候选数据结构定义。

定义各阶段统一的 Candidate 数据结构和 PipelineStage 接口，
确保召回 → 粗排 → 精排 → 重排 各阶段数据流一致。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


@dataclass
class PipelineStats:
    """记录单个管道阶段的运行统计。

    字段说明：
        stage_name:       阶段名称
        input_candidates:  输入候选总数（跨所有用户）
        output_candidates: 输出候选总数（跨所有用户）
        wall_time_ms:      耗时（毫秒）
        hit_rate:          命中率（如果提供了 ground truth）
    """
    stage_name: str
    input_candidates: int
    output_candidates: int
    wall_time_ms: float
    hit_rate: Optional[float] = None


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

    @property
    def total_candidates(self) -> int:
        """返回所有用户的候选总数。"""
        return sum(len(cands) for cands in self._data.values())

    def __len__(self) -> int:
        return self.total_candidates

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

    def predict_with_stats(self,
                           candidates: CandidateList,
                           context: dict,
                           ground_truth: Optional[Dict[int, int]] = None,
                           ) -> Tuple[CandidateList, PipelineStats]:
        """带统计收集的 predict 包裹方法。

        参数：
            candidates: 输入候选
            context:    上下文信息
            ground_truth: 每个用户的真实目标 item（可选，用于计算命中率）

        返回：
            (处理后的候选列表, 阶段统计)
        """
        input_count = candidates.total_candidates
        t0 = time.perf_counter()
        result = self.predict(candidates, context)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        output_count = result.total_candidates

        # 计算命中率（如果提供了 ground truth）
        hit_rate = None
        if ground_truth:
            hits = 0
            total = 0
            for uid, cands in result.items():
                gt_item = ground_truth.get(uid)
                if gt_item is not None:
                    total += 1
                    if any(c.item_id == gt_item for c in cands):
                        hits += 1
            if total > 0:
                hit_rate = hits / total

        stats = PipelineStats(
            stage_name=self.name,
            input_candidates=input_count,
            output_candidates=output_count,
            wall_time_ms=elapsed_ms,
            hit_rate=hit_rate,
        )
        return result, stats
