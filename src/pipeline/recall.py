'''召回阶段：多路召回 + 融合。

参考：
    - RecBole pop/BPR
    - EasyRec matching 模块
'''

from typing import Dict, List
from .base import Candidate, CandidateList, PipelineStage


class PopularityRecall(PipelineStage):
    def __init__(self, top_k: int = 500):
        super().__init__('recall_pop')
        self.top_k = top_k
        self.pop_items: List[int] = []

    def fit(self, item_counts: Dict[int, int]) -> None:
        self.pop_items = sorted(item_counts, key=item_counts.get, reverse=True)

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        result = CandidateList()
        for user_id in candidates.user_ids:
            for rank, item_id in enumerate(self.pop_items[:self.top_k]):
                result.add(user_id, Candidate(
                    user_id=user_id, item_id=item_id,
                    score=1.0 / (rank + 1),
                    source=self.name,
                ))
        return result


class ItemCFRecall(PipelineStage):
    def __init__(self, top_k: int = 500):
        super().__init__('recall_itemcf')
        self.top_k = top_k
        self.sim_matrix: Dict[int, Dict[int, float]] = {}

    def fit(self, item_sim: Dict[int, Dict[int, float]]) -> None:
        self.sim_matrix = item_sim

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        result = CandidateList()
        user_hists: Dict[int, List[int]] = context.get('user_history', {})
        for user_id in candidates.user_ids:
            scored: Dict[int, float] = {}
            for hist_id in user_hists.get(user_id, []):
                if hist_id not in self.sim_matrix:
                    continue
                for sim_id, sim_score in self.sim_matrix[hist_id].items():
                    scored[sim_id] = scored.get(sim_id, 0) + sim_score
            seen = set(user_hists.get(user_id, []))
            for item_id, score in sorted(scored.items(), key=lambda x: -x[1])[:self.top_k]:
                if item_id in seen:
                    continue
                result.add(user_id, Candidate(
                    user_id=user_id, item_id=item_id,
                    score=score, source=self.name,
                ))
        return result


class DSSMRecall(PipelineStage):
    '''双塔向量召回（存根：需要 Faiss 集成）。'''

    def __init__(self, top_k: int = 500):
        super().__init__('recall_dssm')
        self.top_k = top_k

    def fit(self, user_emb: Dict[int, List[float]], item_emb: Dict[int, List[float]]) -> None:
        self.user_emb = user_emb
        self.item_emb = item_emb

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        # 占位：实际场景需使用 Faiss MIP 搜索
        result = CandidateList()
        for user_id in candidates.user_ids:
            if user_id not in self.user_emb:
                continue
            emb = self.user_emb[user_id]
            scores = []
            for item_id, iemb in self.item_emb.items():
                sim = sum(a*b for a, b in zip(emb, iemb))
                scores.append((item_id, sim))
            scores.sort(key=lambda x: -x[1])
            for item_id, score in scores[:self.top_k]:
                result.add(user_id, Candidate(
                    user_id=user_id, item_id=item_id,
                    score=score, source=self.name,
                ))
        return result
