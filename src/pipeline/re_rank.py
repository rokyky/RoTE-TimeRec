# Re-ranking stage: diversity + business rules.
# Reference: MMR / DPP algorithms

from typing import List
from .base import Candidate, CandidateList, PipelineStage


class MMRReRank(PipelineStage):
    def __init__(self, keep_k: int = 50, lam: float = 0.5):
        super().__init__('rerank_mmr')
        self.keep_k = keep_k
        self.lam = lam

    def _sim(self, a: Candidate, b: Candidate) -> float:
        return 0.0  # placeholder

    def predict(self, candidates, context) -> CandidateList:
        result = CandidateList()
        for uid, cands in candidates.items():
            pool = sorted(cands, key=lambda x: x.score, reverse=True)
            selected = [pool.pop(0)] if pool else []
            while len(selected) < self.keep_k and pool:
                best_i, best_v = 0, -1e9
                for i, c in enumerate(pool):
                    pen = max(self._sim(c, s) for s in selected)
                    mmr = self.lam * c.score - (1 - self.lam) * pen
                    if mmr > best_v: best_i, best_v = i, mmr
                selected.append(pool.pop(best_i))
            for c in selected:
                c.source = self.name
                result.add(uid, c)
        return result
