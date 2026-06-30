'''召回阶段：多路召回 + 融合。

参考：
    - RecBole pop/BPR
    - EasyRec matching 模块
'''

import logging
from typing import Dict, List, Optional

import numpy as np

from .base import Candidate, CandidateList, PipelineStage

logger = logging.getLogger(__name__)


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
    """双塔向量召回 — 使用 Faiss 进行 ANN 搜索。

    fit() 接收 user/item embedding 并构建 Faiss 索引；
    predict() 对每个用户用 Faiss 搜索 top-K 候选。

    Faiss 为可选依赖：未安装时回退到暴力搜索（适合小数据集）。
    """

    def __init__(self, top_k: int = 500):
        super().__init__('recall_dssm')
        self.top_k = top_k
        self.index: Optional[object] = None
        self._index_item_ids: List[int] = []
        self._user_emb: Dict[int, np.ndarray] = {}

    def fit(
        self,
        user_emb: Dict[int, List[float]],
        item_emb: Dict[int, List[float]],
    ) -> None:
        """构建 Faiss 索引。"""
        self._user_emb = {uid: np.array(v, dtype=np.float32) for uid, v in user_emb.items()}

        self._index_item_ids = sorted(item_emb.keys())
        emb_matrix = np.array(
            [item_emb[iid] for iid in self._index_item_ids], dtype=np.float32,
        )

        dim = emb_matrix.shape[1]
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(emb_matrix)
            logger.info(
                "Faiss index built: %d items, dim=%d", self.index.ntotal, dim,
            )
        except ImportError:
            logger.warning(
                "faiss not installed, falling back to brute-force. "
                "Install with: pip install faiss-cpu"
            )
            self.index = None
            self._fallback_embs = emb_matrix

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        result = CandidateList()

        user_ids = [uid for uid in candidates.user_ids if uid in self._user_emb]
        if not user_ids:
            return result

        query = np.array([self._user_emb[uid] for uid in user_ids], dtype=np.float32)

        if self.index is not None:
            import faiss
            scores_mat, indices_mat = self.index.search(query, self.top_k)
            for row_idx, uid in enumerate(user_ids):
                for score, idx in zip(scores_mat[row_idx], indices_mat[row_idx]):
                    item_id = self._index_item_ids[idx]
                    result.add(uid, Candidate(
                        user_id=uid, item_id=item_id,
                        score=float(score), source=self.name,
                    ))
        else:
            emb_matrix = self._fallback_embs
            for row_idx, uid in enumerate(user_ids):
                u = query[row_idx:row_idx + 1]
                scores = emb_matrix @ u.T
                topk = np.argpartition(scores[:, 0], -self.top_k)[-self.top_k:]
                topk = topk[np.argsort(-scores[topk, 0])]
                for idx in topk:
                    item_id = self._index_item_ids[idx]
                    result.add(uid, Candidate(
                        user_id=uid, item_id=item_id,
                        score=float(scores[idx, 0]), source=self.name,
                    ))

        return result
