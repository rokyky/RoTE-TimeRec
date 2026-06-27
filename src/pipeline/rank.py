'''Rank stage: sequential model + deep ranking models.

Reference:
    - DeepCTR-Torch -> DeepFM/DIN implementation
    - RecBole -> sequential model eval protocol
'''

import torch
from typing import Dict, List, Optional

from .base import Candidate, CandidateList, PipelineStage


class SequenceRanker(PipelineStage):
    '''TiSASRec-Cat based sequence ranking.

    Calls the core sequential model (copied from 5.time-aware-seqrec)
    to score candidate items for each user.
    '''

    def __init__(self, model, keep_k: int = 100):
        super().__init__('rank_seq')
        self.model = model
        self.keep_k = keep_k

    def predict(self, candidates: CandidateList, context: dict) -> CandidateList:
        result = CandidateList()
        for user_id, cands in candidates.items():
            seq = context['sequences'].get(user_id, [])
            if not seq:
                for c in cands[:self.keep_k]:
                    result.add(user_id, c)
                continue

            item_ids = [c.item_id for c in cands]
            scores = self._score_items(user_id, seq, item_ids)
            scored = []
            for c, s in zip(cands, scores):
                c.score = float(s)
                scored.append(c)
            scored.sort(key=lambda x: x.score, reverse=True)
            for c in scored[:self.keep_k]:
                result.add(user_id, c)
        return result

    def _score_items(self, user_id: int, seq: List[int], items: List[int]) -> List[float]:
        '''Score candidate items using the sequential model.'''
        self.model.eval()
        with torch.no_grad():
            # simplified: would need proper batch processing in practice
            scores = self.model.predict(seq, items)
        return scores.tolist()
