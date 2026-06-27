'''精排阶段：序列模型 + 深度排序模型。

参考：
    - DeepCTR-Torch -> DeepFM/DIN 实现
    - RecBole -> 序列模型评估协议
'''

import torch
from typing import Dict, List, Optional

from .base import Candidate, CandidateList, PipelineStage


class SequenceRanker(PipelineStage):
    '''基于 TiSASRec-Cat 的序列排序。

    调用核心序列模型（复制自 5.time-aware-seqrec）
    为每个用户对候选物品进行打分。
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
        '''使用序列模型对候选物品进行打分。'''
        self.model.eval()
        with torch.no_grad():
            # 简化：实际场景需要适当的批处理
            scores = self.model.predict(seq, items)
        return scores.tolist()
