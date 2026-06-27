# PipelineRunner: orchestrate multi-stage pipeline.

from .base import CandidateList

class PipelineRunner:
    def __init__(self, config=None):
        self.config = config or {}
        self.stages = {}
        self.stage_order = []

    def add_stage(self, name, stage):
        if name not in self.stages:
            self.stage_order.append(name)
        self.stages[name] = stage

    def _is_recall(self, name):
        return name == 'recall' or name.startswith('recall_')

    def run(self, candidates, context):
        recall_names = [x for x in self.stage_order if self._is_recall(x)]
        other_names = [x for x in self.stage_order if not self._is_recall(x)]
        result = candidates
        if recall_names:
            merged = CandidateList()
            for name in recall_names:
                out = self.stages[name].predict(result, context)
                for uid, cands in out.items():
                    seen = {c.item_id for c in merged.get(uid)}
                    for c in cands:
                        if c.item_id not in seen: merged.add(uid, c)
            result = merged

        for name in other_names:
            result = self.stages[name].predict(result, context)
        return result