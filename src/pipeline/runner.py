# PipelineRunner: 编排多阶段管道，可选统计收集。

from typing import Dict, List, Optional, Tuple
from .base import CandidateList, PipelineStats


class PipelineRunner:
    def __init__(self, config=None, collect_stats: bool = False):
        self.config = config or {}
        self.collect_stats = collect_stats
        self.stages = {}
        self.stage_order = []

    def add_stage(self, name, stage):
        if name not in self.stages:
            self.stage_order.append(name)
        self.stages[name] = stage

    def _is_recall(self, name):
        return name == 'recall' or name.startswith('recall_')

    def run(self, candidates, context,
            ground_truth: Optional[Dict[int, int]] = None
            ) -> Tuple[CandidateList, List[PipelineStats]]:
        """运行管道。

        参数：
            candidates: 初始候选列表
            context:    上下文（序列、特征等）
            ground_truth: 每个用户的真实目标 item（用于命中率统计）

        返回：
            (最终候选列表, 阶段统计列表)
        """
        recall_names = [x for x in self.stage_order if self._is_recall(x)]
        other_names = [x for x in self.stage_order if not self._is_recall(x)]
        stats_list = []
        result = candidates

        if recall_names:
            merged = CandidateList()
            for name in recall_names:
                stage = self.stages[name]
                if self.collect_stats:
                    out, stat = stage.predict_with_stats(result, context, ground_truth)
                    stats_list.append(stat)
                else:
                    out = stage.predict(result, context)

                for uid, cands in out.items():
                    seen = {c.item_id for c in merged.get(uid)}
                    for c in cands:
                        if c.item_id not in seen:
                            merged.add(uid, c)
            result = merged

        for name in other_names:
            stage = self.stages[name]
            if self.collect_stats:
                result, stat = stage.predict_with_stats(result, context, ground_truth)
                stats_list.append(stat)
            else:
                result = stage.predict(result, context)

        return result, stats_list


def format_stats_table(stats: List[PipelineStats]) -> str:
    """将统计列表格式化为可读表格。"""
    header = f"{'阶段':<20} {'入候选':>8} {'出候选':>8} {'耗时(ms)':>10} {'命中率':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for s in stats:
        hr = f"{s.hit_rate:.4f}" if s.hit_rate is not None else "N/A"
        lines.append(
            f"{s.stage_name:<20} {s.input_candidates:>8} "
            f"{s.output_candidates:>8} {s.wall_time_ms:>10.2f} {hr:>8}"
        )
    lines.append(sep)
    return "\n".join(lines)
