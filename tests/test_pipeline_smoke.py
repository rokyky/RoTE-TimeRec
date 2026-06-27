'''管道端到端冒烟测试：recall → pre-rank → rank → re-rank 无错误完成。'''

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import random

from src.pipeline.base import CandidateList, Candidate
from src.pipeline.runner import PipelineRunner
from src.pipeline.recall import PopularityRecall, ItemCFRecall
from src.pipeline.pre_rank import SimplePreRank
from src.pipeline.rank import SequenceRanker
from src.pipeline.re_rank import MMRReRank
from src.models.sasrec import SASRec


def _make_synthetic_data(num_users=20, num_items=15, max_len=10, seed=42):
    '''构造测试用合成数据。'''
    random.seed(seed)
    sequences = {}
    for uid in range(num_users):
        length = random.randint(3, max_len)
        seq = [random.randint(1, num_items) for _ in range(length)]
        sequences[uid] = seq
    return sequences, num_items


class TestPipelineSmoke:
    def test_end_to_end(self):
        '''全链路：recall → pre-rank → rank → re-rank 无异常完成。'''
        sequences, num_items = _make_synthetic_data()

        # 切分 train/eval
        split = int(len(sequences) * 0.8)
        users = list(sequences.keys())
        train_seq = {u: sequences[u] for u in users[:split]}
        eval_seq = {u: sequences[u] for u in users[split:]}

        # 初始化候选
        candidates = CandidateList()
        for uid in eval_seq:
            candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

        context = {
            "sequences": train_seq,
            "user_history": train_seq,
        }

        # 构建 pipeline
        runner = PipelineRunner()

        # Recall: Popularity
        pop = PopularityRecall(top_k=30)
        item_counts = {}
        for seq in train_seq.values():
            for item in seq:
                item_counts[item] = item_counts.get(item, 0) + 1
        pop.fit(item_counts)
        runner.add_stage("recall_pop", pop)

        # PreRank
        pre_rank = SimplePreRank(keep_k=20)
        runner.add_stage("pre_rank", pre_rank)

        # Rank
        model = SASRec(num_items, hidden_dim=16, max_len=10)
        ranker = SequenceRanker(model, keep_k=15)
        runner.add_stage("rank", ranker)

        # ReRank
        rerank = MMRReRank(keep_k=10, lam=0.5)
        runner.add_stage("rerank", rerank)

        # 运行
        result, _stats = runner.run(candidates, context)

        # 验证输出结构
        assert isinstance(result, CandidateList)
        assert len(result.user_ids) == len(eval_seq), (
            f"应有 {len(eval_seq)} 个用户有候选，实际 {len(result.user_ids)}"
        )
        for uid in eval_seq:
            cands = result.get(uid)
            assert len(cands) > 0, f"用户 {uid} 无候选"
            for c in cands:
                assert isinstance(c.item_id, int)
                assert c.item_id >= 1  # 不应该有 padding item

    def test_output_structure(self):
        '''验证输出 CandidateList 结构正确。'''
        sequences, num_items = _make_synthetic_data()
        eval_seq = {u: sequences[u] for u in list(sequences.keys())[-5:]}

        candidates = CandidateList()
        for uid in eval_seq:
            candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

        context = {"sequences": {}, "user_history": {}}

        runner = PipelineRunner()
        pop = PopularityRecall(top_k=10)
        item_counts = {i: 1 for i in range(1, num_items + 1)}
        pop.fit(item_counts)
        runner.add_stage("recall", pop)

        result, _stats = runner.run(candidates, context)

        # 结构检查
        for uid in eval_seq:
            cands = result.get(uid)
            # 每个用户应有 top_k=10 个候选
            assert len(cands) <= 10
            # 候选应按分数降序排列
            for i in range(len(cands) - 1):
                assert cands[i].score >= cands[i + 1].score, (
                    f"用户 {uid} 的候选未按分数降序排列"
                )

    def test_multi_recall_merge(self):
        '''多路召回合并：无重复候选。'''
        sequences, num_items = _make_synthetic_data()
        eval_seq = {0: sequences[0]}

        candidates = CandidateList()
        for uid in eval_seq:
            candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

        context = {"sequences": sequences, "user_history": sequences}

        runner = PipelineRunner()

        # 两路相同配置的召回（会产出重叠候选）
        item_counts = {i: i for i in range(1, num_items + 1)}
        pop1 = PopularityRecall(top_k=5)
        pop1.fit(item_counts)
        pop2 = PopularityRecall(top_k=5)
        pop2.fit(item_counts)

        runner.add_stage("recall_1", pop1)
        runner.add_stage("recall_2", pop2)

        result, _stats = runner.run(candidates, context)

        # 两路召回合并后不应有重复 item
        for uid in eval_seq:
            cands = result.get(uid)
            item_ids = [c.item_id for c in cands]
            assert len(item_ids) == len(set(item_ids)), (
                f"多路召回合并后存在重复候选: {item_ids}"
            )
