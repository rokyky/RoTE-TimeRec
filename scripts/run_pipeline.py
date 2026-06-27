# 运行完整推荐管道端到端（带可观测性统计）。

import argparse, logging, os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.config import load_config
from src.pipeline.base import CandidateList, Candidate
from src.pipeline.runner import PipelineRunner, format_stats_table
from src.pipeline.recall import PopularityRecall, ItemCFRecall
from src.pipeline.pre_rank import SimplePreRank
from src.pipeline.rank import SequenceRanker
from src.pipeline.re_rank import MMRReRank
from src.models.sasrec import SASRec


def generate_synthetic_data(num_users=50, num_items=30, max_len=15, seed=42):
    random.seed(seed)
    sequences = {}
    for uid in range(num_users):
        length = random.randint(3, max_len)
        seq = [random.randint(1, num_items) for _ in range(length)]
        sequences[uid] = seq
    return sequences, num_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-stats", action="store_true",
                        help="不收集阶段统计")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---- 数据 ----
    sequences, num_items = generate_synthetic_data()
    split = int(len(sequences) * 0.8)
    users = list(sequences.keys())
    train_seq = {u: sequences[u] for u in users[:split]}
    eval_seq = {u: sequences[u] for u in users[split:]}

    # 构建 ground truth（用于命中率统计）
    ground_truth = {}
    for uid, seq in eval_seq.items():
        ground_truth[uid] = seq[-1]   # 最后一条为 target

    # 初始化候选
    candidates = CandidateList()
    for uid in eval_seq:
        candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

    # 上下文
    context = {
        "sequences": train_seq,
        "user_history": train_seq,
    }

    # ---- 管道 ----
    collect_stats = not args.no_stats
    runner = PipelineRunner(config, collect_stats=collect_stats)

    # Recall: Popularity
    pop = PopularityRecall(top_k=30)
    item_counts = {}
    for seq in train_seq.values():
        for item in seq:
            item_counts[item] = item_counts.get(item, 0) + 1
    pop.fit(item_counts)
    runner.add_stage("recall_pop", pop)

    # Recall: ItemCF
    itemcf = ItemCFRecall(top_k=30)
    # 简单的共现相似度
    sim_matrix = {}
    for seq in train_seq.values():
        for i, item_a in enumerate(seq):
            if item_a not in sim_matrix:
                sim_matrix[item_a] = {}
            for j, item_b in enumerate(seq):
                if i != j:
                    sim_matrix[item_a][item_b] = sim_matrix[item_a].get(item_b, 0) + 1
    itemcf.fit(sim_matrix)
    runner.add_stage("recall_itemcf", itemcf)

    # PreRank
    pre_rank = SimplePreRank(keep_k=20)
    runner.add_stage("pre_rank", pre_rank)

    # Rank
    model = SASRec(num_items, hidden_dim=16, max_len=15)
    ranker = SequenceRanker(model, keep_k=15)
    runner.add_stage("rank", ranker)

    # ReRank
    rerank = MMRReRank(keep_k=10, lam=0.5)
    runner.add_stage("rerank", rerank)

    # ---- 运行 ----
    result, stats = runner.run(candidates, context, ground_truth=ground_truth)

    total = len(result)
    users_with_cands = sum(1 for uid in eval_seq if len(result.get(uid)) > 0)
    print(f"\n管道完成: {total} 个候选, {users_with_cands}/{len(eval_seq)} 个用户有候选")

    # ---- 统计表格 ----
    if stats:
        print(format_stats_table(stats))
    else:
        print("(统计收集已关闭)")

    return result, stats


if __name__ == "__main__":
    main()
