# 运行完整推荐管道端到端（带可观测性统计）。

import argparse, logging, os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.pipeline.base import CandidateList, Candidate
from src.pipeline.runner import PipelineRunner, format_stats_table
from src.pipeline.recall import PopularityRecall, ItemCFRecall, DSSMRecall
from src.pipeline.pre_rank import SimplePreRank
from src.pipeline.rank import SequenceRanker
from src.pipeline.re_rank import MMRReRank
from src.models.sasrec import SASRec
from src.models.dssm import DSSM, DSSMDataset, collate_dssm, train_epoch_dssm


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

    # Recall: DSSM
    dssm_cfg = config.get('dssm', {})
    num_users_total = len(sequences)
    dssm_model = DSSM(
        num_users=num_users_total,
        num_items=num_items,
        hidden_dim=config.get('model', {}).get('hidden_dim', 64),
        mlp_dims=dssm_cfg.get('mlp_dims', None),
        dropout=config.get('model', {}).get('dropout', 0.0),
    )
    dssm_model.to(device)

    ds_train = DSSMDataset(train_seq)
    dl_train = DataLoader(
        ds_train, batch_size=dssm_cfg.get('batch_size', 256),
        shuffle=True, collate_fn=collate_dssm,
    )
    opt = torch.optim.AdamW(
        dssm_model.parameters(), lr=dssm_cfg.get('lr', 0.001),
    )
    dssm_epochs = dssm_cfg.get('epochs', 20)
    dssm_temp = dssm_cfg.get('temperature', 0.05)
    logger.info("Training DSSM for %d epochs (lr=%.4f, temp=%.2f, batch=%d)",
                dssm_epochs, dssm_cfg.get('lr', 0.001), dssm_temp,
                dssm_cfg.get('batch_size', 256))
    for epoch in range(1, dssm_epochs + 1):
        loss = train_epoch_dssm(dssm_model, dl_train, opt, device, temperature=dssm_temp)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("DSSM epoch %d/%d, loss=%.4f", epoch, dssm_epochs, loss)

    # 提取 embedding（覆盖所有已知用户）
    dssm_model.eval()
    all_users = set(train_seq.keys()) | set(eval_seq.keys())
    user_emb_map = {}
    with torch.no_grad():
        for uid in all_users:
            u_t = torch.tensor([uid], dtype=torch.long, device=device)
            user_emb_map[uid] = dssm_model.encode_user(u_t).cpu().numpy().flatten().tolist()
    item_emb_map = {}
    all_item_embs = dssm_model.get_all_item_embs().cpu().numpy()
    for iid in range(1, num_items + 1):
        item_emb_map[iid] = all_item_embs[iid].tolist()

    dssm = DSSMRecall(top_k=30)
    dssm.fit(user_emb_map, item_emb_map)
    runner.add_stage("recall_dssm", dssm)

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
