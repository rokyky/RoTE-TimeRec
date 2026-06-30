"""在 Amazon Beauty 真实数据上训练 DSSM 双塔模型 + Faiss 召回评估。

数据流：
    raw ratings CSV → 5-core 过滤 → 交互序列 → train/val/test 切分
    → DSSM 训练（in-batch negative softmax）→ 提取 embedding
    → Faiss 建索引 → Recall@K 评估
"""

import argparse, csv, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dssm import DSSM, DSSMDataset, collate_dssm, train_epoch_bpr
from src.pipeline.recall import DSSMRecall

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_amazon_beauty(csv_path: str):
    """加载 ratings_Beauty.csv → (user_id, item_id, timestamp) 三元组列表。

    CSV 格式: userId,itemId,rating,timestamp（无表头）。
    过滤 rating >= 4 作为 positive 交互。
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 4:
                continue
            uid, iid, rating, ts = line[0], line[1], float(line[2]), int(line[3])
            # 标准处理方式：保留 rating >= 4 的交互
            if rating >= 4:
                rows.append((uid, iid, ts))
    logger.info("Loaded %d interactions (rating>=4) from %s", len(rows), csv_path)
    return rows


def filter_k_core(rows, k=5):
    """递归 5-core 过滤：每个 user 和 item 至少 k 次交互。"""
    while True:
        user_counts = {}
        item_counts = {}
        for uid, iid, _ in rows:
            user_counts[uid] = user_counts.get(uid, 0) + 1
            item_counts[iid] = item_counts.get(iid, 0) + 1
        valid_users = {u for u, c in user_counts.items() if c >= k}
        valid_items = {i for i, c in item_counts.items() if c >= k}
        new_rows = [(u, i, t) for u, i, t in rows if u in valid_users and i in valid_items]
        if len(new_rows) == len(rows):
            break
        rows = new_rows
    logger.info("After %d-core: %d users, %d items, %d interactions", k,
                len(valid_users), len(valid_items), len(rows))
    return rows


def build_sequences(rows):
    """按 user 分组、按时间戳排序，构建交互序列。

    返回:
        sequences: idx_user → [item_id, ...]
        user_map: str_uid → idx
        item_map: str_iid → idx
    """
    user_interactions = {}
    for uid, iid, ts in rows:
        user_interactions.setdefault(uid, []).append((iid, ts))

    # 每个用户按时间戳排序
    user_ids = sorted(user_interactions.keys())
    item_set = set()
    sequences = {}
    for raw_uid in user_ids:
        items_ts = sorted(user_interactions[raw_uid], key=lambda x: x[1])
        seq = []
        for iid, _ in items_ts:
            item_set.add(iid)
            seq.append(iid)
        sequences[raw_uid] = seq

    # 编码为整数 ID
    user_map = {u: i for i, u in enumerate(sorted(user_ids))}
    item_list = sorted(item_set)
    item_map = {i: idx + 1 for idx, i in enumerate(item_list)}  # 0 为 padding

    mapped_sequences = {}
    for raw_uid, seq in sequences.items():
        uid = user_map[raw_uid]
        mapped_sequences[uid] = [item_map[iid] for iid in seq]

    return mapped_sequences, user_map, item_map



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/beauty/raw/ratings_Beauty.csv")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-mlp", action="store_true", help="去掉 MLP，仅用 embedding 点积")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    csv_path = os.path.join(os.path.dirname(__file__), "..", args.csv)
    # ---- 1. 加载 + 过滤 ----
    rows = load_amazon_beauty(csv_path)
    rows = filter_k_core(rows, k=5)
    sequences, user_map, item_map = build_sequences(rows)
    num_users = len(sequences)
    num_items = max(item_map.values())
    logger.info("Sequences: %d users, %d items", num_users, num_items)

    # ---- 2. 切分 (leave-one-out per user, all users in train) ----
    # DSSM 需要所有用户的 embedding 都训练过才能做评估
    # 每用户最后一跳做 target，前面的做训练
    train_seq = {}
    val_ground_truth = {}
    for uid in sorted(sequences.keys()):
        seq = sequences[uid]
        if len(seq) >= 2:
            train_seq[uid] = seq[:-1]
            val_ground_truth[uid] = seq[-1]
        else:
            train_seq[uid] = seq

    logger.info("Total users: %d (all in training), val targets: %d",
                len(train_seq), len(val_ground_truth))
    logger.info("Total train interactions: %d",
                sum(len(s) for s in train_seq.values()))

    # ---- 3. 训练 DSSM ----
    model = DSSM(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        mlp_dims=args.mlp_dims,
        dropout=0.1,
        no_mlp=args.no_mlp,
    ).to(device)
    logger.info("DSSM params: %d", sum(p.numel() for p in model.parameters()))

    ds = DSSMDataset(train_seq)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_dssm, num_workers=0, drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch_bpr(model, dl, optimizer, device, num_neg=200)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            logger.info("Epoch %3d/%d  loss=%.4f  elapsed=%ds",
                        epoch, args.epochs, loss, int(elapsed))
    total_time = time.time() - t0
    logger.info("Training done: %.1fs total (%.2fs/epoch)", total_time, total_time / args.epochs)

    # ---- 5. Faiss 召回评估（排除已见物品） ----
    model.eval()
    with torch.no_grad():
        user_emb_map = {}
        for uid in train_seq:
            u_t = torch.tensor([uid], dtype=torch.long, device=device)
            user_emb_map[uid] = model.encode_user(u_t).cpu().numpy().flatten().tolist()

        all_item_embs = model.get_all_item_embs().cpu().numpy()
        item_emb_map = {iid: all_item_embs[iid].tolist() for iid in range(1, num_items + 1)}

    recall_stage = DSSMRecall(top_k=args.top_k + 100)  # 多召一些留出过滤空间
    recall_stage.fit(user_emb_map, item_emb_map)

    from src.pipeline.base import CandidateList, Candidate
    candidates = CandidateList()
    for uid in val_ground_truth:
        candidates.add(uid, Candidate(uid, 0, 0.0, {}, "init"))

    result = recall_stage.predict(candidates, {})

    # 过滤已见物品 + 计算 Recall@K
    def recall_at_k_exclude_seen(ground_truth, predictions, seen_items, k):
        hits = 0
        total = 0
        for uid, gt_item in ground_truth.items():
            preds = predictions.get(uid, [])
            seen = set(seen_items.get(uid, []))
            # 过滤掉已见物品
            filtered = [p for p in preds if p not in seen]
            total += 1
            if gt_item in filtered[:k]:
                hits += 1
        return hits / total

    predictions = {}
    for uid in val_ground_truth:
        cands = result.get(uid)
        if cands:
            predictions[uid] = [c.item_id for c in sorted(cands, key=lambda x: -x.score)]

    seen_items = train_seq  # 训练集所有物品为已见

    for k in [1, 5, 10, 20, 50]:
        if k > args.top_k:
            break
        recall = recall_at_k_exclude_seen(val_ground_truth, predictions, seen_items, k)
        logger.info("Recall@%2d (excl. seen) = %.4f", k, recall)

    total_cands = sum(len(predictions.get(uid, [])) for uid in val_ground_truth)
    users_with_cands = sum(1 for uid in val_ground_truth if predictions.get(uid))
    print(f"\nResults: {users_with_cands}/{len(val_ground_truth)} users with candidates, "
          f"{total_cands} total candidates, "
          f"mean {total_cands/len(val_ground_truth):.0f}/user")


if __name__ == "__main__":
    main()
