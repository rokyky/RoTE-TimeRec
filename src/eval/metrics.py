'''序列推荐的评估指标。

参考：
    - RecBole 全排序评估
    - Recall@K / NDCG@K / MRR@K
    - 长尾分析的分桶评估
    - 排除训练已交互物品的全排序
'''

import math
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def recall_at_k(ranked, ground_truth, k):
    if not ground_truth:
        return 0.0
    hits = len(set(ranked[:k]) & set(ground_truth))
    return hits / len(ground_truth)


def ndcg_at_k(ranked, ground_truth, k):
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / ideal if ideal > 0 else 0.0


def mrr_at_k(ranked, ground_truth, k):
    for i, item in enumerate(ranked[:k]):
        if item in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_full_sort(scores, ground_truth, ks=None, exclude_items=None):
    '''带可选物品排除的全排序评估。

    参数：
        scores: (batch, num_items) 预测分数
        ground_truth: List[List[int]] 每用户的真实目标 item
        ks: K 值列表
        exclude_items: 要排除的 item ID 集合（如 item 0 或训练集已交互 item）

    返回：
        metrics dict: recall@K / ndcg@K
    '''
    if ks is None:
        ks = [1, 5, 10, 20]

    # 掩盖排除的物品：将分数设为 -inf，确保它们不会出现在 top-K 中
    if exclude_items:
        for item_id in exclude_items:
            if 0 <= item_id < scores.size(1):
                scores[:, item_id] = -float('inf')

    results = {}
    _, ranked = torch.topk(scores, k=max(ks), dim=1)
    ranked = ranked.cpu().tolist()
    for i, gt in enumerate(ground_truth):
        for k in ks:
            results.setdefault(f'recall@{k}', []).append(recall_at_k(ranked[i], gt, k))
            results.setdefault(f'ndcg@{k}', []).append(ndcg_at_k(ranked[i], gt, k))
    return {k: sum(v) / len(v) for k, v in results.items()}


def _build_time_deltas_from_hist(hist, item_timestamps, device):
    '''从序列中各位置物品的时间戳构建 time_deltas 矩阵。

    参数：
        hist: (B, L) 序列物品 ID
        item_timestamps: Dict[int, float] 物品到最后交互时间（Unix 秒）
                         或 None（回退为零）
        device: torch device

    返回：
        time_deltas: (B, L, L) 成对时间差（秒），无时间戳时为全零
    '''
    B, L = hist.shape
    if item_timestamps is None:
        logger.warning(
            "_build_time_deltas_from_hist: 未提供 item_timestamps，"
            "使用零 time_deltas。TiSASRec 退化为仅位置注意力。"
        )
        return torch.zeros(B, L, L, device=device)

    td = torch.zeros(B, L, L, device=device)
    for b in range(B):
        ts_list = [item_timestamps.get(hist[b, j].item(), 0.0) for j in range(L)]
        for i in range(L):
            for j in range(L):
                td[b, i, j] = abs(ts_list[i] - ts_list[j])
    return td


def _build_same_cat_mask_from_hist(hist, item_categories, device):
    '''从序列中各位置物品的类目构建 same_cat_mask。

    参数：
        hist: (B, L) 序列物品 ID
        item_categories: Dict[int, int] item_id → category_id
                         或 None（回退为全 False）
        device: torch device

    返回：
        same_cat_mask: (B, L, L) 布尔张量，True = 同叶子类目
                       无类目时为全 False
    '''
    B, L = hist.shape
    if item_categories is None:
        logger.warning(
            "_build_same_cat_mask_from_hist: 未提供 item_categories，"
            "使用全 False same_cat_mask。TiSASRec-Cat 退化为 TiSASRec。"
        )
        return torch.zeros(B, L, L, dtype=torch.bool, device=device)

    scm = torch.zeros(B, L, L, dtype=torch.bool, device=device)
    for b in range(B):
        cats = [item_categories.get(hist[b, j].item(), -1) for j in range(L)]
        for i in range(L):
            for j in range(L):
                if cats[i] >= 0 and cats[j] >= 0 and cats[i] == cats[j]:
                    scm[b, i, j] = True
    return scm


def _model_forward(model, hist, pos, device,
                   time_deltas=None, same_cat_mask=None, timestamps=None):
    '''统一的模型前向传播，处理不同类型模型的输入差异。

    TiSASRec:    需要 time_deltas (B,L,L)
    TiSASRec-Cat: 需要 time_deltas + same_cat_mask (B,L,L) bool
    SASRec:       仅需 hist + pos
    SASRec-RoTE:  需要 timestamps (B,L)
    TiSASRec-RoTE:需要 time_deltas + timestamps

    time_deltas/same_cat_mask/timestamps 为 None 时使用零占位并发出 warning。
    '''
    from src.models.tisasrec import TiSASRec
    from src.models.tisasrec_cat import TiSASRecCat
    from src.models.sasrec_rote import SASRecRoTE
    from src.models.tisasrec_rote import TiSASRecRoTE

    B, L = hist.shape

    if isinstance(model, TiSASRecCat):
        if time_deltas is None:
            logger.warning(
                "_model_forward: TiSASRecCat 收到 time_deltas=None，"
                "使用零值。时间感知注意力已禁用。"
            )
            td = torch.zeros(B, L, L, device=device)
        else:
            td = time_deltas

        if same_cat_mask is None:
            logger.warning(
                "_model_forward: TiSASRecCat 收到 same_cat_mask=None，"
                "使用全 False 掩码。类别条件偏置已禁用。"
            )
            cm = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        else:
            cm = same_cat_mask

        ts = timestamps.to(device) if timestamps is not None else None
        if ts is not None:
            return model(hist, pos, td, cm, timestamps=ts)
        return model(hist, pos, td, cm)

    elif isinstance(model, TiSASRecRoTE):
        if time_deltas is None:
            logger.warning(
                "_model_forward: TiSASRecRoTE 收到 time_deltas=None，"
                "使用零值。时间感知注意力已禁用。"
            )
            td = torch.zeros(B, L, L, device=device)
        else:
            td = time_deltas

        ts = timestamps.to(device) if timestamps is not None else None
        return model(hist, pos, td, timestamps=ts)

    elif isinstance(model, TiSASRec):
        if time_deltas is None:
            logger.warning(
                "_model_forward: TiSASRec 收到 time_deltas=None，"
                "使用零值。时间感知注意力已禁用。"
            )
            td = torch.zeros(B, L, L, device=device)
        else:
            td = time_deltas
        return model(hist, pos, td)

    elif isinstance(model, SASRecRoTE):
        ts = timestamps.to(device) if timestamps is not None else None
        return model(hist, pos, timestamps=ts)

    else:
        return model(hist, pos)


def model_eval(model, eval_loader, device, ks=None,
               exclude_items=None,
               item_categories=None,
               item_timestamps=None):
    '''评估序列推荐模型。

    参数：
        model: SASRec / TiSASRec / TiSASRec-Cat / SASRecRoTE / TiSASRecRoTE
        eval_loader: DataLoader，batch 为 4/5/6/7 元组
        device: torch device
        ks: K 值列表，默认 [1, 5, 10, 20]
        exclude_items: 要排除的 item ID 集合（如 {0} 或训练集已交互 item）
        item_categories: Dict[int, int] item_id → category_id（可选）
        item_timestamps: Dict[int, float] 每个 item 的最后交互时间戳（可选）
        user_timestamps: Dict[int, List[float]] 每个用户的交互时间戳序列（可选）

    返回：
        metrics dict: recall@K / ndcg@K
    '''
    if ks is None:
        ks = [1, 5, 10, 20]

    # 默认排除物品 0（填充）— 除非显式覆盖
    if exclude_items is None:
        exclude_items = {0}

    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch_len = len(batch)
            hist, pos = batch[0].to(device), batch[1].to(device)
            target = batch[2]

            # 从 batch 或外部数据确定 time_deltas, same_cat_mask, timestamps
            time_deltas = None
            same_cat_mask = None
            timestamps = None

            for extra in batch[4:]:
                if extra.dtype == torch.bool:
                    same_cat_mask = extra.to(device)
                elif extra.dim() == 3:
                    time_deltas = extra.to(device)
                elif extra.dim() == 2:
                    timestamps = extra.to(device)

            # 如果 batch 不包含这些，但外部数据提供了，则构建它们
            if time_deltas is None and item_timestamps is not None:
                time_deltas = _build_time_deltas_from_hist(hist, item_timestamps, device)

            if same_cat_mask is None and item_categories is not None:
                same_cat_mask = _build_same_cat_mask_from_hist(hist, item_categories, device)

            scores = _model_forward(model, hist, pos, device,
                                    time_deltas=time_deltas,
                                    same_cat_mask=same_cat_mask,
                                    timestamps=timestamps)
            all_scores.append(scores)
            all_targets.append(target)

    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    ground_truth = [[t.item()] for t in targets]

    return evaluate_full_sort(scores, ground_truth, ks, exclude_items=exclude_items)


def evaluate_by_bucket(model, eval_loader, device, bucket_info, ks=None,
                       exclude_items=None):
    if ks is None:
        ks = [5, 10, 20]
    if exclude_items is None:
        exclude_items = {0}

    model.eval()
    bucket_results = {}
    with torch.no_grad():
        for batch in eval_loader:
            hist = batch[0].to(device)
            pos = batch[1].to(device)
            target = batch[2]
            uid = batch[3]

            scores = _model_forward(model, hist, pos, device)

            # 应用排除
            for item_id in exclude_items:
                if 0 <= item_id < scores.size(1):
                    scores[:, item_id] = -float('inf')

            _, ranked = torch.topk(scores, k=max(ks), dim=1)
            ranked = ranked.cpu().tolist()
            for i, uid_i in enumerate(uid):
                uid_i = uid_i.item()
                bucket = bucket_info.get(uid_i, 'unknown')
                if bucket not in bucket_results:
                    bucket_results[bucket] = {'recall': {k: [] for k in ks},
                                              'ndcg': {k: [] for k in ks}}
                gt = [target[i].item()]
                for k in ks:
                    bucket_results[bucket]['recall'][k].append(
                        recall_at_k(ranked[i], gt, k))
                    bucket_results[bucket]['ndcg'][k].append(
                        ndcg_at_k(ranked[i], gt, k))

    summary = {}
    for bucket, metrics in bucket_results.items():
        summary[bucket] = {}
        for k in ks:
            r = metrics['recall'][k]
            n = metrics['ndcg'][k]
            summary[bucket][f'recall@{k}'] = sum(r) / len(r) if r else 0.0
            summary[bucket][f'ndcg@{k}'] = sum(n) / len(n) if n else 0.0
    return summary


def compute_popularity_buckets(user_sequences, item_popularity):
    bucket_info = {}
    pops = list(item_popularity.values()) if item_popularity else [1]
    if pops:
        p66 = np.percentile(pops, 66) if len(pops) > 1 else 1
        p33 = np.percentile(pops, 33) if len(pops) > 1 else 0
    else:
        p66, p33 = 0, 0
    for uid, seq in user_sequences.items():
        seq_pops = [item_popularity.get(item, 1) for item in seq] if item_popularity else [1]
        avg_pop = np.mean(seq_pops)
        if avg_pop > p66:
            bucket_info[uid] = 'high_pop'
        elif avg_pop > p33:
            bucket_info[uid] = 'mid_pop'
        else:
            bucket_info[uid] = 'low_pop'
    return bucket_info
