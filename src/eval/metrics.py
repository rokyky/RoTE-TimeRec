'''Evaluation metrics for sequential recommendation.

Reference:
    - RecBole full-sort evaluation
    - Recall@K / NDCG@K / MRR@K
    - Bucket evaluation for long-tail analysis
'''

import math
import numpy as np
import torch
from typing import Dict, List

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

def evaluate_full_sort(scores, ground_truth, ks=None):
    if ks is None: ks = [1, 5, 10, 20]
    results = {}
    _, ranked = torch.topk(scores, k=max(ks), dim=1)
    ranked = ranked.cpu().tolist()
    for i, gt in enumerate(ground_truth):
        for k in ks:
            results.setdefault(f'recall@{k}', []).append(recall_at_k(ranked[i], gt, k))
            results.setdefault(f'ndcg@{k}', []).append(ndcg_at_k(ranked[i], gt, k))
    return {k: sum(v)/len(v) for k, v in results.items()}

def _model_forward(model, hist, pos, device):
    from src.models.tisasrec import TiSASRec
    from src.models.tisasrec_cat import TiSASRecCat
    if isinstance(model, TiSASRecCat):
        B, L = hist.shape
        td = torch.zeros(B, L, L, device=device)
        cm = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        return model(hist, pos, td, cm)
    elif isinstance(model, TiSASRec):
        B, L = hist.shape
        td = torch.zeros(B, L, L, device=device)
        return model(hist, pos, td)
    else:
        return model(hist, pos)

def model_eval(model, eval_loader, device, ks=None):
    if ks is None: ks = [1, 5, 10, 20]
    model.eval()
    all_scores, all_targets = [], []
    with torch.no_grad():
        for hist, pos, target, uid in eval_loader:
            hist, pos = hist.to(device), pos.to(device)
            scores = _model_forward(model, hist, pos, device)
            all_scores.append(scores)
            all_targets.append(target)
    scores = torch.cat(all_scores, dim=0)
    targets = torch.cat(all_targets, dim=0)
    ground_truth = [[t.item()] for t in targets]
    return evaluate_full_sort(scores, ground_truth, ks)

def evaluate_by_bucket(model, eval_loader, device, bucket_info, ks=None):
    if ks is None: ks = [5, 10, 20]
    model.eval()
    bucket_results = {}
    with torch.no_grad():
        for hist, pos, target, uid in eval_loader:
            hist, pos = hist.to(device), pos.to(device)
            scores = _model_forward(model, hist, pos, device)
            _, ranked = torch.topk(scores, k=max(ks), dim=1)
            ranked = ranked.cpu().tolist()
            for i, uid_i in enumerate(uid):
                uid_i = uid_i.item()
                bucket = bucket_info.get(uid_i, 'unknown')
                if bucket not in bucket_results:
                    bucket_results[bucket] = {'recall': {k: [] for k in ks}, 'ndcg': {k: [] for k in ks}}
                gt = [target[i].item()]
                for k in ks:
                    bucket_results[bucket]['recall'][k].append(recall_at_k(ranked[i], gt, k))
                    bucket_results[bucket]['ndcg'][k].append(ndcg_at_k(ranked[i], gt, k))
    summary = {}
    for bucket, metrics in bucket_results.items():
        summary[bucket] = {}
        for k in ks:
            r = metrics['recall'][k]
            n = metrics['ndcg'][k]
            summary[bucket][f'recall@{k}'] = sum(r)/len(r) if r else 0.0
            summary[bucket][f'ndcg@{k}'] = sum(n)/len(n) if n else 0.0
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
        if avg_pop > p66: bucket_info[uid] = 'high_pop'
        elif avg_pop > p33: bucket_info[uid] = 'mid_pop'
        else: bucket_info[uid] = 'low_pop'
    return bucket_info