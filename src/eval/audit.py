"""RoTE-TimeRec 评估审计系统。

提供：
1. 切分协议审计：比较各切分协议间的指标。
2. 困难切片评估：历史长度、时间间隔、流行度、类目切换。
3. 运行时指标：延迟、内存、参数量。
4. 结果导出：聚合表、切片表、运行时摘要、JSON。

用法：
    auditor = SplitProtocolAuditor(model, device)
    results = auditor.audit(eval_data, split_protocols=['leave_one_out', 'no_sss'])
    auditor.print_results(results)
"""

import json
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .metrics import (
    evaluate_full_sort,
    recall_at_k,
    ndcg_at_k,
    _model_forward,
)
from ..data.split_protocols import apply_split, check_target_leakage

logger = logging.getLogger(__name__)


def compute_runtime_metrics(
    model: nn.Module,
    sample_inputs: Dict[str, torch.Tensor],
    device: str = 'cpu',
    num_warmup: int = 5,
    num_runs: int = 50,
) -> Dict[str, float]:
    """计算模型前向传播的平均和 P95 延迟。

    参数：
        model: 要评测的模型。
        sample_inputs: 输入张量字典（键取决于模型类型）。
        device: 运行设备。
        num_warmup: 预热迭代次数。
        num_runs: 计时迭代次数。

    返回：
        包含 'avg_latency_ms', 'p95_latency_ms', 'peak_memory_mb'（或 -1）
        和 'param_count' 的字典。
    """
    model.eval()
    model.to(device)

    # 将输入移至设备。某些调用者传递遗留辅助键，如
    # "model" 或 "device"；保持 benchmark API 的容错性，但
    # 使用一个显式的 model/device 对调用 _model_forward。
    inputs = {}
    for k, v in sample_inputs.items():
        if k in ('model', 'device'):
            continue
        inputs[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    # 参数量
    param_count = sum(p.numel() for p in model.parameters())

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = _model_forward(model, device=device, **inputs)

    # 计时运行
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = _model_forward(model, device=device, **inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # 毫秒

    avg_latency = sum(latencies) / len(latencies)
    sorted_lats = sorted(latencies)
    p95_idx = int(len(sorted_lats) * 0.95)
    p95_latency = sorted_lats[min(p95_idx, len(sorted_lats) - 1)]

    # 峰值 GPU 显存（如果可用）
    peak_memory = -1.0
    if device != 'cpu' and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    return {
        'avg_latency_ms': round(avg_latency, 4),
        'p95_latency_ms': round(p95_latency, 4),
        'peak_memory_mb': round(peak_memory, 2) if peak_memory > 0 else -1.0,
        'param_count': param_count,
    }


def _get_history_length(item_sequence: torch.Tensor) -> int:
    """计算序列中非填充物品的数量。"""
    return (item_sequence != 0).sum().item()


def _get_time_gap(
    item_sequence: torch.Tensor,
    timestamp_sequence: torch.Tensor,
) -> float:
    """获取最后一个历史物品与目标之间的时间间隔（秒）。

    间隔基于最后一个非填充位置的时间戳与目标时间戳的对比
    （此处未传入目标时间戳，因此进行近似计算）。
    返回历史序列内部的时间跨度。
    """
    non_pad_mask = item_sequence != 0
    if non_pad_mask.sum() < 2:
        return float('inf')

    # 获取最后两个非填充物品的位置
    indices = non_pad_mask.nonzero(as_tuple=True)[0]
    if len(indices) < 2:
        return float('inf')

    last_idx = indices[-1].item()
    second_last_idx = indices[-2].item()
    gap = abs(timestamp_sequence[last_idx].item() - timestamp_sequence[second_last_idx].item())
    return gap


def _get_item_popularity(
    item_id: int,
    item_freq: Optional[Dict[int, int]],
) -> int:
    """获取物品的频次计数。"""
    if item_freq is None:
        return 0
    return item_freq.get(item_id, 0)


def compute_hard_slice_metrics(
    model: nn.Module,
    samples: List[Tuple],
    device: str,
    item_freq: Optional[Dict[int, int]] = None,
    item_categories: Optional[Dict[int, int]] = None,
    ks: Optional[List[int]] = None,
    exclude_items: Optional[Set[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """按困难类别切片计算指标。

    切片：
        - history_length: 短历史（底部 33%）/ 长历史（顶部 33%）
        - time_gap: 短间隔（< 1 天）/ 长间隔（> 7 天）
        - item_popularity: 头部（顶部 20%）/ 尾部（底部 50%）
        - category_switch: 同类 / 类目切换

    参数：
        model: 要评估的模型。
        samples: (item_seq, ts_seq, target_item, target_ts, user_id) 列表。
        device: Torch 设备。
        item_freq: item_id -> 交互次数 字典（用于流行度）。
        item_categories: item_id -> category_id 字典。
        ks: 指标的 K 值。
        exclude_items: 要从排序中排除的物品。

    返回：
        嵌套字典：slice_name -> metric_name -> value。
    """
    if ks is None:
        ks = [1, 5, 10, 20]
    if exclude_items is None:
        exclude_items = {0}

    model.eval()

    # 按切片准备数据结构
    slice_data = defaultdict(lambda: {'scores': [], 'targets': []})

    # 计算历史长度、间隔等，用于百分位数切分
    hist_lengths = []
    gaps = []
    pop_values = []

    for sample in samples:
        item_seq = sample[0]
        hist_lengths.append(_get_history_length(item_seq))
        gap = _get_time_gap(item_seq, sample[1])
        gaps.append(gap)
        target_item = sample[2].item()
        pop_values.append(_get_item_popularity(target_item, item_freq))

    # 计算阈值
    if hist_lengths:
        hist_sorted = sorted(hist_lengths)
        short_thresh = hist_sorted[len(hist_sorted) // 3] if len(hist_sorted) >= 3 else 0
        long_thresh = hist_sorted[2 * len(hist_sorted) // 3] if len(hist_sorted) >= 3 else float('inf')
    else:
        short_thresh, long_thresh = 0, float('inf')

    if pop_values:
        pop_sorted = sorted(pop_values, reverse=True)
        head_thresh = pop_sorted[len(pop_sorted) // 5] if len(pop_sorted) >= 5 else 1
        tail_thresh = pop_sorted[len(pop_sorted) // 2] if len(pop_sorted) >= 2 else 0
    else:
        head_thresh, tail_thresh = 1, 0

    # 分类和累积样本
    with torch.no_grad():
        for sample in samples:
            item_seq = sample[0].to(device)
            ts_seq = sample[1].to(device)
            target_item = sample[2]
            target_ts = sample[3]
            uid = sample[4]

            # 创建位置序列
            pos = torch.arange(item_seq.size(0), dtype=torch.long, device=device).unsqueeze(0)

            # 前向传播 - 需要处理模型类型
            # 为简化起见，使用不带 time_deltas/cat_mask 的 _model_forward
            # 因为切分协议产生的样本不包含这些
            scores = _model_forward(
                model, item_seq.unsqueeze(0), pos, device,
                timestamps=ts_seq.unsqueeze(0) if ts_seq is not None else None,
            )
            target = torch.tensor([target_item.item()], device=device)

            # 应用排除
            for eid in exclude_items:
                if 0 <= eid < scores.size(1):
                    scores[:, eid] = -float('inf')

            hist_len = _get_history_length(item_seq)
            gap = _get_time_gap(item_seq, ts_seq)
            pop = _get_item_popularity(target_item.item(), item_freq)

            # 历史长度切片
            if hist_len <= short_thresh:
                slice_data['short_history']['scores'].append(scores)
                slice_data['short_history']['targets'].append(target)
            if hist_len >= long_thresh:
                slice_data['long_history']['scores'].append(scores)
                slice_data['long_history']['targets'].append(target)

            # 时间间隔切片
            if gap < 86400.0:  # < 1 天
                slice_data['short_gap']['scores'].append(scores)
                slice_data['short_gap']['targets'].append(target)
            if gap > 7 * 86400.0:  # > 7 天
                slice_data['long_gap']['scores'].append(scores)
                slice_data['long_gap']['targets'].append(target)

            # 物品流行度切片
            if pop >= head_thresh:
                slice_data['head_items']['scores'].append(scores)
                slice_data['head_items']['targets'].append(target)
            if pop <= tail_thresh:
                slice_data['tail_items']['scores'].append(scores)
                slice_data['tail_items']['targets'].append(target)

            # 类目切换切片
            if item_categories is not None:
                target_cat = item_categories.get(target_item.item(), -1)
                # 获取最后一个历史物品的类目
                non_pad = item_seq[item_seq != 0]
                if len(non_pad) > 0:
                    last_hist_item = non_pad[-1].item()
                    last_cat = item_categories.get(last_hist_item, -1)
                    if target_cat == last_cat and target_cat >= 0:
                        slice_data['same_category']['scores'].append(scores)
                        slice_data['same_category']['targets'].append(target)
                    elif target_cat >= 0 and last_cat >= 0:
                        slice_data['category_switch']['scores'].append(scores)
                        slice_data['category_switch']['targets'].append(target)

    # 按切片计算指标
    results = {}
    for slice_name, data in slice_data.items():
        if not data['scores']:
            continue
        all_scores = torch.cat(data['scores'], dim=0)
        all_targets = torch.cat(data['targets'], dim=0)
        ground_truth = [[t.item()] for t in all_targets]

        metrics = evaluate_full_sort(all_scores, ground_truth, ks, exclude_items=None)
        results[slice_name] = metrics

    return results


class SplitProtocolAuditor:
    """跨多个切分协议运行评估并聚合结果。

    比较各切分协议间的 HR/NDCG/Recall，并可选
    计算困难切片指标和运行时基准。
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ks: Optional[List[int]] = None,
    ):
        self.model = model
        self.device = device
        self.ks = ks or [1, 5, 10, 20]

    def audit(
        self,
        user_sequences: Dict[int, List[int]],
        split_protocols: Optional[List[str]] = None,
        user_timestamps: Optional[Dict[int, List[float]]] = None,
        item_categories: Optional[Dict[int, int]] = None,
        item_freq: Optional[Dict[int, int]] = None,
        max_len: int = 50,
        exclude_items: Optional[Set[int]] = None,
        compute_hard_slices: bool = True,
        compute_runtime: bool = True,
    ) -> Dict[str, Any]:
        """运行完整审计。

        参数：
            user_sequences: user_id -> 物品序列 的字典。
            split_protocols: 要比较的协议列表。默认：全部。
            user_timestamps: user_id -> 时间戳序列 的字典。
            item_categories: item_id -> category_id 的字典。
            item_freq: item_id -> 交互次数 的字典。
            max_len: 最大序列长度。
            exclude_items: 要排除的物品（例如 {0}）。
            compute_hard_slices: 是否计算困难切片指标。
            compute_runtime: 是否计算运行时基准。

        返回：
            包含 'aggregate', 'slices', 'runtime', 'config' 字段的字典。
        """
        if split_protocols is None:
            split_protocols = ['leave_one_out', 'no_sss', 'sliding_window_sss', 'prefix_target_sss']
        if exclude_items is None:
            exclude_items = {0}

        aggregate = {}
        slices = {}
        runtime = None

        for protocol in split_protocols:
            logger.info("Running audit for protocol: %s", protocol)

            # 为此协议生成样本
            kwargs = {}
            if protocol == 'sliding_window_sss':
                kwargs['window_size'] = 10
            elif protocol == 'prefix_target_sss':
                kwargs['prefix_min_len'] = 3

            samples = apply_split(
                protocol,
                user_sequences,
                max_len=max_len,
                user_timestamps=user_timestamps,
                **kwargs,
            )

            # 检查目标泄露
            leakage_count = 0
            for sample in samples:
                if check_target_leakage(sample[0], sample[2]):
                    leakage_count += 1
            if leakage_count > 0:
                logger.warning(
                    "%s: %d/%d samples have target leakage",
                    protocol, leakage_count, len(samples),
                )

            # 评估
            with torch.no_grad():
                all_scores = []
                all_targets = []
                for sample in samples:
                    item_seq = sample[0].to(self.device)
                    ts_seq = sample[1].to(self.device)
                    target_item = sample[2]

                    pos = torch.arange(item_seq.size(0), dtype=torch.long, device=self.device).unsqueeze(0)

                    scores = _model_forward(
                        self.model,
                        item_seq.unsqueeze(0),
                        pos,
                        self.device,
                        timestamps=ts_seq.unsqueeze(0) if ts_seq is not None else None,
                    )

                    # 应用排除
                    for eid in exclude_items:
                        if 0 <= eid < scores.size(1):
                            scores[:, eid] = -float('inf')

                    all_scores.append(scores)
                    all_targets.append(target_item.unsqueeze(0).to(self.device))

                if all_scores:
                    cat_scores = torch.cat(all_scores, dim=0)
                    cat_targets = torch.cat(all_targets, dim=0)
                    ground_truth = [[t.item()] for t in cat_targets]
                    metrics = evaluate_full_sort(cat_scores, ground_truth, self.ks, exclude_items=None)
                    aggregate[protocol] = metrics

                # 困难切片指标（仅 leave_one_out，最具代表性）
                if compute_hard_slices and protocol == 'leave_one_out':
                    slice_metrics = compute_hard_slice_metrics(
                        self.model, samples, self.device,
                        item_freq=item_freq,
                        item_categories=item_categories,
                        ks=self.ks,
                        exclude_items=exclude_items,
                    )
                    slices[protocol] = slice_metrics

                # 运行时基准（仅第一个协议以避免冗余）
                if compute_runtime and protocol == split_protocols[0] and samples:
                    sample = samples[0]
                    sample_inputs = {
                        'hist': sample[0].unsqueeze(0).to(self.device),
                        'pos': torch.arange(sample[0].size(0), dtype=torch.long, device=self.device).unsqueeze(0),
                        'device': self.device,
                    }
                    if sample[1] is not None:
                        sample_inputs['timestamps'] = sample[1].unsqueeze(0).to(self.device)
                    runtime = compute_runtime_metrics(self.model, sample_inputs, device=self.device)

        return {
            'aggregate': aggregate,
            'slices': slices,
            'runtime': runtime,
            'config': {
                'split_protocols': split_protocols,
                'ks': self.ks,
                'max_len': max_len,
                'device': self.device,
            },
        }

    def print_results(self, results: Dict[str, Any]) -> None:
        """打印可读格式的审计结果。"""
        print("\n" + "=" * 70)
        print("RoTE-TimeRec Split Protocol Audit Results")
        print("=" * 70)

        # 聚合表
        aggregate = results.get('aggregate', {})
        if aggregate:
            print("\n--- Aggregate Metrics ---")
            header = f"{'Protocol':<25}"
            for k in self.ks:
                header += f"  {'Recall@' + str(k):<12}  {'NDCG@' + str(k):<12}"
            print(header)
            print("-" * len(header))
            for protocol, metrics in aggregate.items():
                row = f"{protocol:<25}"
                for k in self.ks:
                    r = metrics.get(f'recall@{k}', 0.0)
                    n = metrics.get(f'ndcg@{k}', 0.0)
                    row += f"  {r:<12.4f}  {n:<12.4f}"
                print(row)

        # 困难切片表
        slices = results.get('slices', {})
        if slices:
            print("\n--- Hard-Slice Metrics (leave_one_out) ---")
            for protocol, slice_data in slices.items():
                for slice_name, metrics in slice_data.items():
                    row = f"  {slice_name:<20}"
                    for k in self.ks:
                        r = metrics.get(f'recall@{k}', 0.0)
                        n = metrics.get(f'ndcg@{k}', 0.0)
                        row += f"  R@{k}={r:.4f} N@{k}={n:.4f}"
                    print(row)

        # 运行时表
        runtime = results.get('runtime')
        if runtime:
            print("\n--- Runtime Metrics ---")
            print(f"  Avg latency:  {runtime.get('avg_latency_ms', 'N/A'):>8} ms")
            print(f"  P95 latency:  {runtime.get('p95_latency_ms', 'N/A'):>8} ms")
            print(f"  Peak memory:  {runtime.get('peak_memory_mb', 'N/A'):>8} MB")
            print(f"  Parameters:   {runtime.get('param_count', 'N/A'):>8}")

        print("=" * 70)

    def export_json(self, results: Dict[str, Any], path: str) -> None:
        """将结果导出到 JSON 文件。

        参数：
            results: audit() 返回的字典。
            path: 输出 JSON 文件路径。
        """
        def _convert(obj):
            if isinstance(obj, (float,)):
                return round(obj, 6)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        export = _convert(results)
        with open(path, 'w') as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        logger.info("Results exported to %s", path)

    def export_tables(self, results: Dict[str, Any], prefix: str = 'audit_results') -> Dict[str, str]:
        """将结果导出为格式化表格。

        参数：
            results: audit() 返回的字典。
            prefix: 输出的文件路径前缀。

        返回：
            table_type -> file_path 的字典。
        """
        import os

        paths = {}

        # 聚合表
        agg_path = f"{prefix}_aggregate.csv"
        with open(agg_path, 'w') as f:
            header = ['protocol']
            for k in self.ks:
                header.append(f'recall@{k}')
                header.append(f'ndcg@{k}')
            f.write(','.join(header) + '\n')
            for protocol, metrics in results.get('aggregate', {}).items():
                row = [protocol]
                for k in self.ks:
                    row.append(f"{metrics.get(f'recall@{k}', 0.0):.6f}")
                    row.append(f"{metrics.get(f'ndcg@{k}', 0.0):.6f}")
                f.write(','.join(row) + '\n')
        paths['aggregate_csv'] = os.path.abspath(agg_path)
        logger.info("Aggregate table exported to %s", agg_path)

        # 切片表
        sl_path = f"{prefix}_slices.csv"
        with open(sl_path, 'w') as f:
            header = ['protocol', 'slice']
            for k in self.ks:
                header.append(f'recall@{k}')
                header.append(f'ndcg@{k}')
            f.write(','.join(header) + '\n')
            for protocol, slice_data in results.get('slices', {}).items():
                for slice_name, metrics in slice_data.items():
                    row = [protocol, slice_name]
                    for k in self.ks:
                        row.append(f"{metrics.get(f'recall@{k}', 0.0):.6f}")
                        row.append(f"{metrics.get(f'ndcg@{k}', 0.0):.6f}")
                    f.write(','.join(row) + '\n')
        paths['slices_csv'] = os.path.abspath(sl_path)
        logger.info("Slice table exported to %s", sl_path)

        return paths
