"""序列推荐的数据切分协议。

提供四种序列切分策略：
1. leave_one_out  - 最后交互作为目标，其余作为历史
2. no_sss         - 每用户一个序列，无子序列增强
3. sliding_window_sss - 在用户序列上滑动窗口
4. prefix_target_sss  - 从每个序列生成多个前缀-目标对

每个样本保留：物品序列、时间戳序列、目标物品、
目标时间戳、用户ID。

同时还提供目标泄露检测函数。
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SplitSample = Tuple[
    torch.LongTensor,    # item_sequence  (max_len,)
    torch.FloatTensor,   # timestamp_sequence (max_len,)
    torch.LongTensor,    # target_item (scalar)
    torch.FloatTensor,   # target_timestamp (scalar)
    torch.LongTensor,    # user_id (scalar)
]


def _pad_sequence(seq: List, max_len: int, pad_value=0) -> List:
    """将序列左填充到 max_len。"""
    if len(seq) >= max_len:
        return seq[-max_len:]
    pad_len = max_len - len(seq)
    return [pad_value] * pad_len + seq


def _pad_timestamp(seq: List[float], max_len: int, pad_value=0.0) -> List[float]:
    """将时间戳序列左填充到 max_len。"""
    if len(seq) >= max_len:
        return seq[-max_len:]
    pad_len = max_len - len(seq)
    return [pad_value] * pad_len + seq


def _drop_target_from_history(
    history: List[int],
    timestamps: Optional[List[float]],
    target: int,
) -> Tuple[List[int], Optional[List[float]]]:
    """从历史中移除重复的目标 ID，同时保持对齐。"""
    if timestamps is None:
        return [item for item in history if item != target], None

    filtered_items = []
    filtered_ts = []
    for item, ts in zip(history, timestamps):
        if item == target:
            continue
        filtered_items.append(item)
        filtered_ts.append(ts)
    return filtered_items, filtered_ts


def check_target_leakage(
    item_sequence: torch.LongTensor,
    target_item: torch.LongTensor,
    history_end_idx: int = -1,
) -> bool:
    """检查 target_item 是否出现在历史前缀中。

    参数：
        item_sequence: 填充后的物品序列 (max_len,)。
        target_item: 要检查的目标物品。
        history_end_idx: 指示历史结束位置的索引
                         （默认：整个序列除去最后一个位置）。

    返回：
        若目标泄露到历史中则返回 True，否则返回 False。
    """
    if history_end_idx < 0:
        # 使用所有非填充物品，可能除去最后一个
        seq = item_sequence[item_sequence != 0]
        if len(seq) == 0:
            return False
        return target_item.item() in seq.tolist()

    # 只检查到 history_end_idx
    prefix = item_sequence[:history_end_idx]
    prefix = prefix[prefix != 0]
    return target_item.item() in prefix.tolist()


def split_leave_one_out(
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
) -> List[SplitSample]:
    """留一法切分：最后交互作为目标，其余作为历史。

    对于每个 >= 2 次交互的用户：
        - history = 除最后一个外的所有物品（截断/填充至 max_len）
        - target = 最后一个物品
        - 保留历史和目标的对应时间戳

    返回：
        (item_seq, ts_seq, target_item, target_ts, user_id) 元组列表。
    """
    samples = []
    for uid, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        history = seq[:-1]
        target = seq[-1]
        ts_list = user_timestamps.get(uid, None) if user_timestamps else None
        ts_history = ts_list[:len(seq) - 1] if ts_list is not None and len(ts_list) >= len(seq) else None
        history, ts_history = _drop_target_from_history(history, ts_history, target)

        # 填充历史
        hist_padded = _pad_sequence(history, max_len, pad_value=0)
        item_seq = torch.tensor(hist_padded, dtype=torch.long)

        # 目标时间戳
        if ts_list is not None and len(ts_list) >= len(seq):
            target_ts = ts_list[-1]
            ts_padded = _pad_timestamp(ts_history, max_len, pad_value=0.0)
        else:
            target_ts = 0.0
            ts_padded = [0.0] * max_len

        ts_seq = torch.tensor(ts_padded, dtype=torch.float64)

        samples.append((
            item_seq,
            ts_seq,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(target_ts, dtype=torch.float64),
            torch.tensor(uid, dtype=torch.long),
        ))

    return samples


def split_no_sss(
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
) -> List[SplitSample]:
    """无子序列切分：每用户一个完整序列，无增强。

    整个用户序列用作上下文。（截断后）序列的最后一个物品是目标。

    返回：
        (item_seq, ts_seq, target_item, target_ts, user_id) 元组列表。
    """
    samples = []
    for uid, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        # 截断到 max_len
        truncated = seq[-max_len:] if len(seq) > max_len else seq
        history = truncated[:-1] if len(truncated) >= 2 else []
        target = truncated[-1]

        if not history:
            continue

        ts_list = user_timestamps.get(uid, None) if user_timestamps else None

        if ts_list is not None and len(ts_list) >= len(truncated):
            ts_truncated = ts_list[-len(truncated):] if len(ts_list) >= len(truncated) else ts_list
            target_ts = ts_truncated[-1]
            ts_history = ts_truncated[:-1]
            history, ts_history = _drop_target_from_history(history, ts_history, target)
            ts_padded = _pad_timestamp(ts_history, max_len, pad_value=0.0)
        else:
            target_ts = 0.0
            history, _ = _drop_target_from_history(history, None, target)
            ts_padded = [0.0] * max_len

        hist_padded = _pad_sequence(history, max_len, pad_value=0)
        item_seq = torch.tensor(hist_padded, dtype=torch.long)
        ts_seq = torch.tensor(ts_padded, dtype=torch.float64)

        samples.append((
            item_seq,
            ts_seq,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(target_ts, dtype=torch.float64),
            torch.tensor(uid, dtype=torch.long),
        ))

    return samples


def split_sliding_window_sss(
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    window_size: int = 10,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
) -> List[SplitSample]:
    """滑动窗口子序列切分。

    对每个用户，在交互序列上滑动窗口以创建多个训练样本。每个窗口：
        - history = window[:-1]（前 window_size - 1 个物品）
        - target = window[-1]（窗口中的最后一个物品）
        - 窗口步长为 1

    参数：
        window_size: 每个窗口中的物品数。必须 >= 2。
                     历史长度为 window_size - 1 个物品。

    返回：
        (item_seq, ts_seq, target_item, target_ts, user_id) 元组列表。
    """
    if window_size < 2:
        raise ValueError(f"window_size must be >= 2, got {window_size}")

    samples = []
    for uid, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        ts_list = user_timestamps.get(uid, None) if user_timestamps else None

        for start in range(max(0, len(seq) - window_size + 1)):
            window = seq[start:start + window_size]
            if len(window) < 2:
                continue

            history = window[:-1]
            target = window[-1]

            if ts_list is not None and len(ts_list) >= start + window_size:
                ts_window = ts_list[start:start + window_size]
                target_ts = ts_window[-1]
                ts_history = ts_window[:-1]
                history, ts_history = _drop_target_from_history(history, ts_history, target)
                ts_padded = _pad_timestamp(ts_history, max_len, pad_value=0.0)
            else:
                target_ts = 0.0
                history, _ = _drop_target_from_history(history, None, target)
                ts_padded = [0.0] * max_len

            hist_padded = _pad_sequence(history, max_len, pad_value=0)
            item_seq = torch.tensor(hist_padded, dtype=torch.long)
            ts_seq = torch.tensor(ts_padded, dtype=torch.float64)

            samples.append((
                item_seq,
                ts_seq,
                torch.tensor(target, dtype=torch.long),
                torch.tensor(target_ts, dtype=torch.float64),
                torch.tensor(uid, dtype=torch.long),
            ))

    return samples


def split_prefix_target_sss(
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    prefix_min_len: int = 3,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
) -> List[SplitSample]:
    """前缀-目标子序列切分。

    对每个用户，创建多个前缀-目标对：
        - 对于序列中每个位置 i >= prefix_min_len：
            - history = seq[:i]
            - target = seq[i]
        - 每用户生成 (seq_len - prefix_min_len) 个样本。

    参数：
        prefix_min_len: 有效样本的最小历史长度。必须 >= 1。

    返回：
        (item_seq, ts_seq, target_item, target_ts, user_id) 元组列表。
    """
    if prefix_min_len < 1:
        raise ValueError(f"prefix_min_len must be >= 1, got {prefix_min_len}")

    samples = []
    for uid, seq in user_sequences.items():
        if len(seq) < prefix_min_len + 1:
            continue

        ts_list = user_timestamps.get(uid, None) if user_timestamps else None

        for i in range(prefix_min_len, len(seq)):
            history = seq[:i]
            target = seq[i]

            if ts_list is not None and len(ts_list) > i:
                target_ts = ts_list[i]
                ts_history = ts_list[:i]
                history, ts_history = _drop_target_from_history(history, ts_history, target)
                ts_padded = _pad_timestamp(ts_history, max_len, pad_value=0.0)
            else:
                target_ts = 0.0
                history, _ = _drop_target_from_history(history, None, target)
                ts_padded = [0.0] * max_len

            hist_padded = _pad_sequence(history, max_len, pad_value=0)
            item_seq = torch.tensor(hist_padded, dtype=torch.long)
            ts_seq = torch.tensor(ts_padded, dtype=torch.float64)

            samples.append((
                item_seq,
                ts_seq,
                torch.tensor(target, dtype=torch.long),
                torch.tensor(target_ts, dtype=torch.float64),
                torch.tensor(uid, dtype=torch.long),
            ))

    return samples


# 切分函数注册表
SPLIT_REGISTRY = {
    'leave_one_out': split_leave_one_out,
    'no_sss': split_no_sss,
    'sliding_window_sss': split_sliding_window_sss,
    'prefix_target_sss': split_prefix_target_sss,
}


def apply_split(
    name: str,
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
    **kwargs,
) -> List[SplitSample]:
    """应用指定的切分协议。

    参数：
        name: 切分协议名称（leave_one_out, no_sss,
              sliding_window_sss, prefix_target_sss）。
        user_sequences: user_id -> 交互序列 的字典。
        max_len: 最大序列长度（填充/截断）。
        user_timestamps: user_id -> 时间戳序列 的字典。
        **kwargs: 额外的协议特定参数（window_size, prefix_min_len）。

    返回：
        切分后的样本列表。

    抛出：
        ValueError: 若切分名称未知。
    """
    if name not in SPLIT_REGISTRY:
        raise ValueError(
            f"Unknown split '{name}'. Available: {list(SPLIT_REGISTRY.keys())}"
        )

    split_fn = SPLIT_REGISTRY[name]
    return split_fn(
        user_sequences,
        max_len=max_len,
        user_timestamps=user_timestamps,
        **kwargs,
    )
