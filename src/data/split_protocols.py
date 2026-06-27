"""Data split protocols for sequential recommendation.

Provides four sequence split strategies:
1. leave_one_out  - last interaction as target, rest as history
2. no_sss         - one sequence per user, no sub-sequence augmentation
3. sliding_window_sss - sliding window over user sequence
4. prefix_target_sss  - multiple prefix-target pairs from each sequence

Each sample retains: item_sequence, timestamp_sequence, target_item,
target_timestamp, user_id.

Also provides a target leakage check function.
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
    """Left-pad a sequence to max_len."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    pad_len = max_len - len(seq)
    return [pad_value] * pad_len + seq


def _pad_timestamp(seq: List[float], max_len: int, pad_value=0.0) -> List[float]:
    """Left-pad a timestamp sequence to max_len."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    pad_len = max_len - len(seq)
    return [pad_value] * pad_len + seq


def _drop_target_from_history(
    history: List[int],
    timestamps: Optional[List[float]],
    target: int,
) -> Tuple[List[int], Optional[List[float]]]:
    """Remove repeated target ids from history while preserving alignment."""
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
    """Check if target_item appears in the history prefix.

    Args:
        item_sequence: Padded item sequence (max_len,).
        target_item: Target item to check.
        history_end_idx: Index indicating where history ends
                         (default: entire sequence minus last position).

    Returns:
        True if target leaks into history, False otherwise.
    """
    if history_end_idx < 0:
        # Use all non-padding items except possibly the last
        seq = item_sequence[item_sequence != 0]
        if len(seq) == 0:
            return False
        return target_item.item() in seq.tolist()

    # Only check up to history_end_idx
    prefix = item_sequence[:history_end_idx]
    prefix = prefix[prefix != 0]
    return target_item.item() in prefix.tolist()


def split_leave_one_out(
    user_sequences: Dict[int, List[int]],
    max_len: int = 50,
    user_timestamps: Optional[Dict[int, List[float]]] = None,
) -> List[SplitSample]:
    """Leave-one-out split: last interaction as target, rest as history.

    For each user with >= 2 interactions:
        - history = all items except last (truncated/padded to max_len)
        - target = last item
        - preserves timestamps for both history and target

    Returns:
        List of (item_seq, ts_seq, target_item, target_ts, user_id) tuples.
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

        # Pad history
        hist_padded = _pad_sequence(history, max_len, pad_value=0)
        item_seq = torch.tensor(hist_padded, dtype=torch.long)

        # Target timestamp
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
    """No sub-sequence split: one full sequence per user, no augmentation.

    The entire user sequence is used as context. The last item in the
    (truncated) sequence is the target.

    Returns:
        List of (item_seq, ts_seq, target_item, target_ts, user_id) tuples.
    """
    samples = []
    for uid, seq in user_sequences.items():
        if len(seq) < 2:
            continue

        # Truncate to max_len
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
    """Sliding window sub-sequence split.

    For each user, slides a window over their interaction sequence to
    create multiple training samples. Each window:
        - history = window[:-1] (the first window_size - 1 items)
        - target = window[-1] (the last item in the window)
        - window slides by 1 step

    Args:
        window_size: Number of items in each window. Must be >= 2.
                     History will be window_size - 1 items.

    Returns:
        List of (item_seq, ts_seq, target_item, target_ts, user_id) tuples.
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
    """Prefix-target sub-sequence split.

    For each user, creates multiple prefix-target pairs:
        - For each position i >= prefix_min_len in the sequence:
            - history = seq[:i]
            - target = seq[i]
        - This creates (seq_len - prefix_min_len) samples per user.

    Args:
        prefix_min_len: Minimum history length for a valid sample.
                        Must be >= 1.

    Returns:
        List of (item_seq, ts_seq, target_item, target_ts, user_id) tuples.
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


# Registry of split functions
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
    """Apply a named split protocol.

    Args:
        name: Split protocol name (leave_one_out, no_sss,
              sliding_window_sss, prefix_target_sss).
        user_sequences: Dict of user_id -> interaction sequence.
        max_len: Maximum sequence length (padding/truncation).
        user_timestamps: Dict of user_id -> timestamp sequence.
        **kwargs: Additional protocol-specific args
                  (window_size, prefix_min_len).

    Returns:
        List of split samples.

    Raises:
        ValueError: If split name is unknown.
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
