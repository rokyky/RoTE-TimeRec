# prepare_data.py — 下载并预处理 Amazon 数据集
#
# 支持两种切分模式：
#   leave_one_out (默认): 每个用户最后 1 条 → test, 倒数第 2 → val, 其余 → train
#   random:            按用户随机切分（保持向下兼容）
#
# 输出文件（{data_dir}/{dataset}/）：
#   train.pt, val.pt, test.pt    — Dict[int, List[int]] 用户序列
#   timestamps.pt                — Dict[int, List[float]] 用户交互时间戳（Unix 秒）
#   item_categories.pt           — Dict[int, int] item_id → category_id
#   num_items.pt                 — int
#   num_users.pt                 — int
#   stats.json                   — 统计摘要

import os, json, gzip, sys, argparse, random, logging
from collections import defaultdict, Counter
import torch
import urllib.request

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 数据集下载 URL
DATA_URLS = {
    'beauty': 'https://jmcauley.ucsd.edu/data/amazon_v2/reviewFiles/Beauty.json.gz',
    'sports': 'https://jmcauley.ucsd.edu/data/amazon_v2/reviewFiles/Sports_and_Outdoors.json.gz',
}
META_URLS = {
    'beauty': 'https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/Beauty.json.gz',
    'sports': 'https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/Sports_and_Outdoors.json.gz',
}


def download(url, dest):
    '''下载文件到 dest（若已存在则跳过）。'''
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        logger.info(f'下载: {url}')
        logger.info(f'保存: {dest}')
        urllib.request.urlretrieve(url, dest)
    else:
        logger.info(f'已存在，跳过下载: {dest}')
    return dest


def parse_reviews(path, min_user=5, min_item=5):
    '''解析 review JSON.gz，返回 (user_items, user_timestamps)。

    user_items:     Dict[user_id, List[item_id]]  按时间排序
    user_timestamps: Dict[user_id, List[float]]   对应的 Unix 时间戳
    '''
    user_data = defaultdict(list)  # uid -> [(item_id, timestamp), ...]

    logger.info(f'解析 reviews: {path}')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            uid = r.get('reviewerID', '')
            iid = r.get('asin', '')
            ts = r.get('unixReviewTime', 0)
            if uid and iid:
                user_data[uid].append((iid, ts))

    # 按时间排序
    for uid in user_data:
        user_data[uid].sort(key=lambda x: x[1])

    # 过滤低频物品
    item_cnt = Counter()
    for items in user_data.values():
        for iid, _ in items:
            item_cnt[iid] += 1
    keep_items = {iid for iid, cnt in item_cnt.items() if cnt >= min_item}

    # 过滤短序列用户
    user_items = {}
    user_timestamps = {}
    for uid, items in user_data.items():
        filtered = [(iid, ts) for iid, ts in items if iid in keep_items]
        # 去重（保留第一次出现）
        seen = set()
        deduped = []
        for iid, ts in filtered:
            if iid not in seen:
                seen.add(iid)
                deduped.append((iid, ts))
        if len(deduped) >= min_user:
            user_items[uid] = [iid for iid, _ in deduped]
            user_timestamps[uid] = [ts for _, ts in deduped]

    num_items = len(keep_items)
    logger.info(f'  用户数: {len(user_items)}, 物品数: {num_items}')
    return user_items, user_timestamps, num_items


def parse_meta(path, item_ids):
    '''解析 meta JSON.gz，提取叶子类目映射。

    返回：
        item_categories: Dict[item_id, int]  item_id → category_id
    '''
    logger.info(f'解析 meta: {path}')
    item_categories = {}
    cat_to_id = {}
    next_cat_id = 1

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            m = json.loads(line)
            asin = m.get('asin', '')
            if asin not in item_ids:
                continue

            # 提取叶子类目：category path 的最后一级
            # Amazon meta 的 category 字段是 List[str]，每项是完整路径
            categories = m.get('category', [])
            if not categories:
                continue

            # 取第一个 category path 的最后一级
            cat_path = categories[0] if isinstance(categories[0], str) else str(categories[0])
            leaf = cat_path.split('|')[-1] if '|' in cat_path else cat_path.split('>')[-1]
            leaf = leaf.strip()

            if leaf not in cat_to_id:
                cat_to_id[leaf] = next_cat_id
                next_cat_id += 1
            item_categories[asin] = cat_to_id[leaf]

    logger.info(f'  类目数: {len(cat_to_id)}')
    logger.info(f'  有类目的物品数: {len(item_categories)}')
    return item_categories


def remap_items(user_items, user_timestamps):
    '''将原始 item ID（asin）映射为连续整数 1..N。

    返回：
        remapped_items: Dict[int, List[int]]
        remapped_ts:     Dict[int, List[float]]
        num_items:       int
        id_map:          Dict[original_id, new_id]
    '''
    item2idx = {}
    idx = 1  # 从 1 开始，0 保留给 padding
    for seq in user_items.values():
        for iid in seq:
            if iid not in item2idx:
                item2idx[iid] = idx
                idx += 1

    id_map = item2idx.copy()

    # 用户 ID 也重映射
    users = list(user_items.keys())
    uid_map = {uid: i for i, uid in enumerate(users)}

    remapped_items = {}
    remapped_ts = {}
    for uid in users:
        remapped_items[uid_map[uid]] = [item2idx[iid] for iid in user_items[uid]]
        remapped_ts[uid_map[uid]] = list(user_timestamps[uid])

    return remapped_items, remapped_ts, len(item2idx), id_map, uid_map


def remap_categories(item_categories, id_map):
    '''将 item_categories 的 key 从原始 ID 映射为连续整数。'''
    return {id_map[asin]: cat_id
            for asin, cat_id in item_categories.items()
            if asin in id_map}


def split_leave_one_out(sequences, timestamps):
    '''Leave-one-out 切分：最后 1 条 → test，倒数第 2 → val，其余 → train。

    要求每个用户至少有 3 条交互。
    '''
    train, val, test = {}, {}, {}
    train_ts, val_ts, test_ts = {}, {}, {}

    for uid, seq in sequences.items():
        if len(seq) < 3:
            continue
        train[uid] = seq[:-2]
        val[uid] = seq[:-1]
        test[uid] = seq

        ts = timestamps[uid]
        train_ts[uid] = ts[:-2]
        val_ts[uid] = ts[:-1]
        test_ts[uid] = ts

    return (train, val, test), (train_ts, val_ts, test_ts)


def split_random(sequences, timestamps, val_pct=0.1, test_pct=0.1, seed=42):
    '''按用户随机切分（保持向下兼容）。'''
    users = list(sequences.keys())
    rng = random.Random(seed)
    rng.shuffle(users)

    nv = int(len(users) * val_pct)
    nt = int(len(users) * test_pct)
    val_u = set(users[:nv])
    test_u = set(users[nv:nv + nt])
    train_u = set(users[nv + nt:])

    train = {u: sequences[u] for u in train_u}
    val = {u: sequences[u] for u in val_u}
    test = {u: sequences[u] for u in test_u}
    train_ts = {u: timestamps[u] for u in train_u}
    val_ts = {u: timestamps[u] for u in val_u}
    test_ts = {u: timestamps[u] for u in test_u}

    return (train, val, test), (train_ts, val_ts, test_ts)


def compute_stats(train, val, test, num_items, item_categories):
    '''计算并返回统计摘要。'''
    all_seqs = list(train.values()) + list(val.values()) + list(test.values())
    lengths = [len(s) for s in all_seqs]
    total_interactions = sum(lengths)
    num_users_with_cat = sum(
        1 for u_s, s in [('train', train), ('val', val), ('test', test)]
        for uid, seq in s.items()
        if any(item_categories.get(item) is not None for item in seq)
    )
    total_users = len(train) + len(val) + len(test)

    return {
        'total_users': total_users,
        'train_users': len(train),
        'val_users': len(val),
        'test_users': len(test),
        'num_items': num_items,
        'num_categories': len(set(item_categories.values())),
        'total_interactions': total_interactions,
        'sparsity': f'{100 * (1 - total_interactions / (total_users * num_items)):.2f}%',
        'avg_seq_len': f'{sum(lengths) / len(lengths):.1f}',
        'median_seq_len': f'{sorted(lengths)[len(lengths) // 2]}',
        'min_seq_len': min(lengths),
        'max_seq_len': max(lengths),
        'items_with_category': len(item_categories),
        'items_without_category': num_items - len(item_categories),
    }


def main():
    parser = argparse.ArgumentParser(
        description='下载并预处理 Amazon 数据集')
    parser.add_argument('--dataset', choices=['beauty', 'sports'],
                        required=True, help='数据集名称')
    parser.add_argument('--data-dir', default='./data',
                        help='数据输出目录 (默认: ./data)')
    parser.add_argument('--split-mode', choices=['leave_one_out', 'random'],
                        default='leave_one_out',
                        help='切分模式 (默认: leave_one_out)')
    parser.add_argument('--min-user', type=int, default=5,
                        help='最少交互数 (默认: 5)')
    parser.add_argument('--min-item', type=int, default=5,
                        help='最少数 (默认: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-meta', action='store_true',
                        help='不下载/解析 meta（跳过 item_categories.pt）')
    parser.add_argument('--download-only', action='store_true',
                        help='仅下载，不预处理')
    args = parser.parse_args()

    random.seed(args.seed)

    name = args.dataset
    out_dir = os.path.join(args.data_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 下载 ----
    raw_dir = os.path.join(out_dir, 'raw')
    review_url = DATA_URLS[name]
    review_path = download(review_url,
                           os.path.join(raw_dir, f'{name}.json.gz'))

    if args.download_only:
        if not args.no_meta and name in META_URLS:
            meta_url = META_URLS[name]
            download(meta_url, os.path.join(raw_dir, f'{name}_meta.json.gz'))
        logger.info('下载完成（download-only 模式）')
        return

    # ---- 2. 解析 reviews ----
    user_items, user_timestamps, num_items_raw = parse_reviews(
        review_path, args.min_user, args.min_item)

    # ---- 3. 解析 meta（提取叶子类目） ----
    item_categories = {}
    if not args.no_meta and name in META_URLS:
        meta_url = META_URLS[name]
        meta_path = download(meta_url,
                             os.path.join(raw_dir, f'{name}_meta.json.gz'))
        # 收集所有原始 item ID
        all_asins = set()
        for seq in user_items.values():
            all_asins.update(seq)
        item_categories_raw = parse_meta(meta_path, all_asins)
    else:
        logger.info('跳过 meta 解析')
        item_categories_raw = {}

    # ---- 4. ID 重映射 ----
    (remapped_items, remapped_ts, num_items,
     id_map, uid_map) = remap_items(user_items, user_timestamps)

    item_categories_remapped = remap_categories(item_categories_raw, id_map)

    # ---- 5. 切分 ----
    if args.split_mode == 'leave_one_out':
        logger.info('切分模式: leave-one-out')
        # 先过滤掉序列长度 < 3 的用户
        before = len(remapped_items)
        remapped_items = {u: s for u, s in remapped_items.items() if len(s) >= 3}
        remapped_ts = {u: ts for u, ts in remapped_ts.items() if u in remapped_items}
        if len(remapped_items) < before:
            logger.info(f'  过滤 {before - len(remapped_items)} 个序列长度 < 3 的用户')

        (train, val, test), (train_ts, val_ts, test_ts) = split_leave_one_out(
            remapped_items, remapped_ts)
    else:
        logger.info('切分模式: random')
        (train, val, test), (train_ts, val_ts, test_ts) = split_random(
            remapped_items, remapped_ts, seed=args.seed)

    # ---- 6. 保存 ----
    torch.save(train, os.path.join(out_dir, 'train.pt'))
    torch.save(val, os.path.join(out_dir, 'val.pt'))
    torch.save(test, os.path.join(out_dir, 'test.pt'))
    torch.save(num_items, os.path.join(out_dir, 'num_items.pt'))
    torch.save(len(train) + len(val) + len(test), os.path.join(out_dir, 'num_users.pt'))

    # 保存时间戳和类目
    all_timestamps = {**train_ts, **val_ts, **test_ts}
    torch.save(all_timestamps, os.path.join(out_dir, 'timestamps.pt'))
    torch.save(item_categories_remapped, os.path.join(out_dir, 'item_categories.pt'))

    # ---- 7. 统计 ----
    stats = compute_stats(train, val, test, num_items, item_categories_remapped)

    logger.info('=' * 50)
    logger.info(f'数据集: Amazon {name}')
    logger.info(f'用户数: {stats["total_users"]} '
                f'(train={stats["train_users"]}, val={stats["val_users"]}, '
                f'test={stats["test_users"]})')
    logger.info(f'物品数: {stats["num_items"]}')
    logger.info(f'类目数: {stats["num_categories"]}')
    logger.info(f'总交互: {stats["total_interactions"]}')
    logger.info(f'稀疏度: {stats["sparsity"]}')
    logger.info(f'平均序列长度: {stats["avg_seq_len"]}')
    logger.info(f'中位数序列长度: {stats["median_seq_len"]}')
    logger.info(f'类目覆盖率: {stats["items_with_category"]}/{stats["num_items"]} items')
    logger.info(f'输出目录: {out_dir}')
    logger.info('=' * 50)

    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f'统计已保存: {os.path.join(out_dir, "stats.json")}')


if __name__ == '__main__':
    main()
