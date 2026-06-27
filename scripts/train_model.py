'''训练序列推荐模型。
用法：
    python train_model.py --model sasrec --epochs 5
    python train_model.py --model sasrec_rote --epochs 5
    python train_model.py --model tisasrec_rote --epochs 5
    python train_model.py --model sasrec --split sliding_window_sss

支持模型：
    sasrec, tisasrec, tisasrec_cat, sasrec_rote, tisasrec_rote

支持切分协议：
    leave_one_out, no_sss, sliding_window_sss, prefix_target_sss
'''

import argparse, json, logging, os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.data.loader import SeqRecDataset, EvalDataset
from src.data.split_protocols import apply_split
from src.models import build_model
from src.trainer import Trainer
from src.eval.metrics import model_eval


def build_model_factory(name, num_items, config):
    '''使用模型注册表构建模型。'''
    return build_model(name, num_items, config)


def _model_needs_timestamps(model):
    '''检查模型是否需要时间戳参数。'''
    from src.models.sasrec_rote import SASRecRoTE
    from src.models.tisasrec_rote import TiSASRecRoTE
    return isinstance(model, (SASRecRoTE, TiSASRecRoTE))


def generate_synthetic_data(num_users=100, num_items=50, seed=42):
    '''生成合成数据：序列、时间戳（秒）、物品类目。

    返回：
        sequences: Dict[int, List[int]]
        timestamps: Dict[int, List[float]]
        item_categories: Dict[int, int]
    '''
    random.seed(seed)
    sequences = {}
    timestamps = {}
    # 给每个物品随机分配类目（3 个类目）
    item_categories = {i: random.randint(1, 3) for i in range(1, num_items + 1)}

    base_time = 1_700_000_000  # 约 2023-11-14 unix 时间
    for uid in range(num_users):
        length = random.randint(5, 20)
        seq = [random.randint(1, num_items) for _ in range(length)]
        sequences[uid] = seq
        # 生成递增时间戳，间隔服从指数分布（小时 -> 秒）
        ts = [base_time]
        for _ in range(length - 1):
            gap_hours = random.expovariate(1 / 24)  # 平均 24 小时
            ts.append(ts[-1] + gap_hours * 3600)
        timestamps[uid] = ts

    return sequences, timestamps, item_categories


def build_dataset_with_timestamps(sequences, timestamps_dict, max_len, num_items,
                                   split_protocol='leave_one_out', **split_kwargs):
    '''使用切分协议构造训练样本，带时间戳。

    返回 DataLoader 就绪的 Dataset。
    '''
    samples = apply_split(
        split_protocol,
        sequences,
        max_len=max_len,
        user_timestamps=timestamps_dict,
        **split_kwargs,
    )
    return SplitDataset(samples, num_items)


class SplitDataset(torch.utils.data.Dataset):
    '''包装切分协议产出的样本为数据集。

    每个样本：(item_seq, ts_seq, target_item, target_ts, user_id)
    返回：(hist, pos, target, uid, [timestamps])
    '''

    def __init__(self, samples, num_items):
        self.samples = samples
        self.num_items = num_items

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_seq, ts_seq, target_item, target_ts, user_id = self.samples[idx]
        L = item_seq.size(0)
        pos = torch.arange(L, dtype=torch.long)
        return (
            item_seq,
            pos,
            target_item,
            user_id,
            ts_seq,  # RoTE 模型的原始时间戳
        )


def main():
    parser = argparse.ArgumentParser(description='训练序列推荐模型')
    parser.add_argument('--config', default=None, help='Path to config YAML')
    parser.add_argument('--model', default=None, help='Model name (sasrec, tisasrec, tisasrec_cat, sasrec_rote, tisasrec_rote)')
    parser.add_argument('--split', default=None, help='Split protocol (leave_one_out, no_sss, sliding_window_sss, prefix_target_sss)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=None, help='Output path for results JSON')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['trainer']['epochs'] = args.epochs

    split_protocol = args.split or config.get('data', {}).get('split_protocol', 'leave_one_out')
    split_kwargs = {}
    if split_protocol == 'sliding_window_sss':
        split_kwargs['window_size'] = config.get('data', {}).get('sliding_window_size', 10)
    elif split_protocol == 'prefix_target_sss':
        split_kwargs['prefix_min_len'] = config.get('data', {}).get('prefix_min_len', 3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ---- 合成数据 ----
    random.seed(args.seed)
    num_users, num_items = 100, 50
    sequences, timestamps_dict, item_categories = generate_synthetic_data(
        num_users, num_items, seed=args.seed)

    # 按用户切分 train/val
    split = int(len(sequences) * 0.8)
    users = list(sequences.keys())
    random.shuffle(users)
    train_seq = {u: sequences[u] for u in users[:split]}
    val_seq = {u: sequences[u] for u in users[split:]}
    train_timestamps = {u: timestamps_dict[u] for u in users[:split]}
    val_timestamps = {u: timestamps_dict[u] for u in users[split:]}

    ml = config.get('model', {}).get('max_len', 50)
    bs = config.get('trainer', {}).get('batch_size', 128)
    model_name = config.get('model', {}).get('name', 'sasrec')

    # ---- 训练集：使用切分协议 + 时间戳 ----
    train_ds = build_dataset_with_timestamps(
        train_seq, train_timestamps, ml, num_items,
        split_protocol=split_protocol, **split_kwargs,
    )
    logger.info(f"Split protocol '{split_protocol}': {len(train_ds)} training samples")

    # ---- 验证集：始终使用 leave_one_out + EvalDataset ----
    val_ds = EvalDataset(
        val_seq,
        max_len=ml,
        timestamps=val_timestamps,
        item_categories=item_categories,
        return_timestamps=model_name in ('sasrec_rote', 'tisasrec_rote'),
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    # ---- 收集训练集已交互 item（用于 full-ranking 排除） ----
    train_items = set()
    for seq in train_seq.values():
        train_items.update(seq)
    train_items.add(0)

    # ---- 构建 item_timestamps（每个 item 的最后交互时间） ----
    item_timestamps = {}
    for uid, seq in sequences.items():
        ts_list = timestamps_dict[uid]
        for item, ts in zip(seq, ts_list):
            if item not in item_timestamps or ts > item_timestamps[item]:
                item_timestamps[item] = ts

    # ---- 模型（通过 factory） ----
    model = build_model(model_name, num_items, config)
    logger.info(f"Built model: {model_name} (params: {sum(p.numel() for p in model.parameters()):,})")

    # ---- 训练 ----
    tc = config.get('trainer', {})
    trainer = Trainer(model, train_loader, val_loader, device=device,
        lr=tc.get('lr', 1e-3), epochs=tc.get('epochs', 5),
        patience=tc.get('patience', 3), checkpoint_dir='./checkpoints',
        eval_fn=lambda m, l, d: model_eval(
            m, l, d,
            exclude_items=train_items,
            item_categories=item_categories,
            item_timestamps=item_timestamps,
        ).get('recall@10', 0.0))
    history = trainer.train()

    # ---- 最终评估 ----
    results = model_eval(trainer.model, val_loader, device,
                         exclude_items=train_items,
                         item_categories=item_categories,
                         item_timestamps=item_timestamps)
    print(f'\nEval results [{model_name}] (split={split_protocol}):')
    for k, v in sorted(results.items()):
        print(f'  {k}: {v:.4f}')

    # ---- 可选 JSON 输出 ----
    if args.output:
        output_data = {
            'model': model_name,
            'split_protocol': split_protocol,
            'config': {
                'hidden_dim': config.get('model', {}).get('hidden_dim', 64),
                'num_layers': config.get('model', {}).get('num_layers', 2),
                'epochs': tc.get('epochs', 5),
            },
            'results': {k: round(v, 6) for k, v in results.items()},
            'best_val': round(trainer.best_val, 6),
            'param_count': sum(p.numel() for p in model.parameters()),
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    return history, results


if __name__ == '__main__':
    main()
