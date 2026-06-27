'''Train a sequential recommendation model.
Usage: python train_model.py --config ../configs/default.yaml
'''

import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.data.loader import SeqRecDataset, EvalDataset
from src.models.sasrec import SASRec
from src.models.tisasrec import TiSASRec
from src.models.tisasrec_cat import TiSASRecCat
from src.trainer import Trainer
from src.eval.metrics import model_eval

def build_model(name, num_items, config):
    mc = config.get('model', {})
    if name == 'sasrec':
        return SASRec(num_items, mc.get('hidden_dim', 64))
    elif name == 'tisasrec':
        return TiSASRec(num_items, mc.get('hidden_dim', 64))
    elif name == 'tisasrec_cat':
        return TiSASRecCat(num_items, mc.get('hidden_dim', 64))
    raise ValueError(f'Unknown model: {name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model: config['model']['name'] = args.model
    if args.epochs: config['trainer']['epochs'] = args.epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(level=logging.INFO)

    import random; random.seed(42)
    num_users, num_items = 100, 50
    sequences = {u: [random.randint(1, num_items) for _ in range(random.randint(5, 20))] for u in range(num_users)}
    split = int(len(sequences) * 0.8)
    users = list(sequences.keys())
    train_seq = {u: sequences[u] for u in users[:split]}
    val_seq = {u: sequences[u] for u in users[split:]}

    ml = config.get('model', {}).get('max_len', 50)
    bs = config.get('trainer', {}).get('batch_size', 128)
    train_ds = SeqRecDataset(train_seq, max_len=ml, num_items=num_items)
    val_ds = EvalDataset(val_seq, max_len=ml)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)

    model = build_model(config.get('model', {}).get('name', 'sasrec'), num_items, config)
    tc = config.get('trainer', {})
    trainer = Trainer(model, train_loader, val_loader, device=device,
        lr=tc.get('lr', 1e-3), epochs=tc.get('epochs', 5),
        patience=tc.get('patience', 3), checkpoint_dir='./checkpoints',
        eval_fn=lambda m, l, d: model_eval(m, l, d).get('recall@10', 0.0))
    history = trainer.train()
    results = model_eval(trainer.model, val_loader, device)
    print('Eval results:', results)
    return history, results

if __name__ == '__main__': main()
