# Trainer for sequential recommendation models.
# SASRec / TiSASRec / TiSASRec-Cat
# Early stopping, checkpoint, LR scheduling, metric logging

import os
import logging
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = 'cpu',
        epochs: int = 50,
        patience: int = 5,
        eval_fn: Optional[Callable] = None,
        checkpoint_dir: str = './checkpoints',
        model_name: str = 'best',
        clip_grad_norm: float = 1.0,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 3,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.eval_fn = eval_fn
        self.clip_grad_norm = clip_grad_norm
        self.model_name = model_name

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.best_val = 0.0
        self.best_state = None
        self.early_stop_counter = 0
        self.current_epoch = 0

        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def _needs_time_deltas(self):
        from .models.tisasrec import TiSASRec
        from .models.tisasrec_cat import TiSASRecCat
        return isinstance(self.model, (TiSASRec, TiSASRecCat))

    @property
    def _needs_cat_mask(self):
        from .models.tisasrec_cat import TiSASRecCat
        return isinstance(self.model, TiSASRecCat)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            if len(batch) == 4:
                hist, pos, target, uid = batch
            else:
                hist, pos, target, uid, time_deltas = batch

            hist = hist.to(self.device)
            pos = pos.to(self.device)
            target = target.to(self.device)

            if self._needs_time_deltas and len(batch) == 5:
                td = time_deltas.to(self.device)
            elif self._needs_time_deltas:
                td = torch.zeros(hist.size(0), hist.size(1), hist.size(1), device=self.device)
            else:
                td = None

            if self._needs_cat_mask:
                # Default: all cross-category (False), making TiSASRecCat act like TiSASRec
                # when no real category info is available
                cm = torch.zeros(hist.size(0), hist.size(1), hist.size(1), dtype=torch.bool, device=self.device)
                logits = self.model(hist, pos, td, cm)
            elif self._needs_time_deltas:
                logits = self.model(hist, pos, td)
            else:
                logits = self.model(hist, pos)

            loss = self.criterion(logits, target)

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self) -> Dict[str, float]:
        history = {'loss': [], 'val_metric': []}

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            loss = self.train_epoch()
            history['loss'].append(loss)

            val_metric = 0.0
            if self.val_loader and self.eval_fn:
                val_metric = self.eval_fn(self.model, self.val_loader, self.device)
                history['val_metric'].append(val_metric)
                self.scheduler.step(val_metric)

                if val_metric > self.best_val:
                    self.best_val = val_metric
                    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self._save_checkpoint(val_metric)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

                if self.early_stop_counter >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch} (best: {self.best_val:.4f})')
                    self._restore_best()
                    break

            msg = f'Epoch {epoch}/{self.epochs} | loss: {loss:.4f}'
            if val_metric: msg += f' | val: {val_metric:.4f}'
            logger.info(msg)
            print(msg)

        if epoch == self.epochs and self.best_state:
            self._restore_best()

        return history

    def _save_checkpoint(self, metric: float) -> str:
        path = os.path.join(self.checkpoint_dir, f'{self.model_name}.pt')
        torch.save({
            'epoch': self.current_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val': self.best_val,
        }, path)
        logger.info(f'Checkpoint saved: {path} (val: {metric:.4f})')
        return path

    def _restore_best(self) -> None:
        if self.best_state:
            self.model.load_state_dict(self.best_state)
            logger.info('Restored best model weights')

    def predict(self, seqs, positions, time_deltas=None, same_cat_mask=None):
        self.model.eval()
        with torch.no_grad():
            seqs = seqs.to(self.device)
            positions = positions.to(self.device)
            if self._needs_cat_mask and same_cat_mask is not None:
                cm = same_cat_mask.to(self.device)
                return self.model(seqs, positions, time_deltas.to(self.device) if time_deltas is not None else None, cm)
            elif self._needs_time_deltas and time_deltas is not None:
                return self.model(seqs, positions, time_deltas.to(self.device))
            else:
                return self.model(seqs, positions)

    def save(self, path: str) -> None:
        torch.save({'model_state': self.model.state_dict()}, path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device)['model_state'])
