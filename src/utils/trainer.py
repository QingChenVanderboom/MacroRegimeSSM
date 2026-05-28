"""
trainer.py - 加入entropy正则版本
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
import time

from src.models.macro_regime_ssm import MacroRegimeSSM
from src.utils.losses import elbo_loss, kl_annealing_weight, gumbel_tau_annealing
from src.utils.metrics import evaluate


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device:        str   = 'cuda',
        lr:            float = 1e-3,
        weight_decay:  float = 1e-4,
        max_epochs:    int   = 100,
        warmup_steps:  int   = 5000,
        tau_start:     float = 1.0,
        tau_end:       float = 0.5,   # 不退火到0.1，保持一定软度防collapse
        lambda2:       float = 0.1,
        lambda3:       float = 5.0,
        lambda4:       float = 1.0,   # entropy正则权重
        patience:      int   = 20,
        save_dir:      str   = 'experiments/checkpoints',
        run_name:      str   = 'default',
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.max_epochs   = max_epochs
        self.warmup_steps = warmup_steps
        self.tau_start    = tau_start
        self.tau_end      = tau_end
        self.lambda2      = lambda2
        self.lambda3      = lambda3
        self.lambda4      = lambda4
        self.patience     = patience

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=lr * 0.01
        )

        self.save_dir = Path(save_dir) / run_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.global_step       = 0
        self.best_val_nll      = float('inf')
        self.epochs_no_improve = 0
        self.history = {
            'train_nll':  [], 'train_kl':   [], 'train_rank': [],
            'train_ent':  [],
            'val_nll':    [], 'val_ic':     [], 'val_icir':   [],
            'val_sharpe': [],
        }

    def _get_tau(self):
        total = self.max_epochs * len(self.train_loader)
        return gumbel_tau_annealing(
            self.global_step, total, self.tau_start, self.tau_end
        )

    def _get_lambda1(self):
        return kl_annealing_weight(self.global_step, self.warmup_steps)

    def _forward(self, batch, tau, hard=False):
        x_seq    = batch['x_seq'].to(self.device)
        tech_seq = batch['tech_seq'].to(self.device)
        m_seq    = batch['m_seq'].to(self.device)
        x_target = batch['x_target'].to(self.device)
        out = self.model(x_seq, tech_seq, m_seq, tau=tau, hard=hard)
        return out, x_target

    def train_epoch(self):
        self.model.train()
        sums = {'nll': 0., 'regime_kl': 0., 'rank_loss': 0., 'entropy': 0.}
        n = 0

        for batch in self.train_loader:
            tau     = self._get_tau()
            lambda1 = self._get_lambda1()

            self.optimizer.zero_grad()
            out, x_target = self._forward(batch, tau)

            losses = elbo_loss(
                x_target, out['mu'], out['log_sigma'],
                out['z_seq'], out['prior_seq'], out['z_prev_seq'],
                lambda1=lambda1, lambda2=self.lambda2,
                lambda3=self.lambda3, lambda4=self.lambda4,
            )
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.optimizer.step()

            for k in sums:
                sums[k] += losses[k].item()
            n += 1
            self.global_step += 1

        return {k: v / n for k, v in sums.items()}

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        total_nll = 0.
        all_preds, all_actuals = [], []
        n = 0

        for batch in self.val_loader:
            out, x_target = self._forward(batch, tau=self.tau_end)
            losses = elbo_loss(
                x_target, out['mu'], out['log_sigma'],
                out['z_seq'], out['prior_seq'], out['z_prev_seq'],
                lambda1=1.0, lambda2=self.lambda2,
                lambda3=self.lambda3, lambda4=self.lambda4,
            )
            total_nll += losses['nll'].item()
            n += 1

            B, T, N = out['mu'].shape
            all_preds.append(out['mu'].reshape(B*T, N).cpu().numpy())
            all_actuals.append(x_target.reshape(B*T, N).cpu().numpy())

        preds   = np.concatenate(all_preds,   axis=0)
        actuals = np.concatenate(all_actuals, axis=0)
        metrics = evaluate(preds, actuals)

        return {
            'nll':    total_nll / n,
            'IC':     metrics['IC_mean'],
            'ICIR':   metrics['ICIR'],
            'Sharpe': metrics['Sharpe'],
        }

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch':        epoch,
            'global_step':  self.global_step,
            'model_state':  self.model.state_dict(),
            'optim_state':  self.optimizer.state_dict(),
            'best_val_nll': self.best_val_nll,
            'history':      self.history,
        }
        torch.save(ckpt, self.save_dir / 'last.pt')
        if is_best:
            torch.save(ckpt, self.save_dir / 'best.pt')

    def fit(self):
        print(f"开始训练 | 设备:{self.device} | "
              f"train:{len(self.train_loader)}batch | "
              f"val:{len(self.val_loader)}batch")
        print(f"保存路径: {self.save_dir}\n")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch()
            val_m   = self.val_epoch()
            self.scheduler.step()

            self.history['train_nll'].append(train_m['nll'])
            self.history['train_kl'].append(train_m['regime_kl'])
            self.history['train_rank'].append(train_m['rank_loss'])
            self.history['train_ent'].append(train_m['entropy'])
            self.history['val_nll'].append(val_m['nll'])
            self.history['val_ic'].append(val_m['IC'])
            self.history['val_icir'].append(val_m['ICIR'])
            self.history['val_sharpe'].append(val_m['Sharpe'])

            is_best = val_m['nll'] < self.best_val_nll
            if is_best:
                self.best_val_nll      = val_m['nll']
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, is_best)

            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"nll={train_m['nll']:.4f} "
                f"kl={train_m['regime_kl']:.3f} "
                f"ent={train_m['entropy']:.3f} "
                f"rank={train_m['rank_loss']:.3f} | "
                f"val_nll={val_m['nll']:.4f} "
                f"IC={val_m['IC']:.4f} "
                f"ICIR={val_m['ICIR']:.3f} | "
                f"τ={self._get_tau():.3f} λ1={self._get_lambda1():.2f} | "
                f"{'★ ' if is_best else ''}{time.time()-t0:.1f}s"
            )

            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping（{self.patience}轮无改善）")
                break

        print(f"\n训练完成，best val NLL: {self.best_val_nll:.4f}")
        return self.history
