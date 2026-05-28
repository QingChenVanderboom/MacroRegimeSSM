"""
train_baseline.py - 序列化版本
baseline模型也用序列化训练，公平对比
"""

import torch
import argparse
from src.data.dataset import build_dataloaders
from src.models.baselines import (
    LinearBaseline, LSTMBaseline,
    TransformerBaseline, MacroLSTMBaseline
)
from src.utils.trainer import Trainer


BASELINES = {
    'linear':      LinearBaseline,
    'lstm':        LSTMBaseline,
    'transformer': TransformerBaseline,
    'macro_lstm':  MacroLSTMBaseline,
}


class BaselineTrainer(Trainer):
    """
    Baseline的trainer：forward接口不同，输出shape也不同
    baseline只看最后一个时间步的输入，预测下一步
    但为了序列化公平对比，在T步上滚动预测
    """
    def _forward(self, batch, tau, hard=False):
        x_seq    = batch['x_seq'].to(self.device)      # (B, L+T, N)
        tech_seq = batch['tech_seq'].to(self.device)
        m_seq    = batch['m_seq'].to(self.device)
        x_target = batch['x_target'].to(self.device)   # (B, T, N)

        B, LT, N = x_seq.shape
        L = LT - x_target.shape[1]
        T = x_target.shape[1]

        # 在T步上滚动，每步取[t:t+L]窗口
        mu_list, log_sigma_list = [], []
        for t in range(T):
            x_win    = x_seq[:, t:t+L, :]
            tech_win = tech_seq[:, t:t+L, :, :]
            m_win    = m_seq[:, t:t+L, :]

            out_t = self.model(x_win, tech_win, m_win)
            mu_list.append(out_t['mu'])
            log_sigma_list.append(out_t['log_sigma'])

        out = {
            'mu':         torch.stack(mu_list,        dim=1),  # (B, T, N)
            'log_sigma':  torch.stack(log_sigma_list, dim=1),
            'z_seq':      torch.zeros(B, T, 1, device=self.device),
            'prior_seq':  torch.ones(B, T, 1, device=self.device),
            'z_prev_seq': torch.zeros(B, T, 1, device=self.device),
        }
        return out, x_target


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}, GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")

    train_loader, val_loader, test_loader, N, F, _ = build_dataloaders(
        L=args.L, T=args.T,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_stride=args.eval_stride,
    )

    for name, ModelClass in BASELINES.items():
        if args.model != 'all' and args.model != name:
            continue

        print(f"\n{'='*50}\n训练 Baseline: {name}\n{'='*50}")

        model = ModelClass(
            n_assets=N, n_macro=F,
            d_model=64, n_layers=2, dropout=0.1, L=args.L,
        )
        print(f"参数量: {model.count_parameters():,}")

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            max_epochs=args.epochs,
            warmup_steps=1000,
            lambda2=0.0,
            lambda3=args.lambda3,
            patience=args.patience,
            save_dir='experiments/checkpoints',
            run_name=f'seq_baseline_{name}_d64_L{args.L}_T{args.T}',
        )
        trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str,   default='all',
                        choices=['all','linear','lstm','transformer','macro_lstm'])
    parser.add_argument('--L',           type=int,   default=20)
    parser.add_argument('--T',           type=int,   default=128)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--lambda3',     type=float, default=5.0)
    parser.add_argument('--patience',    type=int,   default=20)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--eval_stride',  type=int,   default=5)
    args = parser.parse_args()
    main(args)
