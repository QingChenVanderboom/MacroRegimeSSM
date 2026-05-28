"""train.py - 序列化版本"""

import torch
import argparse
from src.data.dataset import build_dataloaders
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.utils.trainer import Trainer


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, N, F, _ = build_dataloaders(
        L=args.L,
        T=args.T,
        batch_size=args.batch_size,
        num_workers=args.num_workers, eval_stride=args.eval_stride,
    )

    model = MacroRegimeSSM(
        n_assets=N, n_macro=F, n_tech=3,
        d_model=args.d_model, n_regimes=args.K,
        n_heads=4, n_layers=2, dropout=args.dropout, L=args.L,
    )
    print(f"参数量: {model.count_parameters():,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=1e-4,
        max_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        lambda2=args.lambda2,
        lambda3=args.lambda3, lambda4=args.lambda4,
        patience=args.patience,
        save_dir='experiments/checkpoints',
        run_name=f'seq_K{args.K}_d{args.d_model}_L{args.L}_T{args.T}',
    )
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K',            type=int,   default=4)
    parser.add_argument('--d_model',      type=int,   default=64)
    parser.add_argument('--L',            type=int,   default=20)
    parser.add_argument('--T',            type=int,   default=128)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--epochs',       type=int,   default=150)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--dropout',      type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int,   default=5000)
    parser.add_argument('--patience',     type=int,   default=20)
    parser.add_argument('--lambda3',      type=float, default=5.0)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--free_bits',    type=float, default=0.1)
    parser.add_argument('--eval_stride',  type=int,   default=5)
    parser.add_argument('--lambda4',      type=float, default=1.0)
    parser.add_argument('--lambda2',      type=float, default=0.1)
    args = parser.parse_args()
    main(args)
