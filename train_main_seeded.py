"""
Train MacroRegimeSSM (K=4) with the hyperparameters from train_ablation.py
(which match the paper: lambda2=0.3, lambda4=2.0, warmup_steps=3000).
Usage: python train_main_seeded.py --seed 42
"""
import random
import numpy as np
import torch
import argparse
from src.data.dataset import build_dataloaders
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.utils.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Seed: {args.seed} | Device: {device}")

    # ── DataLoaders (from train_ablation.py) ──
    train_loader, val_loader, _, N, F, _ = build_dataloaders(
        L=args.L, T=args.T, batch_size=args.batch_size,
        num_workers=args.num_workers, eval_stride=5,
    )

    # ── MacroRegimeSSM (K=4) ──
    model = MacroRegimeSSM(
        n_assets=N, n_macro=F, n_tech=3,
        d_model=args.d_model, n_regimes=4,
        n_heads=4, n_layers=2, dropout=args.dropout, L=args.L,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # ── Trainer (hyperparams from train_ablation.py = paper values) ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
        warmup_steps=args.warmup_steps,    # 5000 (train.py)
        lambda2=args.lambda2,              # 0.1  (train.py)
        lambda3=args.lambda3,              # 5.0
        lambda4=args.lambda4,              # 1.0  (train.py)
        patience=args.patience,
        save_dir='experiments/checkpoints',
        run_name=f'seed{args.seed}_K4_T{args.T}_v2',
    )
    trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)     # paper value
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=5000) # train.py default
    parser.add_argument('--lambda3', type=float, default=5.0)
    parser.add_argument('--lambda4', type=float, default=1.0)     # train.py default
    parser.add_argument('--lambda2', type=float, default=0.1)     # train.py default
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    main(args)
