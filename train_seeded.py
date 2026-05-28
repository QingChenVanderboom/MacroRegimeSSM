"""
Single-seed train + unified eval for MacroRegimeSSM (K=4).
Usage: python train_seeded.py --seed 42
"""
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math
import argparse
from src.data.dataset import build_dataloaders, load_data, load_splits
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.utils.trainer import Trainer
from src.utils.metrics import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def gaussian_nll_np(actuals, preds_mu, preds_sigma=None):
    if preds_sigma is None:
        preds_sigma = np.ones_like(preds_mu) * actuals.std()
    nll = 0.5 * (np.log(2 * math.pi) + 2 * np.log(preds_sigma + 1e-8) +
                 ((actuals - preds_mu) / preds_sigma) ** 2)
    return nll.mean()


@torch.no_grad()
def unified_eval(model, device, test_returns, test_tech, test_macro):
    """Exactly replicates run_unified_eval.py's online single-step evaluation."""
    model.eval()
    L, K = 20, model.K
    preds_mu, preds_sigma, actuals_list = [], [], []
    h = torch.zeros(1, model.d, device=device)
    z = torch.full((1, K), 1.0 / K, device=device)
    prev_logits = torch.zeros(1, K, device=device)

    for t in range(L, len(test_returns)):
        xw = torch.from_numpy(test_returns[t - L:t]).unsqueeze(0).to(device)
        tw = torch.from_numpy(test_tech[t - L:t]).unsqueeze(0).to(device)
        mw = torch.from_numpy(test_macro[t - L:t]).unsqueeze(0).to(device)

        xe = model.asset_encoder(xw)
        te = model.tech_encoder(tw)
        me = model.macro_encoder(mw)

        z_emb = model.inference_net.z_embed(z)
        raw = model.inference_net.net(torch.cat([xe, me, z_emb, te], dim=-1))
        smooth = model.momentum * prev_logits + (1 - model.momentum) * raw
        prev_logits = smooth

        soft = F.softmax(smooth, dim=-1)
        index = soft.argmax(dim=-1)
        z_new = F.one_hot(index, K).float()
        h = model.film_ssm(xe, h, z_new)
        mu, ls = model.pred_head(h, z_new)

        preds_mu.append(mu.squeeze(0).cpu().numpy())
        preds_sigma.append(ls.exp().squeeze(0).cpu().numpy())
        actuals_list.append(test_returns[t])
        z, h = z_new, h.detach()

    mu = np.array(preds_mu)
    sigma = np.array(preds_sigma)
    actuals = np.array(actuals_list)

    nll = gaussian_nll_np(actuals, mu, sigma)
    metrics = evaluate(mu, actuals)
    return nll, metrics


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Seed: {args.seed} | Device: {device}")

    # ── DataLoaders ──
    train_loader, val_loader, test_loader, N, F, _ = build_dataloaders(
        L=args.L, T=args.T, batch_size=args.batch_size,
        num_workers=args.num_workers, eval_stride=args.eval_stride,
    )

    # ── Model ──
    model = MacroRegimeSSM(
        n_assets=N, n_macro=F, n_tech=3,
        d_model=args.d_model, n_regimes=args.K,
        n_heads=4, n_layers=2, dropout=args.dropout, L=args.L,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # ── Trainer ──
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr, weight_decay=1e-4,
        max_epochs=args.epochs, warmup_steps=args.warmup_steps,
        lambda2=args.lambda2, lambda3=args.lambda3, lambda4=args.lambda4,
        patience=args.patience,
        save_dir='experiments/checkpoints',
        run_name=f'seed{args.seed}_K{args.K}_d{args.d_model}_L{args.L}_T{args.T}',
    )
    trainer.fit()

    # ── Load best checkpoint ──
    ckpt_path = trainer.save_dir / 'best.pt'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)

    # ── Unified eval ──
    returns, tech, macro, _, _, _ = load_data()
    splits = load_splits()
    idx = returns.index
    val_end = idx.searchsorted(pd.to_datetime(splits['val_end']), side='right')

    test_returns = returns.iloc[val_end:].values.astype('float32')
    test_tech = tech.iloc[val_end:].values.astype('float32').reshape(-1, N, 3)
    test_macro = macro.iloc[val_end:].values.astype('float32')

    nll, metrics = unified_eval(model, device, test_returns, test_tech, test_macro)

    print(f"\n{'='*60}")
    print(f"RESULTS seed={args.seed} | NLL={nll:.4f} | IC={metrics['IC_mean']:.4f} | "
          f"ICIR={metrics['ICIR']:.4f} | Sharpe={metrics['Sharpe']:.4f} | MSE={metrics['MSE']:.6f}")
    print(f"{'='*60}")

    # Write a one-line result file for easy aggregation
    with open(f'experiments/seed{args.seed}_result.txt', 'w') as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"NLL={nll:.6f}\n")
        f.write(f"IC={metrics['IC_mean']:.6f}\n")
        f.write(f"ICIR={metrics['ICIR']:.6f}\n")
        f.write(f"Sharpe={metrics['Sharpe']:.6f}\n")
        f.write(f"MSE={metrics['MSE']:.8f}\n")

    print(f"Saved to experiments/seed{args.seed}_result.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--L', type=int, default=20)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lambda3', type=float, default=5.0)
    parser.add_argument('--lambda4', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_stride', type=int, default=5)
    args = parser.parse_args()
    main(args)
