"""
evaluate.py - 完整版，含所有baseline、消融、K敏感性分析
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.dataset import build_dataloaders
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.models.baselines import LSTMBaseline, TransformerBaseline, MacroLSTMBaseline, LinearBaseline
from train_ablation import NoMacroSSM, NoRegimeSSM
from src.utils.metrics import evaluate
from src.utils.losses import gaussian_nll

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = Path('experiments/checkpoints')


def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def eval_sequential(model, loader, is_ours=False, L=20):
    all_preds, all_actuals = [], []
    total_nll = 0.
    n = 0

    for batch in loader:
        x_seq    = batch['x_seq'].to(DEVICE)
        tech_seq = batch['tech_seq'].to(DEVICE)
        m_seq    = batch['m_seq'].to(DEVICE)
        x_target = batch['x_target'].to(DEVICE)

        if is_ours:
            out = model(x_seq, tech_seq, m_seq, tau=0.1)
        else:
            B, LT, N = x_seq.shape
            T = x_target.shape[1]
            mu_list, ls_list = [], []
            for t in range(T):
                o = model(x_seq[:, t:t+L], tech_seq[:, t:t+L], m_seq[:, t:t+L])
                mu_list.append(o['mu'])
                ls_list.append(o['log_sigma'])
            out = {
                'mu':        torch.stack(mu_list, dim=1),
                'log_sigma': torch.stack(ls_list, dim=1),
            }

        nll = gaussian_nll(x_target, out['mu'], out['log_sigma'])
        total_nll += nll.item()
        n += 1

        B, T, N = out['mu'].shape
        all_preds.append(out['mu'].reshape(B*T, N).cpu().numpy())
        all_actuals.append(x_target.reshape(B*T, N).cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    metrics = evaluate(preds, actuals)
    metrics['NLL'] = total_nll / n
    return metrics


def main():
    _, _, test_loader, N, F, _ = build_dataloaders(
        L=20, T=64, batch_size=8,
        num_workers=4, eval_stride=5,
    )
    print(f"Test集样本数: {len(test_loader.dataset)}\n")

    configs = [
        # ── 我们的模型 ──
        ('MacroRegimeSSM (K=4)',
         'seq_K4_d64_L20_T64', True,
         MacroRegimeSSM(n_assets=N, n_macro=F, L=20, n_regimes=4, momentum=0.8)),

        ('MacroRegimeSSM (K=3)',
         'seq_K3_d64_L20_T64', True,
         MacroRegimeSSM(n_assets=N, n_macro=F, L=20, n_regimes=3, momentum=0.8)),

        ('MacroRegimeSSM (K=5)',
         'seq_K5_d64_L20_T64', True,
         MacroRegimeSSM(n_assets=N, n_macro=F, L=20, n_regimes=5, momentum=0.8)),

        # ── Baseline ──
        ('Linear',
         'seq_baseline_linear_d64_L20_T64', False,
         LinearBaseline(n_assets=N, L=20)),

        ('LSTM',
         'seq_baseline_lstm_d64_L20_T128', False,
         LSTMBaseline(n_assets=N)),

        ('Transformer',
         'seq_baseline_transformer_d64_L20_T128', False,
         TransformerBaseline(n_assets=N)),

        ('MacroLSTM',
         'seq_baseline_macro_lstm_d64_L20_T128', False,
         MacroLSTMBaseline(n_assets=N, n_macro=F)),

        # ── 消融实验 ──
        ('Ablation: w/o Macro',
         'ablation_no_macro_d64_L20_T64', True,
         NoMacroSSM(n_assets=N, n_macro=F, L=20)),

        ('Ablation: w/o Regime',
         'ablation_no_regime_d64_L20_T64', True,
         NoRegimeSSM(n_assets=N, n_macro=F, L=20)),
    ]

    results = {}
    for name, run_name, is_ours, model in configs:
        ckpt_path = CKPT_DIR / run_name / 'best.pt'
        if not ckpt_path.exists():
            print(f"[SKIP] {name}: {ckpt_path} 不存在")
            continue
        model   = load_model(model, ckpt_path)
        metrics = eval_sequential(model, test_loader, is_ours=is_ours)
        results[name] = metrics
        print(f"{name:30s} | NLL={metrics['NLL']:7.4f} | "
              f"IC={metrics['IC_mean']:.4f} | "
              f"ICIR={metrics['ICIR']:.4f} | "
              f"Sharpe={metrics['Sharpe']:.4f} | "
              f"MSE={metrics['MSE']:.6f}")

    if results:
        print("\n" + "="*80)
        print("Test集最终结果汇总")
        print("="*80)
        df = pd.DataFrame(results).T
        df = df[['NLL', 'IC_mean', 'IC_std', 'ICIR', 'Sharpe', 'MSE', 'MAE']]
        df.columns = ['NLL', 'IC', 'IC_std', 'ICIR', 'Sharpe', 'MSE', 'MAE']
        print(df.round(4).to_string())
        df.to_csv('experiments/test_results.csv')
        print(f"\n已保存至 experiments/test_results.csv")


if __name__ == '__main__':
    main()
