"""
所有模型的交易成本分析
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from src.data.dataset import load_data, load_splits, build_dataloaders
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.models.baselines import LSTMBaseline, TransformerBaseline, MacroLSTMBaseline, LinearBaseline
from train_ablation import NoMacroSSM, NoRegimeSSM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = 'experiments/checkpoints'

returns, tech, macro, N, n_macro, _ = load_data()
splits = load_splits()
idx    = returns.index
val_end = idx.searchsorted(pd.to_datetime(splits['val_end']), side='right')
test_returns = returns.iloc[val_end:].values.astype('float32')
test_tech    = tech.iloc[val_end:].values.astype('float32').reshape(-1, N, 3)
test_macro   = macro.iloc[val_end:].values.astype('float32')

def compute_net_sharpe(preds, actuals, cost_bps):
    T, N = preds.shape
    cost = cost_bps / 10000
    positions = np.zeros((T, N))
    for t in range(T):
        rank = preds[t].argsort()
        positions[t, rank[N//2:]] =  1.0 / (N//2)
        positions[t, rank[:N//2]] = -1.0 / (N//2)
    pnl = (positions * actuals).sum(axis=1)
    if cost > 0 and T > 1:
        turnover = np.abs(positions[1:] - positions[:-1]).sum(axis=1)
        pnl[1:] -= turnover * cost
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
    turnover = np.abs(positions[1:] - positions[:-1]).sum(axis=1).mean()
    return sharpe, turnover

# ── 获取MacroRegimeSSM预测 ────────────────────────────────
def get_ssm_preds(model):
    model.eval()
    L, K = 20, model.K
    preds = []
    h = torch.zeros(1, model.d, device=DEVICE)
    z = torch.full((1, K), 1.0/K, device=DEVICE)
    prev_logits = torch.zeros(1, K, device=DEVICE)
    with torch.no_grad():
        for t in range(L, len(test_returns)):
            xw = torch.from_numpy(test_returns[t-L:t]).unsqueeze(0).to(DEVICE)
            tw = torch.from_numpy(test_tech[t-L:t]).unsqueeze(0).to(DEVICE)
            mw = torch.from_numpy(test_macro[t-L:t]).unsqueeze(0).to(DEVICE)
            xe = model.asset_encoder(xw)
            te = model.tech_encoder(tw)
            me = model.macro_encoder(mw)
            z_emb = model.inference_net.z_embed(z)
            raw = model.inference_net.net(torch.cat([xe, me, z_emb, te], dim=-1))
            smooth = model.momentum * prev_logits + (1 - model.momentum) * raw
            prev_logits = smooth
            soft  = F.softmax(smooth, dim=-1)
            index = soft.argmax(dim=-1)
            z_new = F.one_hot(index, K).float()
            h     = model.film_ssm(xe, h, z_new)
            mu, _ = model.pred_head(h, z_new)
            preds.append(mu.squeeze(0).cpu().numpy())
            z, h = z_new, h.detach()
    return np.array(preds), test_returns[L:]

# ── 获取baseline预测（单步） ──────────────────────────────
def get_baseline_preds(model, is_macro=False):
    model.eval()
    L = 20
    preds = []
    with torch.no_grad():
        for t in range(L, len(test_returns)):
            xw = torch.from_numpy(test_returns[t-L:t]).unsqueeze(0).to(DEVICE)
            tw = torch.from_numpy(test_tech[t-L:t]).unsqueeze(0).to(DEVICE)
            mw = torch.from_numpy(test_macro[t-L:t]).unsqueeze(0).to(DEVICE)
            out = model(xw, tw, mw)
            preds.append(out['mu'].squeeze(0).cpu().numpy())
    return np.array(preds), test_returns[L:]

# ── 获取HMM预测 ───────────────────────────────────────────
def get_hmm_preds():
    from hmmlearn import hmm
    train_end = idx.searchsorted(pd.to_datetime(splits['train_end']), side='right')
    train_r = returns.iloc[:train_end].values
    model_hmm = hmm.GaussianHMM(n_components=4, covariance_type='diag',
                                  n_iter=200, random_state=42)
    model_hmm.fit(train_r)
    states = model_hmm.predict(test_returns)
    preds  = model_hmm.means_[states[:-1]]
    return preds, test_returns[1:]

# ── 运行所有模型 ──────────────────────────────────────────
configs = [
    ('MacroRegimeSSM', 'seq_K4_d64_L20_T64',
     MacroRegimeSSM(n_assets=N, n_macro=n_macro, L=20, n_regimes=4, momentum=0.8),
     'ssm'),
    ('LSTM', 'seq_baseline_lstm_d64_L20_T128',
     LSTMBaseline(n_assets=N), 'baseline'),
    ('Transformer', 'seq_baseline_transformer_d64_L20_T128',
     TransformerBaseline(n_assets=N), 'baseline'),
    ('MacroLSTM', 'seq_baseline_macro_lstm_d64_L20_T128',
     MacroLSTMBaseline(n_assets=N, n_macro=n_macro), 'baseline'),
    ('w/o Macro', 'ablation_no_macro_d64_L20_T64',
     NoMacroSSM(n_assets=N, n_macro=n_macro, L=20), 'ssm'),
    ('w/o Regime', 'ablation_no_regime_d64_L20_T64',
     NoRegimeSSM(n_assets=N, n_macro=n_macro, L=20), 'baseline'),
]

cost_levels = [0, 1, 2, 3, 5]

print(f"\n{'Model':<20} {'Turnover':>10} " +
      " ".join(f"{b}bps".rjust(10) for b in cost_levels))
print("-" * (20 + 12 + 11*len(cost_levels)))

all_results = {}

for name, run_name, model, mode in configs:
    ckpt_path = f'{CKPT}/{run_name}/best.pt'
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'], strict=False)
        model = model.to(DEVICE)

        if mode == 'ssm':
            preds, actuals = get_ssm_preds(model)
        else:
            preds, actuals = get_baseline_preds(model)

        sharpes = []
        for bps in cost_levels:
            s, turnover = compute_net_sharpe(preds, actuals, bps)
            sharpes.append(s)

        _, turnover = compute_net_sharpe(preds, actuals, 0)
        all_results[name] = {'turnover': turnover, 'sharpes': sharpes}

        row = f"{name:<20} {turnover:>10.3f} " + \
              " ".join(f"{s:>10.3f}" for s in sharpes)
        print(row)

    except Exception as e:
        print(f"{name:<20} ERROR: {e}")

# HMM
try:
    preds, actuals = get_hmm_preds()
    sharpes = []
    for bps in cost_levels:
        s, turnover = compute_net_sharpe(preds, actuals, bps)
        sharpes.append(s)
    _, turnover = compute_net_sharpe(preds, actuals, 0)
    row = f"{'HMM':<20} {turnover:>10.3f} " + \
          " ".join(f"{s:>10.3f}" for s in sharpes)
    print(row)
except Exception as e:
    print(f"HMM ERROR: {e}")

print(f"\nCost levels: {cost_levels} bps (one-way)")
