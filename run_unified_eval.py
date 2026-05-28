"""
统一用逐步单步推理方式计算所有模型的Sharpe/IC/ICIR/NLL
与交易成本分析使用相同的评估框架
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from src.data.dataset import load_data, load_splits
from src.models.macro_regime_ssm import MacroRegimeSSM
from src.models.baselines import LSTMBaseline, TransformerBaseline, MacroLSTMBaseline, LinearBaseline
from train_ablation import NoMacroSSM, NoRegimeSSM
from src.utils.metrics import evaluate
import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT   = 'experiments/checkpoints'

returns, tech, macro, N, n_macro, _ = load_data()
splits = load_splits()
idx    = returns.index
val_end = idx.searchsorted(pd.to_datetime(splits['val_end']), side='right')
test_returns = returns.iloc[val_end:].values.astype('float32')
test_tech    = tech.iloc[val_end:].values.astype('float32').reshape(-1, N, 3)
test_macro   = macro.iloc[val_end:].values.astype('float32')

L = 20

def gaussian_nll_np(actuals, preds_mu, preds_sigma=None):
    if preds_sigma is None:
        preds_sigma = np.ones_like(preds_mu) * actuals.std()
    nll = 0.5 * (np.log(2*math.pi) + 2*np.log(preds_sigma+1e-8) +
                 ((actuals - preds_mu)/preds_sigma)**2)
    return nll.mean()

def get_ssm_preds(model):
    model.eval()
    K = model.K
    preds_mu, preds_sigma, actuals_list = [], [], []
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
            smooth = model.momentum * prev_logits + (1-model.momentum) * raw
            prev_logits = smooth
            soft  = F.softmax(smooth, dim=-1)
            index = soft.argmax(dim=-1)
            z_new = F.one_hot(index, K).float()
            h     = model.film_ssm(xe, h, z_new)
            mu, ls = model.pred_head(h, z_new)
            preds_mu.append(mu.squeeze(0).cpu().numpy())
            preds_sigma.append(ls.exp().squeeze(0).cpu().numpy())
            actuals_list.append(test_returns[t])
            z, h = z_new, h.detach()
    return np.array(preds_mu), np.array(preds_sigma), np.array(actuals_list)

def get_baseline_preds(model):
    model.eval()
    preds_mu, preds_sigma, actuals_list = [], [], []
    with torch.no_grad():
        for t in range(L, len(test_returns)):
            xw = torch.from_numpy(test_returns[t-L:t]).unsqueeze(0).to(DEVICE)
            tw = torch.from_numpy(test_tech[t-L:t]).unsqueeze(0).to(DEVICE)
            mw = torch.from_numpy(test_macro[t-L:t]).unsqueeze(0).to(DEVICE)
            out = model(xw, tw, mw)
            preds_mu.append(out['mu'].squeeze(0).cpu().numpy())
            preds_sigma.append(out['log_sigma'].exp().squeeze(0).cpu().numpy())
            actuals_list.append(test_returns[t])
    return np.array(preds_mu), np.array(preds_sigma), np.array(actuals_list)

configs = [
    ('MacroRegimeSSM', 'seq_K4_d64_L20_T64',
     MacroRegimeSSM(n_assets=N, n_macro=n_macro, L=20, n_regimes=4, momentum=0.8), 'ssm'),
    ('LSTM', 'seq_baseline_lstm_d64_L20_T128',
     LSTMBaseline(n_assets=N), 'baseline'),
    ('Transformer', 'seq_baseline_transformer_d64_L20_T128',
     TransformerBaseline(n_assets=N), 'baseline'),
    ('MacroLSTM', 'seq_baseline_macro_lstm_d64_L20_T128',
     MacroLSTMBaseline(n_assets=N, n_macro=n_macro), 'baseline'),
    ('HMM', None, None, 'hmm'),
    ('Linear', 'seq_baseline_linear_d64_L20_T64',
     LinearBaseline(n_assets=N, L=L), 'baseline'),
    ('w/o Macro', 'ablation_no_macro_d64_L20_T64',
     NoMacroSSM(n_assets=N, n_macro=n_macro, L=20), 'ssm'),
    ('w/o Regime', 'ablation_no_regime_d64_L20_T64',
     NoRegimeSSM(n_assets=N, n_macro=n_macro, L=20), 'baseline'),
]

print(f"\n{'Model':<22} {'NLL':>8} {'IC':>8} {'ICIR':>8} {'Sharpe':>8} {'MSE':>10}")
print("-"*70)

for name, run_name, model, mode in configs:
    try:
        if mode == 'hmm':
            from hmmlearn import hmm
            train_end = idx.searchsorted(pd.to_datetime(splits['train_end']), side='right')
            hmm_model = hmm.GaussianHMM(n_components=4, covariance_type='diag',
                                         n_iter=200, random_state=42)
            hmm_model.fit(returns.iloc[:train_end].values)
            states = hmm_model.predict(test_returns)
            mu = hmm_model.means_[states]
            sigma = np.sqrt(hmm_model.covars_[states])
            actuals = test_returns
            nll = gaussian_nll_np(actuals, mu, sigma)
            metrics = evaluate(mu, actuals)
        else:
            ckpt = torch.load(f'{CKPT}/{run_name}/best.pt', map_location=DEVICE)
            model.load_state_dict(ckpt['model_state'], strict=False)
            model = model.to(DEVICE)
            if mode == 'ssm':
                mu, sigma, actuals = get_ssm_preds(model)
            else:
                mu, sigma, actuals = get_baseline_preds(model)
            nll = gaussian_nll_np(actuals, mu, sigma)
            metrics = evaluate(mu, actuals)

        print(f"{name:<22} {nll:>8.4f} {metrics['IC_mean']:>8.4f} "
              f"{metrics['ICIR']:>8.4f} {metrics['Sharpe']:>8.4f} "
              f"{metrics['MSE']:>10.6f}")
    except Exception as e:
        print(f"{name:<22} ERROR: {e}")
