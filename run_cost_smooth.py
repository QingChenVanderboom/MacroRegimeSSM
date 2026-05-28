import torch, torch.nn.functional as F
import numpy as np, pandas as pd
from src.data.dataset import load_data, load_splits
from src.models.macro_regime_ssm import MacroRegimeSSM

returns, tech, macro, N, n_macro, _ = load_data()
splits = load_splits()
idx = returns.index
val_end = idx.searchsorted(pd.to_datetime(splits['val_end']), side='right')
test_returns = returns.iloc[val_end:].values.astype('float32')
test_tech    = tech.iloc[val_end:].values.astype('float32').reshape(-1, N, 3)
test_macro   = macro.iloc[val_end:].values.astype('float32')

DEVICE = 'cuda'
model = MacroRegimeSSM(n_assets=N, n_macro=n_macro, L=20, n_regimes=4, momentum=0.8).to(DEVICE)
ckpt = torch.load('experiments/checkpoints/seq_K4_d64_L20_T64/best.pt', map_location=DEVICE)
model.load_state_dict(ckpt['model_state'], strict=False)
model.eval()

K, L = 4, 20
all_mu, all_actual, smooth_regimes = [], [], []
h = torch.zeros(1, model.d, device=DEVICE)
z = torch.full((1,K), 0.25, device=DEVICE)
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
        smooth_idx = F.softmax(smooth, dim=-1).argmax().item()
        smooth_regimes.append(smooth_idx)
        z_new = F.one_hot(torch.tensor(smooth_idx), K).float().unsqueeze(0).to(DEVICE)
        h = model.film_ssm(xe, h, z_new)
        mu, _ = model.pred_head(h, z_new)
        all_mu.append(mu.squeeze(0).cpu().numpy())
        all_actual.append(test_returns[t])
        z, h = z_new, h.detach()

preds   = np.array(all_mu)
actuals = np.array(all_actual)
regimes = np.array(smooth_regimes)

smo_r = pd.Series(regimes).rolling(5, center=True, min_periods=1).apply(
    lambda x: pd.Series(x).mode()[0]
).astype(int).values

positions = np.zeros_like(preds)
cur_pos = np.zeros(N)
for t in range(len(preds)):
    if t == 0 or smo_r[t] != smo_r[t-1]:
        rank = preds[t].argsort()
        cur_pos = np.zeros(N)
        cur_pos[rank[N//2:]] =  1.0/(N//2)
        cur_pos[rank[:N//2]] = -1.0/(N//2)
    positions[t] = cur_pos

pnl_gross = (positions * actuals).sum(axis=1)
turnover  = np.abs(positions[1:] - positions[:-1]).sum(axis=1)

print("5日平滑信号交易成本分析:")
print("日均换手率: %.4f" % turnover.mean())
print("%-12s %-12s %-12s" % ("成本(bps)", "净Sharpe", "年化收益"))
for bps in [0, 1, 2, 3, 5]:
    pnl = pnl_gross.copy()
    if bps > 0:
        pnl[1:] -= turnover * bps/10000
    sharpe = pnl.mean()/(pnl.std()+1e-8)*np.sqrt(252)
    ann_ret = pnl.mean()*252
    print("%-12d %-12.4f %-12.4f" % (bps, sharpe, ann_ret))
