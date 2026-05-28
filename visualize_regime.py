"""
visualize_regime.py - 优化版
1. Timeline改成单行彩色时间条
2. 底部子图改为每个regime下的平均收益率
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.data.dataset import load_data, load_splits
from src.models.macro_regime_ssm import MacroRegimeSSM

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = Path('experiments/checkpoints')
FIG_DIR  = Path('experiments/figures')
FIG_DIR.mkdir(exist_ok=True)

REGIME_COLORS = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
K = 4


@torch.no_grad()
def get_regime_sequence(model, test_returns, test_tech, test_macro, test_dates):
    L = model.L
    all_regimes = []
    all_dates   = []

    h = torch.zeros(1, model.d, device=DEVICE)
    z = torch.full((1, K), 1.0/K, device=DEVICE)
    prev_logits = torch.zeros(1, K, device=DEVICE)

    model.eval()
    for t in range(L, len(test_returns)):
        xw = torch.from_numpy(test_returns[t-L:t]).unsqueeze(0).to(DEVICE)
        tw = torch.from_numpy(test_tech[t-L:t]).unsqueeze(0).to(DEVICE)
        mw = torch.from_numpy(test_macro[t-L:t]).unsqueeze(0).to(DEVICE)

        xe = model.asset_encoder(xw)
        te = model.tech_encoder(tw)
        me = model.macro_encoder(mw)

        z_emb      = model.inference_net.z_embed(z)
        raw_logits = model.inference_net.net(torch.cat([xe, me, z_emb, te], dim=-1))
        smooth_logits = model.momentum * prev_logits + (1 - model.momentum) * raw_logits
        prev_logits   = smooth_logits

        soft  = F.softmax(smooth_logits, dim=-1)
        index = soft.argmax(dim=-1)
        z_new = F.one_hot(index, K).float()
        h     = model.film_ssm(xe, h, z_new)

        all_regimes.append(index.item())
        all_dates.append(test_dates[t])

        z = z_new
        h = h.detach()

    return pd.Series(all_regimes, index=pd.DatetimeIndex(all_dates))


def plot_regime_timeline(regime_series, test_returns_df):
    """单行彩色时间条 + 每个regime的平均收益率"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                             gridspec_kw={'height_ratios': [2, 8]})

    dates   = regime_series.index
    regimes = regime_series.values

    # ── 上图：单行彩色时间条（5日滚动主导regime平滑）──
    ax = axes[0]
    # 用5日滚动窗口取众数，减少视觉噪声
    import pandas as pd
    reg_series = pd.Series(regimes, index=dates)
    smoothed = reg_series.rolling(5, center=True, min_periods=1).apply(
        lambda x: pd.Series(x).mode()[0]
    ).astype(int)
    smooth_regimes = smoothed.values
    smooth_dates   = smoothed.index

    for i in range(len(smooth_dates) - 1):
        k     = smooth_regimes[i]
        start = smooth_dates[i]
        end   = smooth_dates[i+1]
        ax.axvspan(start, end, ymin=0, ymax=1,
                   color=REGIME_COLORS[k], alpha=1.0)

    # 图例
    patches = [mpatches.Patch(color=REGIME_COLORS[k], label=f'Regime {k}')
               for k in range(K)]
    ax.legend(handles=patches, loc='upper right', ncol=K, fontsize=10)
    ax.set_xlim(dates[0], dates[-1])
    ax.set_yticks([])
    ax.set_ylabel('Regime', fontsize=11)
    ax.set_title('MacroRegimeSSM: Learned Regime Sequence (Test Period 2020–2024)',
                 fontsize=13, fontweight='bold')

    # 分布统计放在y轴右侧（用平滑后的数据）
    u, c = np.unique(smooth_regimes, return_counts=True)
    for ui, ci in zip(u, c):
        pct = ci / len(smooth_regimes) * 100
        ax.text(1.01, (ui + 0.5) / K, f'R{ui}: {pct:.0f}%',
                transform=ax.transAxes, va='center', fontsize=9,
                color=REGIME_COLORS[ui], fontweight='bold')

    # ── 下图：每个regime下各资产平均日收益率 ──
    ax2 = axes[1]

    # 只取代表性资产：hs300(eq), t(bond), au(prec), rb(black), sc(energy), cu(metal)
    rep_assets = ['hs300', 't', 'au', 'rb', 'sc', 'cu']
    asset_labels = ['Equity\n(hs300)', 'Bond\n(t)', 'Prec\n(au)',
                    'Black\n(rb)', 'Energy\n(sc)', 'Metal\n(cu)']

    available = [a for a in rep_assets if a in test_returns_df.columns]
    labels    = [asset_labels[rep_assets.index(a)] for a in available]

    x    = np.arange(len(available))
    width = 0.2
    offsets = np.linspace(-(K-1)*width/2, (K-1)*width/2, K)

    for k in range(K):
        dates_k = regime_series[regime_series == k].index
        dates_k = dates_k[dates_k.isin(test_returns_df.index)]
        if len(dates_k) == 0:
            continue
        means = test_returns_df.loc[dates_k, available].mean() * 100  # 转为%
        ax2.bar(x + offsets[k], means, width,
                color=REGIME_COLORS[k], alpha=0.8, label=f'Regime {k}')

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Avg Daily Return (%)', fontsize=10)
    ax2.set_title('Average Daily Return by Regime and Asset Class', fontsize=11)
    ax2.legend(ncol=K, fontsize=9, loc='upper right')

    plt.tight_layout()
    out_path = FIG_DIR / 'regime_timeline.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def plot_regime_correlation(regime_series, test_returns_df):
    col_map = {
        'eq':     ['hs300', 'if', 'ic'],
        'bond':   ['t', 'tf'],
        'prec':   ['au', 'ag'],
        'black':  ['rb', 'hc', 'i'],
        'energy': ['sc', 'ma'],
        'metal':  ['cu', 'al', 'zn'],
    }
    asset_labels = {
        'eq': 'Equity', 'bond': 'Bond', 'prec': 'Precious',
        'black': 'Black', 'energy': 'Energy', 'metal': 'Metal'
    }

    plot_cols   = []
    plot_labels = []
    for cat, candidates in col_map.items():
        for c in candidates:
            if c in test_returns_df.columns:
                plot_cols.append(c)
                plot_labels.append(f"{asset_labels[cat]}\n({c})")
                break

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for k in range(K):
        dates_k = regime_series[regime_series == k].index
        dates_k = dates_k[dates_k.isin(test_returns_df.index)]

        if len(dates_k) < 10:
            axes[k].text(0.5, 0.5, f'Regime {k}\nInsufficient data',
                        ha='center', va='center', transform=axes[k].transAxes)
            continue

        corr = test_returns_df.loc[dates_k, plot_cols].corr()
        im   = axes[k].imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[k].set_xticks(range(len(plot_cols)))
        axes[k].set_yticks(range(len(plot_cols)))
        axes[k].set_xticklabels(plot_labels, fontsize=8, rotation=45, ha='right')
        axes[k].set_yticklabels(plot_labels, fontsize=8)
        axes[k].set_title(
            f'Regime {k}  (n={len(dates_k)} days)',
            fontsize=11, fontweight='bold', color=REGIME_COLORS[k]
        )
        for i in range(len(plot_cols)):
            for j in range(len(plot_cols)):
                val = corr.values[i, j]
                axes[k].text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7,
                            color='white' if abs(val) > 0.5 else 'black')
        plt.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    plt.suptitle('Asset Correlation Structure by Regime',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = FIG_DIR / 'regime_correlation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close()


def main():
    returns, tech, macro, N, F, _ = load_data()
    splits  = load_splits()
    idx     = returns.index
    val_end = idx.searchsorted(pd.to_datetime(splits['val_end']), side='right')

    test_returns = returns.iloc[val_end:].values.astype('float32')
    test_tech    = tech.iloc[val_end:].values.astype('float32').reshape(-1, N, 3)
    test_macro   = macro.iloc[val_end:].values.astype('float32')
    test_dates   = returns.index[val_end:]
    test_ret_df  = returns.iloc[val_end:]

    model = MacroRegimeSSM(
        n_assets=N, n_macro=F, L=20, n_regimes=K, momentum=0.8
    ).to(DEVICE)
    ckpt = torch.load(
        CKPT_DIR / 'seq_K4_d64_L20_T64' / 'best.pt', map_location=DEVICE
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    print("模型加载完成")

    print("推断regime序列...")
    regime_series = get_regime_sequence(
        model, test_returns, test_tech, test_macro, test_dates
    )

    u, c = np.unique(regime_series.values, return_counts=True)
    print("Regime分布:")
    for ui, ci in zip(u, c):
        print(f"  Regime {ui}: {ci}天 ({ci/len(regime_series)*100:.1f}%)")

    print("\n生成图表...")
    plot_regime_timeline(regime_series, test_ret_df)
    smoothed = regime_series.rolling(5, center=True, min_periods=1).apply(lambda x: pd.Series(x).mode()[0]).astype(int)
    plot_regime_correlation(smoothed, test_ret_df)
    print("✓ 完成，保存至 experiments/figures/")


if __name__ == '__main__':
    main()
