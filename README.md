# MacroRegimeSSM

**Macro-Conditioned Regime Switching State Space Models for Cross-Asset Return Prediction**

---

## Overview

MacroRegimeSSM jointly learns discrete latent macroeconomic regimes and regime-conditioned asset dynamics in a single end-to-end model. Key design choices:

- **Straight-through hard assignment** for true discrete regime semantics
- **Uniform prior + diversity regularisation** to prevent posterior collapse
- **Temporal consistency regularisation** for regime persistence
- **FiLM-conditioned GRU** backbone modulated per regime
- **ListNet ranking loss** directly optimising cross-asset return ranking

Evaluated on 56 Chinese futures and ETF instruments across 6 asset classes (2008–2024).

---

## Repository Structure

```
MacroRegimeSSM/
├── src/
│   ├── data/
│   │   ├── dataset.py          # DataLoader, train/val/test splits
│   │   ├── build_dataset.py    # raw → processed pipeline
│   │   └── fetch_macro.py      # akshare macro download + alignment
│   ├── models/
│   │   ├── macro_regime_ssm.py # full model
│   │   ├── asset_encoder.py    # 2-layer Transformer encoder
│   │   ├── tech_encoder.py     # per-asset GRU + attention pooling
│   │   ├── macro_encoder.py    # single-layer GRU
│   │   ├── film_ssm.py         # FiLM-conditioned GRUCell
│   │   ├── inference_net.py    # posterior q(z|x,m,c,z_prev)
│   │   ├── prediction_head.py  # regime-conditioned μ, σ head
│   │   └── baselines.py        # Linear, LSTM, Transformer, MacroLSTM
│   └── utils/
│       ├── trainer.py          # training loop, early stopping
│       ├── losses.py           # NLL, KL, smooth, rank, diversity
│       └── metrics.py          # IC, ICIR, Sharpe, MSE
├── train.py                    # train MacroRegimeSSM (any K)
├── train_baseline.py           # train all baselines
├── train_ablation.py           # train w/o Macro and w/o Regime variants
├── run_unified_eval.py         # canonical online single-step evaluation
├── evaluate.py                 # batch evaluation across all checkpoints
├── run_cost_all_models.py      # transaction cost analysis (daily)
├── run_cost_smooth.py          # transaction cost analysis (5d smoothed)
├── visualize_regime.py         # regime timeline + correlation figures
└── experiments/
    ├── checkpoints/            # saved model weights (best.pt + last.pt)
    └── figures/                # generated plots
```

---

## Installation

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib hmmlearn akshare
```

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8. Training uses a single NVIDIA A100 80 GB GPU; inference runs on CPU.

---

## Data

Place the following files under `data/processed/` before running any script:

| File | Description |
|---|---|
| `returns.parquet` | Daily returns, shape `(T, 56)`, columns = ticker symbols |
| `tech.parquet` | Technical indicators, shape `(T, 56×3)` — MACD, RSI, Bollinger %B |
| `macro.parquet` | Macroeconomic factors, shape `(T, 6)` — PMI, CPI, PPI, M1, M2, IP |
| `splits.json` | `{"train_end": "2019-12-31", "val_end": "2020-12-31"}` |

The dataset covers 56 Chinese futures and ETF instruments (2008-01-02 to 2024-03-18). Macroeconomic indicators are sourced via `akshare` and aligned to official publication dates to prevent look-ahead bias.

To rebuild macro data from scratch:
```bash
python src/data/fetch_macro.py        # downloads and aligns macro indicators
python src/data/build_dataset.py      # builds returns + tech parquets
```

---

## Results

Pre-trained checkpoints for all experiments are included in `experiments/checkpoints/`. Run the evaluation scripts directly to reproduce the tables below, or retrain from scratch using the training commands.

---

### Table 1 — Main Results (Test 2021–2024)

| Model | NLL ↓ | IC ↑ | ICIR ↑ | Sharpe ↑ | MSE ↓ |
|---|---:|---:|---:|---:|---:|
| HMM (K=4) | — | -0.013 | -0.046 | -0.009 | 0.000363 |
| Linear | -1.594 | -0.009 | -0.049 | -0.747 | 0.000976 |
| LSTM | -2.722 | **0.013** | 0.063 | -0.052 | **0.000263** |
| Transformer | -2.686 | 0.012 | **0.069** | 0.381 | **0.000263** |
| MacroLSTM | -2.274 | 0.006 | 0.031 | 0.052 | 0.000287 |
| **MacroRegimeSSM (K=4)** | **-2.819** | **0.013** | 0.053 | **0.848** | 0.000276 |

**Reproduce (evaluation only — checkpoints included):**
```bash
python run_unified_eval.py
```

**Retrain from scratch:**
```bash
# Main model (K=4)
python train.py --K 4 --T 64 --batch_size 32 --lambda2 0.3 --lambda4 2.0

# Baselines
python train_baseline.py --model all --T 128 --batch_size 16

# Then evaluate
python run_unified_eval.py
```

> `run_unified_eval.py` uses online stateful single-step inference (autoregressive, no future leakage). This is the canonical evaluation method that produced the reported numbers.

---

### Table 2 — Ablation Study

| Model | NLL ↓ | IC ↑ | ICIR ↑ | Sharpe ↑ | ΔSharpe |
|---|---:|---:|---:|---:|---:|
| Full Model (K=4) | **-2.819** | **0.013** | 0.053 | **0.848** | — |
| w/o Macro | -2.806 | 0.012 | **0.063** | 0.442 | -47.9% |
| w/o Regime | -2.646 | 0.007 | 0.035 | 0.352 | -58.5% |

**Reproduce (evaluation only):**
```bash
python evaluate.py
```

**Retrain ablation variants:**
```bash
python train_ablation.py --model no_macro  --T 64 --batch_size 32
python train_ablation.py --model no_regime --T 64 --batch_size 32

# Then evaluate
python evaluate.py
```

> `w/o Macro` removes macroeconomic inputs from the inference network while keeping the regime module. `w/o Regime` removes the FiLM regime module entirely, leaving a plain GRU with all three encoders.

---

### Table 3 — Sensitivity to Number of Regimes K

| K | NLL ↓ | IC ↑ | ICIR ↑ | Sharpe ↑ |
|---|---:|---:|---:|---:|
| K=3 | -2.790 | 0.011 | 0.056 | 0.565 |
| **K=4** | **-2.819** | **0.013** | 0.053 | **0.848** |
| K=5 | -2.817 | 0.012 | **0.060** | 0.327 |

**Reproduce (evaluation only):**
```bash
python evaluate.py
```

**Retrain for a specific K:**
```bash
python train.py --K 3 --T 64 --batch_size 32 --lambda2 0.3 --lambda4 2.0
python train.py --K 4 --T 64 --batch_size 32 --lambda2 0.3 --lambda4 2.0
python train.py --K 5 --T 64 --batch_size 32 --lambda2 0.3 --lambda4 2.0

# Then evaluate
python evaluate.py
```

> K=4 is selected on validation NLL. It also achieves the best test Sharpe, and its four-regime partition qualitatively aligns with the Merrill Lynch Investment Clock.

---

### Table 4 — Transaction Cost Analysis (Net Annualised Sharpe)

Turnover = average daily portfolio turnover. MacroRegimeSSM (5d) updates positions only on regime changes after 5-day majority-vote smoothing.

| Model | Turnover | 0 bps | 1 bps | 2 bps | 3 bps | 5 bps |
|---|---:|---:|---:|---:|---:|---:|
| HMM (K=4) | 1.875 | -0.009 | -0.478 | -0.946 | -1.413 | -2.336 |
| LSTM | 0.325 | -0.070 | -0.172 | -0.274 | -0.375 | -0.578 |
| MacroLSTM | 0.070 | -0.145 | -0.169 | -0.193 | -0.217 | -0.265 |
| MacroRegimeSSM (daily) | 0.766 | 0.592 | 0.401 | 0.210 | 0.019 | -0.364 |
| **MacroRegimeSSM (5d smoothed)** | **0.185** | **0.848** | **0.775** | **0.722** | **0.668** | **0.562** |

**Reproduce:**
```bash
# Daily rebalancing rows (HMM, LSTM, MacroLSTM, MacroRegimeSSM daily)
python run_cost_all_models.py

# 5-day smoothed row (MacroRegimeSSM 5d smoothed)
python run_cost_smooth.py
```

> The 5d smoothed strategy conditions position updates on regime changes detected by a 5-day rolling majority vote, reducing turnover from 76.6% to 18.5% while maintaining Sharpe > 0.56 at up to 5 bps.

---

### Figures — Regime Interpretability

**Figure 2** (`experiments/figures/regime_timeline.png`): Learned regime sequence (2021–2024) and per-regime average daily return by asset class.

**Figure 3** (`experiments/figures/regime_correlation.png`): Cross-asset correlation matrices under each of the four regimes.

```bash
python visualize_regime.py
```

Output is saved to `experiments/figures/`.

---

## Learned Regimes

The four regimes identified on the 2021–2024 test period:

| Regime | Frequency | Interpretation | Key Characteristic |
|---|---:|---|---|
| 0 | 5.0% | Industrial Commodity Boom | Broad positive returns; ferrous, energy, non-ferrous co-move strongly |
| 1 | 76.1% | Sideways Base State | Near-zero returns; dominant low-volatility regime |
| 2 | 6.6% | Industrial Contraction | Equities and ferrous negative; precious metals positive |
| 3 | 12.4% | Risk-Off Defensive | Equity/energy negative; strongest bond–equity hedge (corr = −0.29) |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| K | 4 | Number of discrete regimes |
| d_model | 64 | Hidden dimension |
| L | 20 | History window length (days) |
| T | 64 | Unroll length during training |
| batch_size | 32 | Training batch size |
| lr | 1e-3 | AdamW learning rate |
| weight_decay | 1e-4 | AdamW weight decay |
| λ₂ (smooth) | 0.3 | Temporal consistency loss weight |
| λ₃ (rank) | 5.0 | ListNet ranking loss weight |
| λ₄ (div) | 2.0 | Diversity loss weight |
| patience | 20 | Early stopping patience (epochs) |
| momentum β | 0.8 | Logit momentum smoothing at inference |
| diversity cap α | 0.4 | Max empirical frequency per regime |

---

## Citation

Citation will be added upon publication.
