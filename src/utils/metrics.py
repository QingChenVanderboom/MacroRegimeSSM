"""
metrics.py
金融评估指标：IC、ICIR、Sharpe
"""

import numpy as np
from scipy.stats import spearmanr


def calc_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """
    单截面IC：预测收益率与实际收益率的Spearman相关
    pred, actual: (N,) 单个时间步的预测和实际
    """
    if pred.std() < 1e-8 or actual.std() < 1e-8:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    return float(ic) if not np.isnan(ic) else 0.0


def calc_ic_series(
    preds: np.ndarray,      # (T, N)
    actuals: np.ndarray,    # (T, N)
) -> np.ndarray:
    """计算每个时间步的IC，返回IC序列 (T,)"""
    T = preds.shape[0]
    ics = np.array([calc_ic(preds[t], actuals[t]) for t in range(T)])
    return ics


def calc_icir(ic_series: np.ndarray) -> float:
    """ICIR = mean(IC) / std(IC)"""
    if ic_series.std() < 1e-8:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def calc_sharpe(
    preds: np.ndarray,      # (T, N)
    actuals: np.ndarray,    # (T, N)
    annualize: bool = True,
) -> float:
    """
    基于预测的简单多空组合Sharpe：
    做多预测收益率最高的一半资产，做空另一半
    """
    T = preds.shape[0]
    pnl = []
    for t in range(T):
        rank = preds[t].argsort()
        n = len(rank)
        mid = n // 2
        long_idx  = rank[mid:]
        short_idx = rank[:mid]
        daily_pnl = actuals[t][long_idx].mean() - actuals[t][short_idx].mean()
        pnl.append(daily_pnl)

    pnl = np.array(pnl)
    if pnl.std() < 1e-8:
        return 0.0

    sharpe = pnl.mean() / pnl.std()
    if annualize:
        sharpe *= np.sqrt(252)
    return float(sharpe)


def evaluate(
    preds: np.ndarray,
    actuals: np.ndarray,
) -> dict:
    """一次性计算所有指标"""
    ic_series = calc_ic_series(preds, actuals)
    return {
        'IC_mean':  float(ic_series.mean()),
        'IC_std':   float(ic_series.std()),
        'ICIR':     calc_icir(ic_series),
        'Sharpe':   calc_sharpe(preds, actuals),
        'MSE':      float(((preds - actuals) ** 2).mean()),
        'MAE':      float(np.abs(preds - actuals).mean()),
    }
