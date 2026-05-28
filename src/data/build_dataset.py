"""
build_dataset.py
从标的收益率计算技术因子，合并宏观数据，生成完整训练数据

输出：
  data/processed/returns.parquet      (T, N) 标的收益率
  data/processed/technicals.parquet  (T, N*3) 技术因子
  data/processed/macro.parquet       (T, F) 宏观因子
  data/processed/splits.csv          切分索引
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2008-01-01"
END_DATE    = "2024-03-18"


# ─────────────────────────────────────────────
# 1. 技术因子计算
# ─────────────────────────────────────────────

def calc_macd(price: pd.Series, fast=12, slow=26, signal=9):
    """返回 (macd_line, signal_line, histogram) 三列"""
    ema_fast = price.ewm(span=fast, adjust=False).mean()
    ema_slow = price.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    hist     = macd - sig
    return macd, sig, hist


def calc_rsi(price: pd.Series, window=14):
    delta = price.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window-1, adjust=False).mean()
    avg_loss = loss.ewm(com=window-1, adjust=False).mean()
    rs  = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_boll(price: pd.Series, window=20, n_std=2):
    """返回 (上轨-中轨)/中轨, (中轨-下轨)/中轨, (price-中轨)/中轨 三个归一化指标"""
    mid   = price.rolling(window).mean()
    std   = price.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # 归一化，消除价格水平差异
    width     = (upper - lower) / (mid.abs() + 1e-8)   # 带宽
    pct_b     = (price - lower) / (upper - lower + 1e-8)  # %B位置
    deviation = (price - mid) / (mid.abs() + 1e-8)     # 偏离中轨
    return width, pct_b, deviation


def build_technicals(returns: pd.DataFrame) -> pd.DataFrame:
    """
    对每个标的计算技术因子
    输入：returns (T, N) 收益率
    输出：tech_df (T, N*3) 技术因子，列名格式 {col}_macd/{col}_rsi/{col}_boll
    
    注意：技术因子基于累积价格（净值曲线）计算，而非直接用收益率
    """
    print("计算技术因子...")
    tech_frames = []

    for col in returns.columns:
        r = returns[col].fillna(0)
        # 构造价格序列（从1开始的净值曲线）
        price = (1 + r).cumprod()

        # MACD：用histogram作为主信号（momentum方向）
        _, _, hist = calc_macd(price)
        # 归一化：除以价格水平
        macd_norm = hist / (price.abs() + 1e-8)

        # RSI：归一化到[-1, 1]
        rsi = calc_rsi(price)
        rsi_norm = (rsi - 50) / 50

        # 布林带：用%B位置（已在0-1范围内）
        _, pct_b, _ = calc_boll(price)
        boll_norm = pct_b * 2 - 1  # 映射到[-1, 1]

        tech_frames.append(pd.DataFrame({
            f'{col}_macd': macd_norm,
            f'{col}_rsi':  rsi_norm,
            f'{col}_boll': boll_norm,
        }))

    tech_df = pd.concat(tech_frames, axis=1)
    print(f"  技术因子: {tech_df.shape}")
    return tech_df


# ─────────────────────────────────────────────
# 2. 加载和清洗标的收益率
# ─────────────────────────────────────────────

def load_returns() -> pd.DataFrame:
    """加载标的收益率，清洗并截止到END_DATE"""
    df = pd.read_excel(
        RAW_DIR / "asset_returns.xlsx",
        sheet_name='标的收益率'
    )
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 截止到END_DATE
    df = df.loc[:END_DATE]

    print(f"原始标的收益率: {df.shape}")
    print(f"时间范围: {df.index.min().date()} ~ {df.index.max().date()}")

    # 缺失值处理：forward fill后再backward fill（期货节假日停牌）
    # 但不超过5天连续填充，避免填充太多
    df = df.ffill(limit=5).bfill(limit=5)

    # 截取训练起点后的数据（技术因子需要warm-up，所以保留2005年数据用于计算）
    remaining_nan = df.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"  剩余NaN {remaining_nan} 个，用0填充")
        df = df.fillna(0)

    print(f"清洗后: {df.shape}, 缺失值: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# 3. 加载宏观数据
# ─────────────────────────────────────────────

def load_macro(asset_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """加载已生成的宏观数据，对齐到资产交易日"""
    macro_path = RAW_DIR / "macro_raw.parquet"
    if not macro_path.exists():
        raise FileNotFoundError(
            f"找不到宏观数据：{macro_path}\n"
            "请先运行 src/data/fetch_macro.py"
        )
    macro = pd.read_parquet(macro_path)
    macro.index = pd.to_datetime(macro.index)

    # union索引后ffill，再筛选资产交易日
    all_dates = asset_dates.union(macro.index).sort_values()
    macro_expanded = macro.reindex(all_dates).ffill()
    macro_aligned  = macro_expanded.reindex(asset_dates)

    # 早期缺失用0填充
    macro_aligned = macro_aligned.fillna(0)
    print(f"宏观因子: {macro_aligned.shape}")
    return macro_aligned


# ─────────────────────────────────────────────
# 4. 标准化
# ─────────────────────────────────────────────

def normalize(df: pd.DataFrame, n_train: int, name: str) -> pd.DataFrame:
    """用训练集统计量做z-score标准化，clip到[-5, 5]"""
    train_mean = df.iloc[:n_train].mean()
    train_std  = df.iloc[:n_train].std().replace(0, 1)
    normalized = (df - train_mean) / train_std
    normalized = normalized.clip(-5, 5)
    print(f"  {name}: mean~0 ✓, std~1 ✓, clip[-5,5] ✓")

    # 保存统计量
    stats = pd.DataFrame({'mean': train_mean, 'std': train_std})
    stats.to_csv(PROCESSED_DIR / f"{name}_stats.csv")
    return normalized


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("MacroRegimeSSM 数据构建（56标的 + 技术因子版）")
    print("=" * 55)

    # Step 1: 加载收益率（含2005-2007用于技术因子warm-up）
    returns_full = load_returns()

    # Step 2: 计算技术因子（在完整数据上计算，避免边界效应）
    tech_full = build_technicals(returns_full)

    # Step 3: 裁剪到训练起点
    returns = returns_full.loc[TRAIN_START:]
    tech    = tech_full.loc[TRAIN_START:]

    # Step 4: 加载宏观数据
    macro = load_macro(returns.index)
    macro = macro.loc[TRAIN_START:]

    # Step 5: 确保时间对齐
    common_idx = returns.index.intersection(tech.index).intersection(macro.index)
    returns = returns.loc[common_idx]
    tech    = tech.loc[common_idx]
    macro   = macro.loc[common_idx]

    print(f"\n对齐后样本数: {len(common_idx)}")
    print(f"时间范围: {common_idx.min().date()} ~ {common_idx.max().date()}")
    print(f"资产数N: {returns.shape[1]}")
    print(f"技术因子数: {tech.shape[1]}（= N×3）")
    print(f"宏观因子数: {macro.shape[1]}")

    # Step 6: 切分索引（70/10/20）
    T = len(common_idx)
    n_train = int(T * 0.7)
    n_val   = int(T * 0.1)
    n_test  = T - n_train - n_val

    splits = {
        'train_end': common_idx[n_train - 1].strftime('%Y-%m-%d'),
        'val_end':   common_idx[n_train + n_val - 1].strftime('%Y-%m-%d'),
        'test_end':  common_idx[-1].strftime('%Y-%m-%d'),
        'n_train':   n_train,
        'n_val':     n_val,
        'n_test':    n_test,
    }
    print(f"\nTrain: {n_train}天 → {splits['train_end']}")
    print(f"Val:   {n_val}天 → {splits['val_end']}")
    print(f"Test:  {n_test}天 → {splits['test_end']}")

    # Step 7: 标准化（技术因子和宏观因子，收益率不标准化）
    print("\n标准化：")
    tech_norm  = normalize(tech,  n_train, "technicals")
    macro_norm = normalize(macro, n_train, "macro")

    # Step 8: 保存
    returns.to_parquet(PROCESSED_DIR / "returns.parquet")
    tech_norm.to_parquet(PROCESSED_DIR / "technicals.parquet")
    macro_norm.to_parquet(PROCESSED_DIR / "macro.parquet")
    pd.Series(splits).to_csv(PROCESSED_DIR / "splits.csv", header=False)

    print(f"\n✓ 保存完成:")
    print(f"  returns.parquet:     {returns.shape}")
    print(f"  technicals.parquet:  {tech_norm.shape}")
    print(f"  macro.parquet:       {macro_norm.shape}")
    print(f"  splits.csv:          {splits}")


if __name__ == "__main__":
    main()
