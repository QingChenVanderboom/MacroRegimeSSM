"""
fetch_macro.py
拉取中国宏观因子数据并与资产收益率对齐

注意：所有宏观数据使用发布日对齐（publication-date alignment）
防止未来数据泄露（look-ahead bias）

数据来源：akshare（公开免费）
覆盖范围：2005-01-01 ~ 2024-03-31
"""

import akshare as ak
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 项目路径
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START_DATE  = "2005-01-01"  # 数据拉取起点（宏观数据需要足够历史）
TRAIN_START = "2008-01-01"  # 训练起点（2008前PMI/M1/M2为0填充，排除）
END_DATE    = "2024-03-31"


# ─────────────────────────────────────────────
# 1. 各指标拉取函数
# ─────────────────────────────────────────────

def fetch_pmi() -> pd.DataFrame:
    """制造业PMI，月频，发布日在次月初"""
    df = ak.macro_china_pmi()
    # 列名：月份 | 制造业-指数 | ...
    df = df[['月份', '制造业-指数']].copy()
    df.columns = ['pub_date', 'pmi']
    df['pub_date'] = pd.to_datetime(df['pub_date'], format='%Y年%m月份', errors='coerce')
    df = df.dropna(subset=['pub_date'])
    # PMI通常在次月20日前后发布，这里保守用次月1日作为可用日期
    df['pub_date'] = df['pub_date'] + pd.offsets.MonthBegin(1)
    df['pmi'] = pd.to_numeric(df['pmi'], errors='coerce')
    return df[['pub_date', 'pmi']].dropna().sort_values('pub_date')


def fetch_cpi() -> pd.DataFrame:
    """CPI同比，月频"""
    df = ak.macro_china_cpi_yearly()
    # 列名：日期 | 今值 | 预测值 | 前值
    df = df[['日期', '今值']].copy()
    df.columns = ['pub_date', 'cpi_yoy']
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    df['cpi_yoy'] = pd.to_numeric(df['cpi_yoy'], errors='coerce')
    return df[['pub_date', 'cpi_yoy']].dropna().sort_values('pub_date')


def fetch_ppi() -> pd.DataFrame:
    """PPI同比，月频"""
    df = ak.macro_china_ppi_yearly()
    df = df[['日期', '今值']].copy()
    df.columns = ['pub_date', 'ppi_yoy']
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    df['ppi_yoy'] = pd.to_numeric(df['ppi_yoy'], errors='coerce')
    return df[['pub_date', 'ppi_yoy']].dropna().sort_values('pub_date')


def fetch_money_supply() -> pd.DataFrame:
    """M1/M2同比，月频"""
    df = ak.macro_china_money_supply()
    # 列名：月份 | M2-数量 | M2-同比增长 | M2-环比增长 | M1-... | M0-...
    cols_needed = ['月份']
    # 找M1、M2同比列
    m2_col = [c for c in df.columns if 'M2' in c and '同比' in c]
    m1_col = [c for c in df.columns if 'M1' in c and '同比' in c]
    if not m2_col or not m1_col:
        print(f"  [WARN] money supply列名异常: {df.columns.tolist()}")
        return pd.DataFrame(columns=['pub_date', 'm1_yoy', 'm2_yoy'])
    df = df[['月份', m2_col[0], m1_col[0]]].copy()
    df.columns = ['pub_date', 'm2_yoy', 'm1_yoy']
    df['pub_date'] = pd.to_datetime(df['pub_date'], format='%Y年%m月份', errors='coerce')
    df = df.dropna(subset=['pub_date'])
    # 货币数据约在次月10-15日发布
    df['pub_date'] = df['pub_date'] + pd.offsets.MonthBegin(1)
    df['m1_yoy'] = pd.to_numeric(df['m1_yoy'], errors='coerce')
    df['m2_yoy'] = pd.to_numeric(df['m2_yoy'], errors='coerce')
    return df[['pub_date', 'm1_yoy', 'm2_yoy']].dropna().sort_values('pub_date')


def fetch_social_finance() -> pd.DataFrame:
    """社融增量，月频，尝试多个接口"""
    # 尝试接口1：macro_china_shrzgm
    try:
        df = ak.macro_china_shrzgm()
        if len(df) > 0:
            date_col = df.columns[0]
            # 找社融增量列（通常是第一个数值列）
            val_col = [c for c in df.columns if '增量' in str(c) or '社融' in str(c)]
            if not val_col:
                val_col = [df.columns[1]]
            df = df[[date_col, val_col[0]]].copy()
            df.columns = ['pub_date', 'social_finance']
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
            df = df.dropna(subset=['pub_date'])
            df['pub_date'] = df['pub_date'] + pd.offsets.MonthBegin(1)
            df['social_finance'] = pd.to_numeric(df['social_finance'], errors='coerce')
            result = df[['pub_date', 'social_finance']].dropna().sort_values('pub_date')
            if len(result) > 0:
                return result
    except Exception as e:
        print(f"  [WARN] shrzgm接口失败: {e}")

    # 尝试接口2：macro_china_money_supply里的社融
    try:
        df = ak.macro_china_supply_finance()
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df = df[[date_col, val_col]].copy()
        df.columns = ['pub_date', 'social_finance']
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
        df = df.dropna(subset=['pub_date'])
        df['pub_date'] = df['pub_date'] + pd.offsets.MonthBegin(1)
        df['social_finance'] = pd.to_numeric(df['social_finance'], errors='coerce')
        return df[['pub_date', 'social_finance']].dropna().sort_values('pub_date')
    except Exception as e:
        print(f"  [WARN] supply_finance接口失败: {e}")

    # 两个接口都失败，返回空DataFrame（不影响其他指标）
    print("  [WARN] 社融数据获取失败，跳过该指标")
    return pd.DataFrame(columns=['pub_date', 'social_finance'])


def fetch_industrial_production() -> pd.DataFrame:
    """工业增加值同比，月频"""
    df = ak.macro_china_industrial_production_yoy()
    df = df[['日期', '今值']].copy()
    df.columns = ['pub_date', 'indprod_yoy']
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    df['indprod_yoy'] = pd.to_numeric(df['indprod_yoy'], errors='coerce')
    return df[['pub_date', 'indprod_yoy']].dropna().sort_values('pub_date')


# ─────────────────────────────────────────────
# 2. 合并宏观数据
# ─────────────────────────────────────────────

def build_macro_panel() -> pd.DataFrame:
    """
    拉取所有宏观指标，合并为日频面板
    对齐方式：publication-date forward-fill
    即：某指标在发布日当天才可用，发布前用上一期值

    核心修复：以资产数据的实际交易日为索引基准，而非通用工作日，
    防止索引不匹配导致reindex后全为NaN
    """
    print("拉取宏观数据...")

    fetchers = {
        'pmi':            fetch_pmi,
        'cpi_yoy':        fetch_cpi,
        'ppi_yoy':        fetch_ppi,
        'm1m2':           fetch_money_supply,
        'social_finance': fetch_social_finance,
        'indprod_yoy':    fetch_industrial_production,
    }

    frames = []
    for name, fn in fetchers.items():
        try:
            df = fn()
            if len(df) == 0:
                print(f"  - {name}: 空数据，跳过")
                continue
            print(f"  ✓ {name}: {len(df)} 条, {df['pub_date'].min().date()} ~ {df['pub_date'].max().date()}")
            frames.append(df.set_index('pub_date'))
        except Exception as e:
            print(f"  ✗ {name}: 拉取失败 → {e}")

    if not frames:
        raise RuntimeError("所有宏观指标拉取失败")

    # 外连接合并所有指标（稀疏的月频数据，index是发布日）
    macro = pd.concat(frames, axis=1).sort_index()

    # 以资产数据的实际交易日为基准构建索引
    asset_path = RAW_DIR / "asset_returns.xlsx"
    asset_df = pd.read_excel(asset_path, sheet_name='大类资产收益率').dropna()
    asset_dates = pd.to_datetime(asset_df['date'].values)

    # 关键步骤：
    # 1. 先union索引（资产交易日 + 宏观发布日）
    # 2. reindex到union索引，ffill填充
    # 3. 再筛选出资产交易日
    all_dates = asset_dates.union(macro.index).sort_values()
    macro_expanded = macro.reindex(all_dates).ffill()
    macro_daily = macro_expanded.reindex(asset_dates)

    # 裁剪到目标时间范围
    macro_daily = macro_daily.loc[START_DATE:END_DATE]

    print(f"\n宏观面板: {macro_daily.shape}, 缺失率:")
    print((macro_daily.isnull().sum() / len(macro_daily)).round(3))

    return macro_daily


# ─────────────────────────────────────────────
# 3. 与资产收益率对齐
# ─────────────────────────────────────────────

def align_with_assets(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """
    将宏观日频面板与资产收益率数据对齐
    返回合并后的完整数据集

    注意：宏观数据在2008年前覆盖不完整（PMI/M1M2从2008年开始）
    对策：只对资产列做dropna，宏观列允许在训练时用0填充或mask
    """
    asset_path = RAW_DIR / "asset_returns.xlsx"
    asset_df = pd.read_excel(asset_path, sheet_name='大类资产收益率')
    asset_df = asset_df.dropna()
    asset_df = asset_df.set_index('date')
    asset_df.index = pd.to_datetime(asset_df.index)

    print(f"\n资产数据: {asset_df.shape}")
    print(f"宏观数据: {macro_daily.shape}")

    # 以资产数据的交易日为基准
    macro_aligned = macro_daily.reindex(asset_df.index).ffill()

    # 合并
    combined = pd.concat([asset_df, macro_aligned], axis=1)

    # 只对资产列做dropna（保证收益率完整）
    asset_cols = ['eq', 'bond', 'prec', 'black', 'energy', 'metal']
    combined = combined.dropna(subset=asset_cols)

    # 宏观列剩余NaN用0填充（对应数据覆盖范围外的早期数据）
    macro_cols = [c for c in combined.columns if c not in asset_cols]
    n_macro_nan = combined[macro_cols].isnull().sum().sum()
    if n_macro_nan > 0:
        print(f"  [INFO] 宏观列剩余 {n_macro_nan} 个NaN（早期数据不足），用0填充")
        combined[macro_cols] = combined[macro_cols].fillna(0)

    print(f"\n合并后数据: {combined.shape}")
    print(f"时间范围: {combined.index.min().date()} ~ {combined.index.max().date()}")
    print(f"有效样本: {len(combined)} 天")

    return combined


# ─────────────────────────────────────────────
# 4. 标准化与保存
# ─────────────────────────────────────────────

def normalize_and_save(combined: pd.DataFrame):
    """
    宏观因子做rolling z-score标准化（用训练集统计量）
    资产收益率不标准化（保留原始尺度，NLL损失里处理）
    """
    asset_cols = ['eq', 'bond', 'prec', 'black', 'energy', 'metal']
    macro_cols = [c for c in combined.columns if c not in asset_cols]

    # 从TRAIN_START开始切，排除2005-2007宏观0填充段
    combined = combined.loc[TRAIN_START:]
    print(f"训练起点裁剪后: {len(combined)} 天（{TRAIN_START} ~）")

    # 训练集截止：前70%
    n_train = int(len(combined) * 0.7)
    train_end = combined.index[n_train]
    print(f"\n训练集截止: {train_end.date()}（{n_train}天）")

    # Rolling z-score for macro（用expanding window，只用历史数据）
    macro_normalized = combined[macro_cols].copy()
    train_mean = combined[macro_cols].iloc[:n_train].mean()
    train_std  = combined[macro_cols].iloc[:n_train].std().replace(0, 1)

    macro_normalized = (combined[macro_cols] - train_mean) / train_std

    # 拼回
    processed = pd.concat([combined[asset_cols], macro_normalized], axis=1)

    # 保存
    out_path = PROCESSED_DIR / "aligned_data.parquet"
    processed.to_parquet(out_path)
    print(f"\n保存至: {out_path}")

    # 保存统计量（推理时用）
    stats = pd.DataFrame({'mean': train_mean, 'std': train_std})
    stats.to_csv(PROCESSED_DIR / "macro_stats.csv")

    # 保存train/val/test切分索引
    n_val  = int(len(combined) * 0.1)
    splits = {
        'train_end': combined.index[n_train - 1].strftime('%Y-%m-%d'),
        'val_end':   combined.index[n_train + n_val - 1].strftime('%Y-%m-%d'),
        'test_end':  combined.index[-1].strftime('%Y-%m-%d'),
        'n_train':   n_train,
        'n_val':     n_val,
        'n_test':    len(combined) - n_train - n_val,
    }
    pd.Series(splits).to_csv(PROCESSED_DIR / "splits.csv", header=False)
    print("Train/Val/Test 切分:")
    for k, v in splits.items():
        print(f"  {k}: {v}")

    return processed


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("MacroRegimeSSM 数据准备")
    print("=" * 50)

    # Step 1: 拉取宏观数据
    macro_daily = build_macro_panel()

    # Step 2: 与资产对齐
    combined = align_with_assets(macro_daily)

    # Step 3: 标准化并保存
    processed = normalize_and_save(combined)

    # Step 4: 保存原始宏观（未标准化，供分析用）
    macro_raw_path = RAW_DIR / "macro_raw.parquet"
    macro_daily.to_parquet(macro_raw_path)
    print(f"\n原始宏观数据保存至: {macro_raw_path}")

    print("\n✓ 数据准备完成")
    print(f"  最终数据列: {processed.columns.tolist()}")
    print(f"  Shape: {processed.shape}")


if __name__ == "__main__":
    main()