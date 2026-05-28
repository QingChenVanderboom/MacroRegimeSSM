"""
dataset.py - 序列化训练版本（修复val/test stride）
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def load_data():
    returns = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    tech    = pd.read_parquet(PROCESSED_DIR / "technicals.parquet")
    macro   = pd.read_parquet(PROCESSED_DIR / "macro.parquet")

    common = returns.index.intersection(tech.index).intersection(macro.index)
    returns = returns.loc[common]
    tech    = tech.loc[common]
    macro   = macro.loc[common]

    N = returns.shape[1]
    F = macro.shape[1]
    asset_cols = returns.columns.tolist()
    return returns, tech, macro, N, F, asset_cols


def load_splits():
    return pd.read_csv(
        PROCESSED_DIR / "splits.csv", header=None, index_col=0
    ).squeeze()


class SequentialDataset(Dataset):
    def __init__(
        self,
        returns:   pd.DataFrame,
        tech:      pd.DataFrame,
        macro:     pd.DataFrame,
        start_idx: int,
        end_idx:   int,
        L:      int = 20,
        T:      int = 128,
        stride: int = 1,
    ):
        super().__init__()
        self.L = L
        self.T = T
        self.window = L + T

        subset_ret  = returns.iloc[start_idx:end_idx].values.astype(np.float32)
        N = subset_ret.shape[1]
        subset_tech = tech.iloc[start_idx:end_idx].values.astype(np.float32)
        subset_mac  = macro.iloc[start_idx:end_idx].values.astype(np.float32)

        self.x_data    = subset_ret
        self.tech_data = subset_tech.reshape(-1, N, 3)
        self.m_data    = subset_mac
        self.N         = N

        total = len(subset_ret)
        self.starts = list(range(0, total - self.window + 1, stride))
        assert len(self.starts) > 0, \
            f"数据太短({total}行)，window={self.window}, stride={stride}"

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        e = s + self.window

        x_full    = self.x_data[s:e]
        tech_full = self.tech_data[s:e]
        m_full    = self.m_data[s:e]
        x_target  = self.x_data[s+self.L:e]

        return {
            'x_seq':    torch.from_numpy(x_full),
            'tech_seq': torch.from_numpy(tech_full),
            'm_seq':    torch.from_numpy(m_full),
            'x_target': torch.from_numpy(x_target),
        }


def build_dataloaders(
    L:          int  = 20,
    T:          int  = 128,
    batch_size: int  = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_stride: int = 1,
    eval_stride:  int = 20,   # val/test用较小stride保证样本数够
):
    returns, tech, macro, N, F, asset_cols = load_data()
    splits = load_splits()

    idx = returns.index
    train_end = idx.searchsorted(pd.to_datetime(splits['train_end']), side='right')
    val_end   = idx.searchsorted(pd.to_datetime(splits['val_end']),   side='right')

    split_ranges = {
        'train': (0,         train_end),
        'val':   (train_end, val_end),
        'test':  (val_end,   len(idx)),
    }

    print(f"数据: returns{returns.shape}, tech{tech.shape}, macro{macro.shape}")
    for name, (s, e) in split_ranges.items():
        print(f"  {name}: {e-s}行, {idx[s].date()} ~ {idx[e-1].date()}")

    def make_loader(split, shuffle):
        s, e = split_ranges[split]
        stride = train_stride if split == 'train' else eval_stride
        ds = SequentialDataset(
            returns, tech, macro, s, e,
            L=L, T=T, stride=stride
        )
        print(f"  {split} samples: {len(ds)}")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=(split == 'train'),
        )

    train_loader = make_loader('train', shuffle=True)
    val_loader   = make_loader('val',   shuffle=False)
    test_loader  = make_loader('test',  shuffle=False)

    return train_loader, val_loader, test_loader, N, F, asset_cols
