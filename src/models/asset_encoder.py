"""
asset_encoder.py
用Transformer对历史资产收益率序列编码
输入：x_seq (B, L, N)
输出：x_emb  (B, d)
"""

import torch
import torch.nn as nn
import math


class AssetEncoder(nn.Module):
    def __init__(self, n_assets: int = 6, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 输入投影：N → d_model
        self.input_proj = nn.Linear(n_assets, d_model)

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # (B, L, d)
            norm_first=True,    # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B, L, N)
        return: (B, d)
        """
        # 投影到d_model维度
        x = self.input_proj(x_seq)          # (B, L, d)
        x = self.pos_enc(x)                 # (B, L, d)

        # Transformer编码
        x = self.transformer(x)             # (B, L, d)

        # 取最后时间步作为序列表示
        x_emb = self.norm(x[:, -1, :])     # (B, d)
        return x_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
