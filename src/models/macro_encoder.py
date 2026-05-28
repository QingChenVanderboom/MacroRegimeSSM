"""
macro_encoder.py
用GRU对宏观因子序列编码
输入：m_seq (B, L, F)
输出：m_emb  (B, d)
"""

import torch
import torch.nn as nn


class MacroEncoder(nn.Module):
    def __init__(self, n_macro: int = 6, d_model: int = 64, dropout: float = 0.1):
        super().__init__()

        # 输入clip防止极端值（宏观因子标准化后偶有-9.8这类极端值）
        self.clip_val = 5.0

        # 输入投影
        self.input_proj = nn.Linear(n_macro, d_model)

        # 单层GRU
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, m_seq: torch.Tensor) -> torch.Tensor:
        """
        m_seq: (B, L, F)
        return: (B, d)
        """
        # clip极端值
        m = m_seq.clamp(-self.clip_val, self.clip_val)

        # 投影
        m = self.input_proj(m)              # (B, L, d)
        m = self.dropout(m)

        # GRU，取最后时间步隐状态
        _, h_n = self.gru(m)               # h_n: (1, B, d)
        m_emb = self.norm(h_n.squeeze(0))  # (B, d)
        return m_emb
