"""
tech_encoder.py
技术因子编码器
输入：tech_seq (B, L, N, 3) 每个标的L天的3个技术指标
输出：tech_emb (B, d)

策略：先在标的维度做参数共享的GRU，再在N维度做pooling
"""

import torch
import torch.nn as nn


class TechEncoder(nn.Module):
    def __init__(self, n_tech: int = 3, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        # 输入投影：3 → d（per-asset，参数共享）
        self.input_proj = nn.Linear(n_tech, d_model)

        # GRU在时间维度展开（参数在所有标的间共享）
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # 跨标的注意力pooling（比简单mean pooling更强）
        self.attn = nn.Linear(d_model, 1)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tech_seq: torch.Tensor) -> torch.Tensor:
        """
        tech_seq: (B, L, N, 3)
        return:   (B, d)
        """
        B, L, N, _ = tech_seq.shape

        # reshape成(B*N, L, 3)，让GRU在时间维度展开，参数在标的间共享
        x = tech_seq.permute(0, 2, 1, 3)           # (B, N, L, 3)
        x = x.reshape(B * N, L, 3)                 # (B*N, L, 3)

        x = self.input_proj(x)                     # (B*N, L, d)
        x = self.dropout(x)

        _, h_n = self.gru(x)                       # h_n: (1, B*N, d)
        h = h_n.squeeze(0)                         # (B*N, d)
        h = h.reshape(B, N, -1)                    # (B, N, d)

        # 注意力pooling：学习哪些标的的技术信号更重要
        attn_w = torch.softmax(self.attn(h), dim=1)  # (B, N, 1)
        tech_emb = (h * attn_w).sum(dim=1)           # (B, d)
        tech_emb = self.norm(tech_emb)

        return tech_emb
