"""
film_ssm.py
FiLM调制的GRU状态空间模型
regime通过FiLM（Feature-wise Linear Modulation）调制资产动态

输入：x_emb (B, d), h_prev (B, d), z_t (B, K)
输出：h_t   (B, d)
"""

import torch
import torch.nn as nn


class FiLMSSM(nn.Module):
    def __init__(self, d_model: int = 64, n_regimes: int = 4, dropout: float = 0.1):
        super().__init__()

        # GRU状态转移
        self.gru_cell = nn.GRUCell(
            input_size=d_model,
            hidden_size=d_model,
        )

        # FiLM参数生成：z_t → (γ, β)，各d维
        self.film_layer = nn.Sequential(
            nn.Linear(n_regimes, d_model * 2),
            nn.Tanh(),               # 限制scale/shift的范围，防止梯度爆炸
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_emb: torch.Tensor,    # (B, d)
        h_prev: torch.Tensor,   # (B, d)
        z_t: torch.Tensor,      # (B, K)
    ) -> torch.Tensor:
        """
        返回：h_t (B, d)
        """
        # GRU更新（不含FiLM）
        h_raw = self.gru_cell(x_emb, h_prev)       # (B, d)

        # FiLM调制参数
        film_params = self.film_layer(z_t)          # (B, 2d)
        gamma, beta = film_params.chunk(2, dim=-1)  # 各(B, d)

        # 调制：h_t = γ ⊙ h_raw + β
        h_t = gamma * h_raw + beta                  # (B, d)
        h_t = self.norm(h_t)
        h_t = self.dropout(h_t)

        return h_t
