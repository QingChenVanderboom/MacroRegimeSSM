"""
prediction_head.py
预测头：输出下期收益率的均值和对数标准差
输入：h_t (B, d), z_t (B, K)
输出：mu (B, N), log_sigma (B, N)
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, d_model: int = 64, n_regimes: int = 4,
                 n_assets: int = 6, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model + n_regimes, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_assets * 2),   # 输出mu和log_sigma
        )

        # log_sigma的clamp范围，防止方差过大或过小
        self.log_sigma_min = -6.0
        self.log_sigma_max = 2.0

        self._init_weights()

    def _init_weights(self):
        # 最后一层小初始化，让初始预测接近0（符合金融收益率先验）
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_t: torch.Tensor,      # (B, d)
        z_t: torch.Tensor,      # (B, K)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
          mu:        (B, N) 预测收益率均值
          log_sigma: (B, N) 预测收益率对数标准差
        """
        inp = torch.cat([h_t, z_t], dim=-1)         # (B, d+K)
        out = self.net(inp)                          # (B, 2N)
        mu, log_sigma = out.chunk(2, dim=-1)         # 各(B, N)

        # clamp log_sigma防止数值不稳定
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)

        return mu, log_sigma
