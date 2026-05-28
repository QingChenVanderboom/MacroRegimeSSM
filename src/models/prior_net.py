"""
prior_net.py - 固定均匀先验版本
不让模型学习prior，直接用均匀分布
这样KL = H(q) - log(K)，模型必须学到有意义的后验才能最小化KL
"""

import torch
import torch.nn as nn


class PriorNet(nn.Module):
    def __init__(self, d_model: int = 64, n_regimes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.K = n_regimes
        # 保留参数占位，但forward返回固定均匀分布
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, m_emb: torch.Tensor, z_prev: torch.Tensor) -> torch.Tensor:
        """
        返回均匀先验 (B, K)
        KL(q||uniform) = log(K) - H(q)
        最小化KL等价于最大化后验熵，自然防止collapse
        """
        B = m_emb.shape[0]
        return torch.full(
            (B, self.K), 1.0 / self.K,
            device=m_emb.device, dtype=m_emb.dtype
        )

    def count_parameters(self):
        return 0
