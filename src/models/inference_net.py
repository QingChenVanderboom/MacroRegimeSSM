"""
inference_net.py - VQ风格，用straight-through hard assignment
训练时：forward pass用argmax（hard），backward pass用软分布梯度（straight-through）
这样regime分配是真正离散的，不会出现均匀软分布退化解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InferenceNet(nn.Module):
    def __init__(
        self,
        d_model:   int = 64,
        n_regimes: int = 4,
        dropout:   float = 0.1,
        n_inputs:  int = 4,
    ):
        super().__init__()
        self.K = n_regimes
        self.z_embed = nn.Linear(n_regimes, d_model)

        self.net = nn.Sequential(
            nn.Linear(d_model * n_inputs, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regimes),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(
        self,
        x_emb:    torch.Tensor,
        m_emb:    torch.Tensor,
        z_prev:   torch.Tensor,
        tau:      float = 1.0,
        hard:     bool  = False,
        tech_emb: torch.Tensor = None,
    ) -> tuple:
        z_emb = self.z_embed(z_prev)
        parts = [x_emb, m_emb, z_emb]
        if tech_emb is not None:
            parts.append(tech_emb)

        h = torch.cat(parts, dim=-1)
        logits = self.net(h)                        # (B, K)
        soft   = F.softmax(logits, dim=-1)          # (B, K) 软分布

        # Straight-through hard assignment
        # forward: one-hot (hard discrete)
        # backward: 梯度通过soft流回
        index  = soft.argmax(dim=-1)                # (B,)
        hard_z = F.one_hot(index, self.K).float()   # (B, K)
        # straight-through: hard_z + (soft - soft.detach())
        z_st   = hard_z + (soft - soft.detach())    # (B, K)

        return z_st, logits
