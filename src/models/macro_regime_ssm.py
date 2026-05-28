"""
macro_regime_ssm.py - 加入logit平滑的序列化版本
"""

import torch
import torch.nn as nn
from .asset_encoder   import AssetEncoder
from .tech_encoder    import TechEncoder
from .macro_encoder   import MacroEncoder
from .inference_net   import InferenceNet
from .prior_net       import PriorNet
from .film_ssm        import FiLMSSM
from .prediction_head import PredictionHead


class MacroRegimeSSM(nn.Module):
    def __init__(
        self,
        n_assets:  int   = 56,
        n_macro:   int   = 6,
        n_tech:    int   = 3,
        d_model:   int   = 64,
        n_regimes: int   = 4,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        dropout:   float = 0.1,
        L:         int   = 20,
        momentum:  float = 0.8,   # regime logit的动量平滑系数
    ):
        super().__init__()
        self.K        = n_regimes
        self.d        = d_model
        self.N        = n_assets
        self.L        = L
        self.momentum = momentum

        self.asset_encoder = AssetEncoder(n_assets, d_model, n_heads, n_layers, dropout)
        self.tech_encoder  = TechEncoder(n_tech, d_model, dropout)
        self.macro_encoder = MacroEncoder(n_macro, d_model, dropout)
        self.inference_net = InferenceNet(d_model, n_regimes, dropout, n_inputs=4)
        self.prior_net     = PriorNet(d_model, n_regimes, dropout)
        self.film_ssm      = FiLMSSM(d_model, n_regimes, dropout)
        self.pred_head     = PredictionHead(d_model, n_regimes, n_assets, dropout)

    def forward(
        self,
        x_seq:    torch.Tensor,
        tech_seq: torch.Tensor,
        m_seq:    torch.Tensor,
        tau:  float = 1.0,
        hard: bool  = False,
    ) -> dict:
        B = x_seq.shape[0]
        T = x_seq.shape[1] - self.L

        h = torch.zeros(B, self.d, device=x_seq.device)
        z = torch.full((B, self.K), 1.0/self.K, device=x_seq.device)

        # 用于logit平滑的历史logits
        prev_logits = torch.zeros(B, self.K, device=x_seq.device)

        mu_list, log_sigma_list = [], []
        z_list, prior_list, z_prev_list = [], [], []

        for t in range(T):
            x_win    = x_seq[:, t:t+self.L, :]
            tech_win = tech_seq[:, t:t+self.L, :, :]
            m_win    = m_seq[:, t:t+self.L, :]

            x_emb    = self.asset_encoder(x_win)
            tech_emb = self.tech_encoder(tech_win)
            m_emb    = self.macro_encoder(m_win)

            z_prev_list.append(z)

            # 获取当前logits
            z_emb = self.inference_net.z_embed(z)
            parts = [x_emb, m_emb, z_emb, tech_emb]
            raw_logits = self.inference_net.net(torch.cat(parts, dim=-1))

            # Momentum平滑：当前logits = momentum*历史 + (1-momentum)*当前
            # 这让regime倾向于保持，除非新证据很强
            smooth_logits = self.momentum * prev_logits.detach() + \
                           (1 - self.momentum) * raw_logits
            prev_logits = smooth_logits

            # Straight-through hard assignment on smoothed logits
            import torch.nn.functional as F
            soft  = F.softmax(smooth_logits, dim=-1)
            index = soft.argmax(dim=-1)
            hard_z = F.one_hot(index, self.K).float()
            z_new  = hard_z + (soft - soft.detach())

            prior = self.prior_net(m_emb, z)
            h     = self.film_ssm(x_emb, h, z_new)
            mu, log_sigma = self.pred_head(h, z_new)

            mu_list.append(mu)
            log_sigma_list.append(log_sigma)
            z_list.append(z_new)
            prior_list.append(prior)

            z = z_new
            h = h.detach()

        return {
            'mu':         torch.stack(mu_list,        dim=1),
            'log_sigma':  torch.stack(log_sigma_list, dim=1),
            'z_seq':      torch.stack(z_list,         dim=1),
            'prior_seq':  torch.stack(prior_list,     dim=1),
            'z_prev_seq': torch.stack(z_prev_list,    dim=1),
        }

    def get_regime_sequence(self, x_seq, tech_seq, m_seq):
        with torch.no_grad():
            out = self.forward(x_seq, tech_seq, m_seq, hard=True)
            return out['z_seq'].argmax(dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
