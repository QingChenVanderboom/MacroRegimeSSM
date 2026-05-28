"""
baselines.py
对比baseline模型

1. LinearBaseline    - 简单线性回归
2. LSTMBaseline      - 标准LSTM，无regime
3. TransformerBaseline - 标准Transformer时序预测
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. Linear Baseline
# ─────────────────────────────────────────────

class LinearBaseline(nn.Module):
    """
    展平历史窗口后接线性层
    输入：x_seq (B, L, N)
    输出：mu (B, N), log_sigma (B, N)
    """
    def __init__(self, n_assets=56, L=20, **kwargs):
        super().__init__()
        self.net = nn.Linear(L * n_assets, n_assets * 2)
        self.N = n_assets
        nn.init.normal_(self.net.weight, std=0.01)
        nn.init.zeros_(self.net.bias)

    def forward(self, x_seq, tech_seq=None, m_seq=None, **kwargs):
        B = x_seq.shape[0]
        out = self.net(x_seq.reshape(B, -1))          # (B, 2N)
        mu, log_sigma = out.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-6, 2)
        return {
            'mu': mu, 'log_sigma': log_sigma,
            'z_t': torch.zeros(B, 1, device=x_seq.device),
            'prior_probs': torch.ones(B, 1, device=x_seq.device),
            'z_prev': torch.zeros(B, 1, device=x_seq.device),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 2. LSTM Baseline
# ─────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    标准LSTM，无regime，无宏观
    输入：x_seq (B, L, N)
    输出：mu (B, N), log_sigma (B, N)
    """
    def __init__(self, n_assets=56, d_model=64, n_layers=2,
                 dropout=0.1, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(n_assets, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets * 2),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_seq, tech_seq=None, m_seq=None, **kwargs):
        B = x_seq.shape[0]
        x = self.input_proj(x_seq)              # (B, L, d)
        out, _ = self.lstm(x)                   # (B, L, d)
        h = out[:, -1, :]                       # (B, d)
        pred = self.head(h)                     # (B, 2N)
        mu, log_sigma = pred.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-6, 2)
        return {
            'mu': mu, 'log_sigma': log_sigma,
            'z_t': torch.zeros(B, 1, device=x_seq.device),
            'prior_probs': torch.ones(B, 1, device=x_seq.device),
            'z_prev': torch.zeros(B, 1, device=x_seq.device),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 3. Transformer Baseline
# ─────────────────────────────────────────────

class TransformerBaseline(nn.Module):
    """
    标准Transformer时序预测，无regime，无宏观
    """
    def __init__(self, n_assets=56, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        import math
        self.input_proj = nn.Linear(n_assets, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets * 2),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_seq, tech_seq=None, m_seq=None, **kwargs):
        B = x_seq.shape[0]
        x = self.input_proj(x_seq)
        x = self.transformer(x)
        h = x[:, -1, :]
        pred = self.head(h)
        mu, log_sigma = pred.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-6, 2)
        return {
            'mu': mu, 'log_sigma': log_sigma,
            'z_t': torch.zeros(B, 1, device=x_seq.device),
            'prior_probs': torch.ones(B, 1, device=x_seq.device),
            'z_prev': torch.zeros(B, 1, device=x_seq.device),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# 4. MacroLSTM（消融：有宏观无regime）
# ─────────────────────────────────────────────

class MacroLSTMBaseline(nn.Module):
    """
    LSTM + 宏观因子，无regime
    用于消融：证明regime带来的增益超过单纯加宏观
    """
    def __init__(self, n_assets=56, n_macro=6, d_model=64,
                 n_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.asset_proj = nn.Linear(n_assets, d_model)
        self.macro_proj = nn.Linear(n_macro, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        # 融合宏观
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets * 2),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_seq, tech_seq=None, m_seq=None, **kwargs):
        B = x_seq.shape[0]
        x = self.asset_proj(x_seq)
        out, _ = self.lstm(x)
        h_asset = out[:, -1, :]

        # 宏观编码（取最后时间步）
        m = self.macro_proj(m_seq)
        h_macro = m[:, -1, :]

        h = self.fusion(torch.cat([h_asset, h_macro], dim=-1))
        h = torch.relu(h)
        pred = self.head(h)
        mu, log_sigma = pred.chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(-6, 2)
        return {
            'mu': mu, 'log_sigma': log_sigma,
            'z_t': torch.zeros(B, 1, device=x_seq.device),
            'prior_probs': torch.ones(B, 1, device=x_seq.device),
            'z_prev': torch.zeros(B, 1, device=x_seq.device),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
