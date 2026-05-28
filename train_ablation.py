"""
train_ablation.py
消融实验：no-macro 和 no-regime
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from src.data.dataset import build_dataloaders
from src.models.asset_encoder import AssetEncoder
from src.models.tech_encoder import TechEncoder
from src.models.macro_encoder import MacroEncoder
from src.models.film_ssm import FiLMSSM
from src.models.prediction_head import PredictionHead
from src.utils.trainer import Trainer


class NoMacroSSM(nn.Module):
    """
    去掉宏观输入：推断网络只用资产+技术面
    验证：宏观因子对regime识别的贡献
    """
    def __init__(self, n_assets=56, n_macro=6, n_tech=3,
                 d_model=64, n_regimes=4, n_heads=4, n_layers=2,
                 dropout=0.1, L=20, momentum=0.8):
        super().__init__()
        self.K = n_regimes
        self.d = d_model
        self.N = n_assets
        self.L = L
        self.momentum = momentum

        self.asset_encoder = AssetEncoder(n_assets, d_model, n_heads, n_layers, dropout)
        self.tech_encoder  = TechEncoder(n_tech, d_model, dropout)
        # 无宏观编码器

        # 推断网络：只用x_emb + tech_emb + z_prev（去掉m_emb）
        self.z_embed = nn.Linear(n_regimes, d_model)
        self.inf_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_regimes),
        )
        self.film_ssm  = FiLMSSM(d_model, n_regimes, dropout)
        self.pred_head = PredictionHead(d_model, n_regimes, n_assets, dropout)

    def forward(self, x_seq, tech_seq, m_seq, tau=1.0, hard=False):
        B = x_seq.shape[0]
        T = x_seq.shape[1] - self.L
        h = torch.zeros(B, self.d, device=x_seq.device)
        z = torch.full((B, self.K), 1.0/self.K, device=x_seq.device)
        prev_logits = torch.zeros(B, self.K, device=x_seq.device)

        mu_list, ls_list, z_list, zp_list = [], [], [], []

        for t in range(T):
            xw = x_seq[:, t:t+self.L]
            tw = tech_seq[:, t:t+self.L]

            xe = self.asset_encoder(xw)
            te = self.tech_encoder(tw)
            ze = self.z_embed(z)

            raw = self.inf_net(torch.cat([xe, te, ze], dim=-1))
            smooth = self.momentum * prev_logits.detach() + (1-self.momentum) * raw
            prev_logits = smooth

            soft   = F.softmax(smooth, dim=-1)
            index  = soft.argmax(dim=-1)
            z_new  = F.one_hot(index, self.K).float()
            z_new  = z_new + (soft - soft.detach())

            zp_list.append(z)
            h = self.film_ssm(xe, h, z_new)
            mu, ls = self.pred_head(h, z_new)
            mu_list.append(mu)
            ls_list.append(ls)
            z_list.append(z_new)
            z = z_new
            h = h.detach()

        return {
            'mu':         torch.stack(mu_list,  dim=1),
            'log_sigma':  torch.stack(ls_list,  dim=1),
            'z_seq':      torch.stack(z_list,   dim=1),
            'prior_seq':  torch.full((B,T,self.K), 1.0/self.K, device=x_seq.device),
            'z_prev_seq': torch.stack(zp_list,  dim=1),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NoRegimeSSM(nn.Module):
    """
    去掉regime模块：纯SSM，无离散状态
    验证：regime切换机制的贡献
    """
    def __init__(self, n_assets=56, n_macro=6, n_tech=3,
                 d_model=64, n_regimes=4, n_heads=4, n_layers=2,
                 dropout=0.1, L=20, **kwargs):
        super().__init__()
        self.d = d_model
        self.N = n_assets
        self.L = L
        self.K = 1  # dummy

        self.asset_encoder = AssetEncoder(n_assets, d_model, n_heads, n_layers, dropout)
        self.tech_encoder  = TechEncoder(n_tech, d_model, dropout)
        self.macro_encoder = MacroEncoder(n_macro, d_model, dropout)

        # 普通GRU，无FiLM调制
        self.gru  = nn.GRUCell(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_assets * 2),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_seq, tech_seq, m_seq, tau=1.0, hard=False):
        B = x_seq.shape[0]
        T = x_seq.shape[1] - self.L
        h = torch.zeros(B, self.d, device=x_seq.device)

        mu_list, ls_list = [], []

        for t in range(T):
            xw = x_seq[:, t:t+self.L]
            tw = tech_seq[:, t:t+self.L]
            mw = m_seq[:, t:t+self.L]

            xe = self.asset_encoder(xw)
            te = self.tech_encoder(tw)
            me = self.macro_encoder(mw)

            inp = torch.cat([xe, te, me], dim=-1)
            h   = self.norm(self.gru(inp, h))

            pred = self.head(h)
            mu, ls = pred.chunk(2, dim=-1)
            ls = ls.clamp(-6, 2)
            mu_list.append(mu)
            ls_list.append(ls)
            h = h.detach()

        dummy_z = torch.full((B, T, 1), 1.0, device=x_seq.device)
        return {
            'mu':         torch.stack(mu_list, dim=1),
            'log_sigma':  torch.stack(ls_list, dim=1),
            'z_seq':      dummy_z,
            'prior_seq':  dummy_z,
            'z_prev_seq': dummy_z,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


ABLATIONS = {
    'no_macro':  NoMacroSSM,
    'no_regime': NoRegimeSSM,
}


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")

    train_loader, val_loader, _, N, F, _ = build_dataloaders(
        L=args.L, T=args.T,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_stride=5,
    )

    for name, ModelClass in ABLATIONS.items():
        if args.model != 'all' and args.model != name:
            continue

        print(f"\n{'='*50}\nAblation: {name}\n{'='*50}")
        model = ModelClass(n_assets=N, n_macro=F, d_model=64, L=args.L)
        print(f"参数量: {model.count_parameters():,}")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.lr,
            max_epochs=args.epochs,
            warmup_steps=3000,
            lambda2=0.3,
            lambda3=5.0,
            lambda4=2.0,
            patience=20,
            save_dir='experiments/checkpoints',
            run_name=f'ablation_{name}_d64_L{args.L}_T{args.T}',
        )
        trainer.fit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',       type=str,   default='all',
                        choices=['all', 'no_macro', 'no_regime'])
    parser.add_argument('--L',           type=int,   default=20)
    parser.add_argument('--T',           type=int,   default=64)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int,   default=4)
    args = parser.parse_args()
    main(args)
