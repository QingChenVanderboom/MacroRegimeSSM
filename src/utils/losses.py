"""
losses.py - hard cap diversity loss
"""

import torch
import torch.nn.functional as F
import math


def gaussian_nll(x_target, mu, log_sigma):
    sigma = log_sigma.exp()
    nll = 0.5 * (
        torch.log(2 * torch.tensor(math.pi, device=x_target.device))
        + 2 * log_sigma
        + ((x_target - mu) / sigma) ** 2
    )
    return nll.mean()


def smooth_loss_hard(z_seq):
    diff = (z_seq[:, 1:] - z_seq[:, :-1]).abs().sum(dim=-1)
    return diff.mean()


def listnet_loss(mu, x_target):
    scale   = x_target.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_norm  = x_target / scale
    mu_norm = mu / scale
    p_true   = F.softmax(x_norm,  dim=-1)
    log_pred = F.log_softmax(mu_norm, dim=-1)
    return -(p_true * log_pred).sum(dim=-1).mean()


def diversity_loss(z_seq, max_frac=0.4):
    """
    惩罚任何一个regime占比超过max_frac
    max_frac=0.4即均匀分布上限40%（K=4时均匀是25%）
    """
    marginal = z_seq.mean(dim=(0, 1))              # (K,)
    excess   = F.relu(marginal - max_frac)         # 超出部分
    return excess.sum() * 10.0                     # 乘大系数保证梯度足够强


def elbo_loss(
    x_target, mu, log_sigma, z_seq, prior_seq, z_prev_seq,
    lambda1=1.0, lambda2=0.3, lambda3=5.0, lambda4=2.0,
    free_bits=0.0,
):
    nll  = gaussian_nll(x_target, mu, log_sigma)
    smo  = smooth_loss_hard(z_seq)
    rl   = listnet_loss(mu, x_target)
    div  = diversity_loss(z_seq, max_frac=0.4)

    total = nll + lambda2 * smo + lambda3 * rl + lambda4 * div

    marginal = z_seq.mean(dim=(0, 1))
    regime_entropy = -(marginal * torch.log(marginal + 1e-8)).sum()

    return {
        'total':      total,
        'nll':        nll.detach(),
        'regime_kl':  div.detach(),
        'smooth':     smo.detach(),
        'rank_loss':  rl.detach(),
        'entropy':    regime_entropy.detach(),
    }


def kl_annealing_weight(step, warmup_steps=2000):
    return min(1.0, step / max(warmup_steps, 1))


def gumbel_tau_annealing(step, total_steps, tau_start=1.0, tau_end=0.1):
    frac = min(1.0, step / max(total_steps, 1))
    return tau_start + frac * (tau_end - tau_start)
