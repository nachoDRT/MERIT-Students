"""Minimal TopK SAE wrapper for Qwen-Scope checkpoints."""

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, checkpoint_path: str, k: int = 100, device: str = 'cuda'):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.W_enc = nn.Parameter(ckpt['W_enc'], requires_grad=False)  # (D, d)
        self.b_enc = nn.Parameter(ckpt['b_enc'], requires_grad=False)  # (D,)
        self.W_dec = nn.Parameter(ckpt['W_dec'], requires_grad=False)  # (d, D)
        self.b_dec = nn.Parameter(ckpt['b_dec'], requires_grad=False)  # (d,)
        self.k = k
        self.D, self.d = self.W_enc.shape  # dictionary size, model dim

    def encode(self, x: torch.Tensor, apply_topk: bool = True) -> torch.Tensor:
        """x: (..., d) → features: (..., D)"""
        pre = x @ self.W_enc.T + self.b_enc
        if not apply_topk:
            return pre
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """f: (..., D) → x_reconstructed: (..., d)"""
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def explained_variance(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Fraction of variance explained (R²), centered per dimension.

    Standard SAE reconstruction metric: 1 − FVU, where
        FVU = Σ‖x − x̂‖² / Σ‖x − mean(x)‖²
    and mean(x) is taken per dimension across all samples. Unlike cosine
    similarity it penalises both direction and scale error.

    Returns 1.0 for perfect reconstruction, 0.0 for predicting the
    constant mean, and negative values when the reconstruction is worse
    than that mean.

    NOTE: the baseline is the *mean* vector, so this is only meaningful
    when the data has real spread (e.g. diverse prompts). For a set of
    near-identical activations (same prompt, concentrated distribution)
    the mean is an almost-perfect predictor and this metric collapses to
    a large negative number even when each activation is reconstructed
    faithfully — use `reconstruction_fidelity` in that case.
    """
    x       = x.reshape(-1, x.shape[-1]).float()
    x_recon = x_recon.reshape(-1, x_recon.shape[-1]).float()
    resid_ss = (x - x_recon).pow(2).sum()
    total_ss = (x - x.mean(dim=0, keepdim=True)).pow(2).sum().clamp(min=1e-12)
    return float(1.0 - resid_ss / total_ss)


def reconstruction_fidelity(x: torch.Tensor, x_recon: torch.Tensor) -> float:
    """Uncentered fraction explained: 1 − Σ‖x − x̂‖² / Σ‖x‖².

    Baseline is the zero vector instead of the mean, so it measures
    absolute reconstruction fidelity (direction + magnitude) regardless
    of how concentrated the dataset is. Appropriate when judging whether
    individual activations are faithfully reconstructed before trusting
    their SAE feature decomposition.
    """
    x       = x.reshape(-1, x.shape[-1]).float()
    x_recon = x_recon.reshape(-1, x_recon.shape[-1]).float()
    resid_ss = (x - x_recon).pow(2).sum()
    total_ss = x.pow(2).sum().clamp(min=1e-12)
    return float(1.0 - resid_ss / total_ss)
