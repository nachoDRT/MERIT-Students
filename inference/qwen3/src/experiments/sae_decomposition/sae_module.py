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
