"""
model/attention.py — Self-Attention với RoPE
==============================================
SelfAttentionRoPE : thay nn.MultiheadAttention, hỗ trợ RoPE và no-bias
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, apply_rope


class SelfAttentionRoPE(nn.Module):
    """
    Self-attention chuẩn, có RoPE áp lên Q/K, không bias trên các Linear.
    Pre-Norm (RMSNorm) nằm bên trong — block không cần norm trước khi gọi.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.norm = RMSNorm(d_model)

        # no bias — LLaMA-style
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Wo nằm trên đường residual — bị _scaled_init() trong block ghi đè sau
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.normal_(layer.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        h, dh   = self.n_heads, self.d_head

        x_normed = self.norm(x)

        q = self.Wq(x_normed).view(B, T, h, dh).transpose(1, 2)
        k = self.Wk(x_normed).view(B, T, h, dh).transpose(1, 2)
        v = self.Wv(x_normed).view(B, T, h, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if attn_mask is not None:
            scores = scores + attn_mask

        weights = self.dropout(F.softmax(scores, dim=-1))
        out     = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)