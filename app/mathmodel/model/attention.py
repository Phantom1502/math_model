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
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.norm = RMSNorm(d_model)

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.normal_(layer.weight, std=0.02)

        self.dropout = dropout

    def forward(self, x, freqs_cis, attn_mask=None):
        B, T, D = x.shape
        h, dh   = self.n_heads, self.d_head
        x_normed = self.norm(x)

        q = self.Wq(x_normed).view(B, T, h, dh).transpose(1, 2)
        k = self.Wk(x_normed).view(B, T, h, dh).transpose(1, 2)
        v = self.Wv(x_normed).view(B, T, h, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        dropout_p = self.dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)