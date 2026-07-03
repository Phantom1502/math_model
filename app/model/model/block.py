"""
model/block.py — Transformer Block (LLaMA-style)
=================================================
Luồng:
    x → SelfAttentionRoPE (Pre-Norm bên trong) → residual → FFN (Pre-Norm) → out
"""

import math
import torch
import torch.nn as nn

from .attention import SelfAttentionRoPE
from .layers import RMSNorm, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model  : int,
        n_heads  : int,
        dropout  : float = 0.1,
        n_layers : int   = 8,
    ):
        super().__init__()

        # Self-Attention (Pre-Norm nằm bên trong SelfAttentionRoPE)
        self.self_attn = SelfAttentionRoPE(d_model, n_heads, dropout=dropout)

        # FFN
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, bias=False)

        self._scaled_init(n_layers)

    def _scaled_init(self, n_layers: int):
        """Scale init cho các projection nằm trên đường residual."""
        scale = 1.0 / math.sqrt(2 * n_layers)
        nn.init.normal_(self.self_attn.Wo.weight, std=0.02 * scale)
        nn.init.normal_(self.ffn.w2.weight,       std=0.02 * scale)

    def forward(
        self,
        x        : torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(x, freqs_cis, attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x