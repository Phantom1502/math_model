"""
model/lm.py — MemoryLM (LLaMA-style, không có Context Memory)
===============================================================
Kỹ thuật cốt lõi:
    - RMSNorm + Pre-Norm  : bên trong SelfAttentionRoPE và trước FFN
    - SwiGLU              : trong mỗi TransformerBlock
    - No bias             : toàn bộ Linear đều bias=False
    - RoPE                : áp lên Q/K trong self-attention, không có pos_emb tuyệt đối
    - Scaled init         : 1/sqrt(2*n_layers) cho projection trên đường residual
    - Weight tying        : lm_head.weight = token_emb.weight
"""

import torch
import torch.nn as nn

from .block import TransformerBlock
from .layers import RMSNorm, precompute_freqs_cis


class MemoryLM(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        d_model    : int   = 512,
        n_heads    : int   = 8,
        n_layers   : int   = 8,
        max_seq    : int   = 512,
        dropout    : float = 0.1,
        rope_base  : float = 10000.0,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.max_seq  = max_seq

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, n_layers=n_layers)
            for _ in range(n_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        d_head    = d_model // n_heads
        freqs_cis = precompute_freqs_cis(d_head, max_seq * 2, base=rope_base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        nn.init.normal_(self.token_emb.weight, std=0.02)

    def num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids : (B, T)
        Returns   : logits (B, T, vocab_size)
        """
        B, T   = input_ids.shape
        device = input_ids.device

        x         = self.drop(self.token_emb(input_ids))
        freqs_cis = self.freqs_cis.to(device)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        return self.lm_head(self.norm_out(x))


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Causal mask additive cho autoregressive attention."""
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    return mask.masked_fill(mask.bool(), float("-inf"))


def build_model(cfg) -> MemoryLM:
    """Entry point xây model từ ModelConfig."""
    return MemoryLM(
        vocab_size = cfg.model.vocab_size,
        d_model    = cfg.model.d_model,
        n_heads    = cfg.model.n_heads,
        n_layers   = cfg.model.n_layers,
        max_seq    = cfg.model.max_seq,
        dropout    = cfg.model.dropout,
        rope_base  = getattr(cfg.model, "rope_base", 10000.0),
    )