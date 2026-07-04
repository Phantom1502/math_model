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
    - KV-cache (mới)      : forward(..., use_cache=True) trả thêm kv_cache để
                             generate.py decode 1 token/bước thay vì forward
                             lại toàn bộ sequence mỗi lần sinh token mới.

────────────────────────────────────────────────────────────────────────────
TƯƠNG THÍCH NGƯỢC — QUAN TRỌNG:

use_cache mặc định = False. Khi đó forward() trả về DUY NHẤT logits, y hệt
hành vi cũ 100% (freqs_cis luôn lấy từ vị trí 0, không quan tâm start_pos).
Toàn bộ code train (trainer/base.py) và benchmark (benchmark.py) gọi
model(ids, attn_mask=mask) không cần sửa gì.

use_cache=True chỉ dùng trong generate.py cho suy luận autoregressive:
    - Bước prefill : model(prompt_ids, attn_mask=causal_mask(T,...),
                           use_cache=True, start_pos=0)
                     → trả về (logits, kv_cache)
    - Bước decode  : model(next_token_id,  # shape (1,1)
                           attn_mask=None, use_cache=True,
                           kv_cache=kv_cache, start_pos=cur_pos)
                     → trả về (logits, kv_cache_mới)

GIỚI HẠN: freqs_cis chỉ precompute cho max_seq*2 vị trí (xem
precompute_freqs_cis bên dưới). Do đó tổng số vị trí dùng trong một lần
generate (start_pos + T) không được vượt quá max_seq*2 — generate.py tự
giới hạn max_new theo ràng buộc này (không còn sliding-window giữa chừng
như bản cũ, vì cắt bớt kv_cache giữa chừng sẽ làm lệch vị trí RoPE tuyệt
đối của phần còn lại).
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor = None,
        kv_cache             = None,
        start_pos: int        = 0,
        use_cache: bool       = False,
    ):
        """
        input_ids : (B, T) — T = toàn bộ prompt lúc prefill/train, hoặc T=1
                    lúc decode từng bước với use_cache=True.
        kv_cache  : list[tuple(k,v)] độ dài n_layers — cache của TỪNG layer
                    từ (các) bước gọi trước đó. None nếu là bước prefill
                    hoặc không dùng cache.
        start_pos : vị trí tuyệt đối của token ĐẦU TIÊN trong input_ids —
                    dùng để lấy đúng slice RoPE (chỉ có tác dụng khi
                    use_cache=True; bỏ qua khi use_cache=False để giữ
                    nguyên hành vi cũ).
        use_cache : False (mặc định) → trả về logits (giống hệt bản cũ).
                    True             → trả về (logits, new_kv_cache).
        """
        B, T   = input_ids.shape
        device = input_ids.device

        x         = self.drop(self.token_emb(input_ids))
        freqs_cis = self.freqs_cis.to(device)

        if use_cache:
            freqs_cis_slice = freqs_cis[start_pos : start_pos + T]
        else:
            freqs_cis_slice = freqs_cis[:T]

        new_kv_cache = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past_kv = kv_cache[i] if (use_cache and kv_cache is not None) else None
            x, present_kv = block(x, freqs_cis=freqs_cis_slice, attn_mask=attn_mask, past_kv=past_kv)
            if use_cache:
                new_kv_cache.append(present_kv)

        logits = self.lm_head(self.norm_out(x))

        if use_cache:
            return logits, new_kv_cache
        return logits


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