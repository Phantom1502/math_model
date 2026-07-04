"""
model/attention.py — Self-Attention với RoPE + KV-cache
=========================================================
SelfAttentionRoPE : thay nn.MultiheadAttention, hỗ trợ RoPE, no-bias,
                     và KV-cache cho generate() (decode 1 token/bước
                     thay vì forward lại toàn bộ sequence mỗi lần).

Tương thích ngược: gọi forward() không truyền past_kv (mặc định None)
cho kết quả HỆT như code cũ — chỉ thêm giá trị trả về thứ 2 (present_kv),
các nơi gọi cũ (train loop, benchmark log-prob) không dùng KV-cache nên
không bị ảnh hưởng, chỉ cần nhận thêm 1 giá trị bỏ qua nếu muốn (xem
model/lm.py — đã xử lý việc này ở tầng MemoryLM.forward qua use_cache).
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

    def forward(self, x, freqs_cis, attn_mask=None, past_kv=None):
        """
        freqs_cis : slice CHÍNH XÁC ứng với vị trí tuyệt đối của x — đã được
                    cắt sẵn ở MemoryLM.forward theo start_pos. Hàm này KHÔNG
                    tự cắt lại (apply_rope vẫn làm freqs_cis[:T] nhưng T ở
                    đây luôn khớp sẵn nên là no-op, giữ nguyên layers.py).

        past_kv   : tuple (k_cache, v_cache), mỗi cái shape
                    (B, n_heads, T_past, d_head) — hoặc None nếu không dùng
                    cache (huấn luyện) hoặc đây là bước prefill đầu tiên.

        Returns   : (out, present_kv)
                    present_kv luôn được trả về (kể cả khi past_kv=None) —
                    là (k, v) SAU khi đã nối với past_kv (nếu có), để tầng
                    trên (MemoryLM) build kv_cache mới truyền cho bước sau.
        """
        B, T, D = x.shape
        h, dh   = self.n_heads, self.d_head
        x_normed = self.norm(x)

        q = self.Wq(x_normed).view(B, T, h, dh).transpose(1, 2)
        k = self.Wk(x_normed).view(B, T, h, dh).transpose(1, 2)
        v = self.Wv(x_normed).view(B, T, h, dh).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)   # (B, h, T_past + T, dh)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)

        dropout_p = self.dropout if self.training else 0.0

        # attn_mask=None + is_causal=False khi decode 1 token: đúng về mặt
        # nhân quả vì query (1 token mới nhất) được phép attend TOÀN BỘ
        # key trong cache (đều là quá khứ so với nó) — không cần mask tam giác.
        out = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out), present_kv