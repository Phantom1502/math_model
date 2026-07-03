"""
model/layers.py — Các thành phần kiến trúc LLaMA-style
==========================================================
RMSNorm     : thay LayerNorm, bỏ mean-centering, rẻ hơn, ổn định hơn
RoPE        : Rotary Position Embedding, mã hóa vị trí tương đối qua xoay vector
SwiGLU      : FFN dùng SiLU-gated thay GELU, tăng năng lực biểu diễn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════
# RMSNorm
# ══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Khác LayerNorm: không trừ mean, chỉ chuẩn hóa theo RMS.
    Rẻ hơn (ít phép tính), được LLaMA/Mistral/Qwen dùng thay LayerNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tính theo float32 để ổn định số học khi dùng mixed precision
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x   = x * rms
        return (x * self.weight).to(dtype)


# ══════════════════════════════════════════════════════════════════════════
# RoPE — Rotary Position Embedding
# ══════════════════════════════════════════════════════════════════════════

def precompute_freqs_cis(d_head: int, max_seq: int, base: float = 10000.0) -> torch.Tensor:
    """
    Tính sẵn các giá trị cos/sin cho mỗi vị trí và mỗi cặp chiều.

    Returns: (max_seq, d_head/2) complex tensor — dùng polar form để xoay.
    """
    freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    t     = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)                          # (max_seq, d_head/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Áp dụng phép xoay RoPE lên Q hoặc K.

    x        : (B, n_heads, T, d_head)
    freqs_cis: (T, d_head/2) complex
    """
    B, H, T, D = x.shape

    # Chuyển x thành dạng complex: ghép cặp (x0,x1), (x2,x3), ...
    x_complex = torch.view_as_complex(
        x.float().reshape(B, H, T, D // 2, 2)
    )                                                   # (B, H, T, D/2)

    freqs_cis = freqs_cis[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)

    # Xoay bằng phép nhân số phức
    x_rotated = x_complex * freqs_cis                    # (B, H, T, D/2)

    # Chuyển lại về real, ghép về shape gốc
    out = torch.view_as_real(x_rotated).reshape(B, H, T, D)
    return out.type_as(x)


# ══════════════════════════════════════════════════════════════════════════
# SwiGLU FFN
# ══════════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """
    SwiGLU(x) = (SiLU(xW1) * xW3) W2

    Thay cho Linear → GELU → Linear truyền thống.
    Dùng 3 ma trận thay vì 2, nhưng hidden_dim được giảm xuống ~2/3
    để tổng số params tương đương MLP GELU thông thường (theo paper LLaMA).
    """

    def __init__(self, d_model: int, hidden_mult: float = 8/3, bias: bool = False):
        super().__init__()
        # hidden_dim ~ 8/3 * d_model, làm tròn về bội số của 64 cho hiệu quả GPU
        hidden_dim = int(hidden_mult * d_model)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=bias)   # gate
        self.w3 = nn.Linear(d_model, hidden_dim, bias=bias)   # up
        self.w2 = nn.Linear(hidden_dim, d_model, bias=bias)   # down

        # Init std=0.02 chuẩn. w2 nằm trên đường residual nên sẽ bị
        # MemoryBlock._scaled_init() ghi đè SAU khi block cha khởi tạo xong.
        for layer in [self.w1, self.w2, self.w3]:
            nn.init.normal_(layer.weight, std=0.02)
            if bias:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
