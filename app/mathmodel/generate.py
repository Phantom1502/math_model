"""
generate.py — Sinh văn bản từ model đã train
================================================
Sliding window đơn giản: khi context vượt max_seq thì cắt phần đầu,
không cần prefill hay flush vào memory.

Usage:
    from generate import load_model_for_inference, generate

    model, tokenizer, cfg = load_model_for_inference("checkpoints/best.pt")
    print(generate(model, tokenizer, cfg, "Trí tuệ nhân tạo là"))
"""

import torch
import torch.nn.functional as F

from model import causal_mask


# ══════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════

def _sample_next(
    logits     : torch.Tensor,   # (1, vocab)
    temperature: float,
    top_k      : int,
    top_p      : float,
) -> int:
    logits = logits / temperature

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs   = F.softmax(sorted_logits, dim=-1)
        cumprob = torch.cumsum(probs, dim=-1)
        remove  = cumprob - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()


# ══════════════════════════════════════════════════════════════════════════
# Generate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model,
    tokenizer,
    cfg,
    prompt         : str,
    max_new        : int   = 100,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 0.95,
    new_token_only : bool  = False,
) -> str:
    """
    Sinh văn bản từ prompt với sliding window khi context vượt max_seq.

    Args:
        prompt        : văn bản đầu vào
        max_new       : số token tối đa sinh thêm
        temperature   : nhiệt độ sampling
        top_k         : top-k filtering
        top_p         : nucleus sampling
        new_token_only: True → chỉ trả về phần sinh thêm
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Cắt prompt về max_seq nếu quá dài
    active = prompt_ids[-max_seq:] if len(prompt_ids) > max_seq else prompt_ids
    ids    = torch.tensor([active], dtype=torch.long, device=device)

    generated = []

    for _ in range(max_new):
        # Sliding window: giữ max_seq token cuối
        if ids.size(1) > max_seq:
            ids = ids[:, -max_seq:]

        T      = ids.size(1)
        logits = model(ids, attn_mask=causal_mask(T, device))
        next_tok = _sample_next(logits[:, -1, :], temperature, top_k, top_p)

        generated.append(next_tok)
        ids = torch.cat([ids, torch.tensor([[next_tok]], device=device)], dim=1)

        if next_tok == tokenizer.eos_id:
            break

    if new_token_only:
        return tokenizer.decode(generated)
    return tokenizer.decode(prompt_ids + generated)


# ══════════════════════════════════════════════════════════════════════════
# Load model for inference
# ══════════════════════════════════════════════════════════════════════════

def load_model_for_inference(checkpoint_path: str, device: str = None, fallback_cfg=None):
    """
    Load model + tokenizer + config từ checkpoint.
    Đọc model_cfg đã lưu trong checkpoint để build đúng kiến trúc lúc train.
    """
    from config import get_100m_config, ModelConfig
    from tokenizer import load_tokenizer
    from model import build_model
    from utils import load_checkpoint
    import torch as _torch

    device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    cfg    = fallback_cfg or get_100m_config()
    cfg.train.device = device

    ckpt_raw = _torch.load(checkpoint_path, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        cfg.model = ModelConfig(**ckpt_raw["model_cfg"])
        print(f"  model_cfg: d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
              f"max_seq={cfg.model.max_seq}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg — dùng config mặc định.")

    tokenizer            = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    return model, tokenizer, cfg


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"

    model, tokenizer, cfg = load_model_for_inference(ckpt_path)

    prompts = [
        "Trí tuệ nhân tạo là",
        "Lịch sử Việt Nam bắt đầu",
        "Mô hình ngôn ngữ lớn có khả năng",
    ]
    for p in prompts:
        out = generate(model, tokenizer, cfg, p, max_new=80)
        print(f"\n['{p}']\n{out}\n" + "-" * 60)