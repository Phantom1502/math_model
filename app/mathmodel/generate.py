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

def _sample_next_ids(
    logits     : torch.Tensor,   # (B, vocab)
    temperature: float,
    top_k      : int,
    top_p      : float,
) -> torch.Tensor:
    """Batched version — trả về tensor (B,), mỗi phần tử là 1 token sample
    cho hàng tương ứng. Logic giống hệt _sample_next, chỉ khác không .item()."""
    logits = logits / temperature

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs   = F.softmax(sorted_logits, dim=-1)
        cumprob = torch.cumsum(probs, dim=-1)
        remove  = cumprob - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)  # (B,)


def _sample_next(
    logits     : torch.Tensor,   # (1, vocab)
    temperature: float,
    top_k      : int,
    top_p      : float,
) -> int:
    """Giữ nguyên chữ ký cũ (trả về int) cho generate() — tái dùng _sample_next_ids."""
    return _sample_next_ids(logits, temperature, top_k, top_p).item()


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
    add_bos        : bool  = False,
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
        add_bos       : True → thêm bos_id vào đầu prompt trước khi generate.

                        BẮT BUỘC bật khi generate từ checkpoint đã qua SFT
                        (SFTDataset._build_example luôn thêm bos_id ở đầu
                        MỌI sample lúc train — token "Problem:" luôn nằm ở
                        vị trí 1, không phải vị trí 0). Nếu generate() không
                        thêm lại bos ở đây, toàn bộ vị trí token (và do đó
                        RoPE) bị lệch 1 so với lúc train, phá format đã học
                        dù model đã hội tụ tốt.

                        Mặc định False để KHÔNG đổi hành vi cũ — data pretrain
                        (TokenChunkDataset trong dataset.py) không hề chèn
                        bos/eos giữa các đoạn cắt, nên generate cho model
                        pretrain-only / dùng trong benchmark.py vẫn nên giữ
                        add_bos=False như trước.
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos:
        prompt_ids = [tokenizer.bos_id] + prompt_ids

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
# Generate batch — nhiều continuation SONG SONG cho CÙNG 1 prompt
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    cfg,
    prompt         : str,
    batch_size     : int,
    max_new        : int   = 100,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 0.95,
    add_bos        : bool  = False,
) -> list[str]:
    """
    Sinh `batch_size` continuation ĐỘC LẬP cho CÙNG một prompt, xử lý song
    song trong 1 batch — thay vì gọi generate() tuần tự batch_size lần.

    TẠI SAO CẦN HÀM NÀY (xem thêm scripts/label_prm_data.py — Math-Shepherd):
    mỗi step cần k rollout xuất phát từ CÙNG một prefix. Gọi generate() k
    lần tuần tự (batch=1) nghĩa là trả phí forward qua toàn bộ n_layers
    (ở đây 30 layer) k LẦN — với sequence ngắn (~100-300 token), phần lớn
    thời gian là overhead cố định mỗi lần gọi model() (kernel launch qua
    từng layer), không phải FLOPs của bản thân phép tính. Gộp k rollout
    vào 1 batch khấu hao chi phí cố định đó, giảm SỐ LẦN GỌI model() thay
    vì chỉ giảm số phép tính mỗi lần gọi (đó là điều KV-cache một mình nó
    không giải quyết được khi batch=1).

    LƯU Ý — ĐƠN GIẢN HOÁ có chủ đích cho quy mô demo:
    Không early-stop riêng từng sequence khi nó ra eos giữa chừng (muốn làm
    đúng cần attention mask cho phần "đã xong" của từng hàng trong batch —
    phức tạp không cần thiết ở đây). Mọi sequence trong batch đều chạy đủ
    max_new bước; sequence nào đã gặp eos thì các token sinh SAU đó bị coi
    là rác nhưng vô hại — caller (extract_answer trong label_prm_data.py)
    chỉ tìm "Answer: X" xuất hiện ở bất kỳ đâu trong text, không bị ảnh
    hưởng bởi rác phía sau.

    Returns: list[str] độ dài batch_size — mỗi phần tử là phần TEXT MỚI
             SINH (không gồm prompt), đã decode.
    """
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if add_bos:
        prompt_ids = [tokenizer.bos_id] + prompt_ids

    active = prompt_ids[-max_seq:] if len(prompt_ids) > max_seq else prompt_ids

    rope_buffer_limit = max_seq * 2
    if len(active) + max_new > rope_buffer_limit:
        max_new = max(rope_buffer_limit - len(active), 0)

    # Replicate CÙNG 1 prompt thành batch_size hàng giống hệt nhau
    ids = torch.tensor([active], dtype=torch.long, device=device).repeat(batch_size, 1)
    T0  = ids.size(1)

    logits, kv_cache = model(
        ids, attn_mask=causal_mask(T0, device),
        use_cache=True, start_pos=0,
    )
    cur_pos = T0

    generated = [[] for _ in range(batch_size)]
    finished  = [False] * batch_size

    for step in range(max_new):
        next_toks = _sample_next_ids(logits[:, -1, :], temperature, top_k, top_p)  # (batch_size,)

        for b in range(batch_size):
            if not finished[b]:
                tok = next_toks[b].item()
                generated[b].append(tok)
                if tok == tokenizer.eos_id:
                    finished[b] = True

        if all(finished) or step == max_new - 1:
            break

        next_ids = next_toks.unsqueeze(1)   # (batch_size, 1)
        logits, kv_cache = model(
            next_ids, attn_mask=None,
            kv_cache=kv_cache, start_pos=cur_pos, use_cache=True,
        )
        cur_pos += 1

    return [tokenizer.decode(g) for g in generated]


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