"""
app/llama/model.py — Build HF LlamaForCausalLM chuẩn từ Config
====================================================================
Thay app/memlm/model/lm.py (kiến trúc tự viết: RMSNorm/RoPE/SwiGLU/DepthRouter
tự implement) bằng transformers.LlamaForCausalLM gốc.

Được gì:
    - GQA (num_key_value_heads) built-in
    - KV-cache generate() built-in (nhanh hơn nhiều so với vòng lặp forward
      lại toàn bộ sequence mỗi step như app/memlm/generate.py hiện tại)
    - Tương thích thẳng Trainer/TRL, không cần viết lại DPO/listwise loss tay
    - save_pretrained()/from_pretrained() chuẩn — checkpoint tự chứa config

Mất gì (đã đánh đổi có chủ đích, xem đánh giá trước đó):
    - SequenceRouter/DepthRouter (skip/run routing theo sequence) không còn
      áp dụng được trực tiếp trên LlamaDecoderLayer gốc. Nếu sau này vẫn
      muốn dùng, cần subclass LlamaModel và patch từng layer — ngoài phạm vi
      nhánh này.
"""

from transformers import LlamaForCausalLM


def build_model(cfg, tokenizer=None) -> LlamaForCausalLM:
    """
    Build model mới từ cfg.llama (LlamaConfig, khởi tạo weight random).

    Nếu truyền tokenizer, TỰ ĐỘNG resize embedding đúng bằng len(tokenizer)
    — BẮT BUỘC phải làm sau khi add_tokens() price vocab, nếu không sẽ
    IndexError khi model gặp token id vượt vocab_size khai trong config gốc
    (giống lỗi "size mismatch"/"Missing key(s)" đã note trong README nhánh cũ).
    """
    model = LlamaForCausalLM(cfg.llama)

    if tokenizer is not None:
        n_tok = len(tokenizer)
        cur   = model.get_input_embeddings().weight.shape[0]
        if cur != n_tok:
            model.resize_token_embeddings(n_tok)
            cfg.llama.vocab_size = n_tok
            print(f"  Resized embeddings: {cur:,} → {n_tok:,}")

    return model


def num_params(model, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from config import get_small_llama_config, Config

    cfg = Config(llama=get_small_llama_config())
    model = build_model(cfg)
    print(f"Params: {num_params(model):,}")

    import torch
    ids = torch.randint(0, cfg.llama.vocab_size, (2, 16))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    print(f"Logits shape: {tuple(out.logits.shape)}")