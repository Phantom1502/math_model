"""
scripts/train_tokenizer.py — Train BPE tokenizer từ đầu cho MemoryLM
=======================================================================
Train ByteLevel BPE 16k vocab từ Wikipedia tiếng Việt + VTSNLP curated,
lưu ra thư mục local để dùng thay PhoBERT tokenizer.

Thiết kế:
    - ByteLevel BPE  : không cần lo unicode/dấu thanh, không bao giờ có <unk>
    - Raw text       : không dùng VnCoreNLP word-segmentation
    - Vocab 16k      : embedding = 16k × 512 = ~8M params (thay vì 68k × 512 = ~35M)
    - Price vocab    : 4098 tokens riêng biệt (regex, nằm ngoài BPE — giữ nguyên)
    - Tổng           : ~20k tokens → phù hợp model nhỏ ~70M params

Chạy trên Colab (1 lần duy nhất):
    !python scripts/train_tokenizer.py

Hoặc tùy chỉnh:
    !python scripts/train_tokenizer.py \
        --vocab-size 16000 \
        --wiki-samples 500000 \
        --vtsnlp-samples 500000 \
        --output-dir custom_tokenizer

Sau khi chạy xong, cập nhật config:
    cfg.tokenizer.pretrained_name = "custom_tokenizer"
    # (hoặc đặt TOKENIZER_PATH trong môi trường)

────────────────────────────────────────────────────────────────────────────
LƯU Ý QUAN TRỌNG:

Tokenizer mới KHÔNG tương thích với checkpoint cũ (PhoBERT vocab).
Phải train model từ đầu sau khi đổi tokenizer.
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import time

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


# ══════════════════════════════════════════════════════════════════════════
# Special tokens — giống GPT-2 style, tương thích với VietnameseTokenizer
# ══════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = {
    "bos_token" : "<s>",
    "eos_token" : "</s>",
    "unk_token" : "<unk>",
    "pad_token" : "<pad>",
    "mask_token": "<mask>",
}

# Thứ tự quan trọng — ID sẽ được gán theo thứ tự này (0, 1, 2, ...)
SPECIAL_TOKENS_LIST = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"]


# ══════════════════════════════════════════════════════════════════════════
# Text iterator — stream từ HuggingFace, không load vào RAM
# ══════════════════════════════════════════════════════════════════════════

def text_iterator(
    wiki_samples  : int = 500_000,
    vtsnlp_samples: int = 500_000,
    min_len       : int = 100,
    log_every     : int = 100_000,
):
    """
    Generator yield text từ 2 nguồn:
        1. wikimedia/wikipedia 20231101.vi
        2. VTSNLP/vietnamese_curated_dataset

    Dùng streaming — không cần load toàn bộ vào RAM.
    Tổng ~1M câu là đủ để BPE học tốt vocab tiếng Việt.

    min_len: bỏ qua text quá ngắn (tiêu đề stub, redirect...)
    """
    total = 0

    # ── Nguồn 1: Wikipedia ─────────────────────────────────────────────
    print(f"\n[1/2] Loading Wikipedia (20231101.vi), lấy tối đa {wiki_samples:,} samples...")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.vi",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    wiki_count = 0
    for sample in wiki:
        if wiki_count >= wiki_samples:
            break
        text = sample.get("text", "").strip()
        if len(text) < min_len:
            continue
        yield text
        wiki_count += 1
        total += 1
        if total % log_every == 0:
            print(f"  ... {total:,} texts yielded")

    print(f"  ✓ Wikipedia: {wiki_count:,} samples")

    # ── Nguồn 2: VTSNLP ────────────────────────────────────────────────
    print(f"\n[2/2] Loading VTSNLP/vietnamese_curated_dataset, lấy tối đa {vtsnlp_samples:,} samples...")
    vtsnlp = load_dataset(
        "VTSNLP/vietnamese_curated_dataset",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    vtsnlp_count = 0
    for sample in vtsnlp:
        if vtsnlp_count >= vtsnlp_samples:
            break
        text = sample.get("text", "").strip()
        if len(text) < min_len:
            continue
        yield text
        vtsnlp_count += 1
        total += 1
        if total % log_every == 0:
            print(f"  ... {total:,} texts yielded")

    print(f"  ✓ VTSNLP: {vtsnlp_count:,} samples")
    print(f"\n  ✓ Tổng: {total:,} texts dùng để train tokenizer")


# ══════════════════════════════════════════════════════════════════════════
# Train
# ══════════════════════════════════════════════════════════════════════════

def train_tokenizer(
    output_dir    : str,
    vocab_size    : int = 16_000,
    wiki_samples  : int = 500_000,
    vtsnlp_samples: int = 500_000,
    min_len       : int = 100,
) -> PreTrainedTokenizerFast:
    """
    Train ByteLevel BPE tokenizer và lưu ra output_dir.

    ByteLevel BPE:
        - Pre-tokenize ở cấp byte → không bao giờ có <unk>, kể cả ký tự lạ
        - Học merge tốt cho tiếng Việt với dấu thanh (multi-byte UTF-8)
        - Không cần quét initial_alphabet như Whitespace BPE

    Returns: tokenizer đã wrap thành PreTrainedTokenizerFast (HF-compatible).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Khởi tạo BPE core ───────────────────────────────────────────────
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # ByteLevel pre-tokenizer — xử lý ở cấp byte, add_prefix_space=False
    # để không thêm khoảng trắng đầu câu (khác GPT-2 gốc add_prefix_space=True)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Post-processor và decoder phải khớp với pre-tokenizer
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder        = ByteLevelDecoder()

    # ── Trainer config ───────────────────────────────────────────────────
    trainer = BpeTrainer(
        vocab_size        = vocab_size,
        special_tokens    = SPECIAL_TOKENS_LIST,
        min_frequency     = 2,          # bỏ qua token xuất hiện < 2 lần
        show_progress     = True,
        # Không cần initial_alphabet vì ByteLevel tự handle mọi byte
    )

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Train BPE tokenizer")
    print(f"  vocab_size : {vocab_size:,}")
    print(f"  output_dir : {output_dir}")
    print(f"{'═'*60}")

    t0 = time.time()
    tokenizer.train_from_iterator(
        text_iterator(wiki_samples, vtsnlp_samples, min_len),
        trainer=trainer,
        length=wiki_samples + vtsnlp_samples,   # hint cho progress bar
    )
    elapsed = time.time() - t0
    print(f"\n  ✓ Train xong trong {elapsed/60:.1f} phút")
    print(f"  Vocab size thực tế: {tokenizer.get_vocab_size():,}")

    # ── Wrap thành PreTrainedTokenizerFast ──────────────────────────────
    # PreTrainedTokenizerFast tương thích với AutoTokenizer.from_pretrained()
    # — VietnameseTokenizer dùng AutoTokenizer để load, không cần sửa gì thêm
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        **SPECIAL_TOKENS,
    )

    # ── Verify special token IDs ─────────────────────────────────────────
    print(f"\n  Special token IDs:")
    for name, tok in SPECIAL_TOKENS.items():
        tok_id = fast_tokenizer.convert_tokens_to_ids(tok)
        print(f"    {name:<12} '{tok}' → id={tok_id}")

    # ── Lưu ra disk ─────────────────────────────────────────────────────
    fast_tokenizer.save_pretrained(output_dir)
    print(f"\n  ✓ Đã lưu tokenizer → {output_dir}/")

    return fast_tokenizer


# ══════════════════════════════════════════════════════════════════════════
# Sanity check sau khi train
# ══════════════════════════════════════════════════════════════════════════

def sanity_check(tokenizer_dir: str):
    """
    Load lại tokenizer từ disk và kiểm tra một số trường hợp cơ bản.
    Đảm bảo tokenizer có thể load bình thường qua AutoTokenizer
    (đúng cách VietnameseTokenizer sẽ load trong production).
    """
    from transformers import AutoTokenizer

    print(f"\n{'─'*60}")
    print(f"  Sanity check — load lại từ {tokenizer_dir}")
    print(f"{'─'*60}")

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)

    test_cases = [
        "Trí tuệ nhân tạo đang thay đổi thế giới.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Albert Einstein là nhà vật lý nổi tiếng.",
        "RSI MACD đường_MA kháng_cự phá_vỡ",       # trading terms
        "Năm 2024, GDP tăng trưởng 6.5%.",
        "con chó mèo nhà gà vịt",
    ]

    all_ok = True
    for text in test_cases:
        ids     = tok.encode(text, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        # ByteLevel decode có thể thêm khoảng trắng đầu — strip để so sánh
        match = decoded.strip() == text.strip()
        status = "✓" if match else "✗"
        if not match:
            all_ok = False
        n_tokens = len(ids)
        print(f"  {status} [{n_tokens:>3} tok] {text[:50]}")
        if not match:
            print(f"       decoded: {decoded!r}")

    # Kiểm tra không có <unk> trong roundtrip
    combined = " ".join(test_cases)
    ids_all  = tok.encode(combined, add_special_tokens=False)
    unk_id   = tok.convert_tokens_to_ids("<unk>")
    n_unk    = ids_all.count(unk_id)
    print(f"\n  <unk> count trong toàn bộ test: {n_unk}  {'✓' if n_unk == 0 else '✗ WARN'}")

    print(f"\n  Vocab size (base BPE): {tok.vocab_size}")
    print(f"  {'✓ Sanity check PASSED' if all_ok else '✗ Một số test FAILED — kiểm tra lại'}")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train BPE tokenizer 16k cho MemoryLM"
    )
    p.add_argument("--vocab-size",      type=int, default=16_000,
                   help="Vocab size BPE (không tính price token, default 16000)")
    p.add_argument("--wiki-samples",    type=int, default=500_000,
                   help="Số sample lấy từ Wikipedia (default 500000)")
    p.add_argument("--vtsnlp-samples",  type=int, default=500_000,
                   help="Số sample lấy từ VTSNLP (default 500000)")
    p.add_argument("--min-len",         type=int, default=100,
                   help="Bỏ qua text ngắn hơn N ký tự (default 100)")
    p.add_argument("--output-dir",      type=str, default="custom_tokenizer",
                   help="Thư mục lưu tokenizer (default custom_tokenizer)")
    p.add_argument("--skip-sanity",     action="store_true",
                   help="Bỏ qua sanity check sau khi train")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"{'═'*60}")
    print(f"  MemoryLM — Train BPE Tokenizer")
    print(f"{'═'*60}")
    print(f"  vocab_size     : {args.vocab_size:,}  (BPE base)")
    print(f"  price vocab    : 4098               (regex, riêng biệt)")
    print(f"  total vocab    : ~{args.vocab_size + 4098:,}")
    print(f"  wiki_samples   : {args.wiki_samples:,}")
    print(f"  vtsnlp_samples : {args.vtsnlp_samples:,}")
    print(f"  output_dir     : {args.output_dir}")
    print()

    train_tokenizer(
        output_dir     = args.output_dir,
        vocab_size     = args.vocab_size,
        wiki_samples   = args.wiki_samples,
        vtsnlp_samples = args.vtsnlp_samples,
        min_len        = args.min_len,
    )

    if not args.skip_sanity:
        sanity_check(args.output_dir)

    print(f"\n{'═'*60}")
    print(f"  DONE — Cập nhật config để dùng tokenizer mới:")
    print(f"{'─'*60}")
    print(f"  cfg.tokenizer.pretrained_name = '{args.output_dir}'")
    print(f"  cfg.tokenizer.use_fast        = True   # Fast tokenizer OK")
    print(f"  # Sau đó train model từ đầu (checkpoint cũ không tương thích)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()