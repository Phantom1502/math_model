"""
scripts/add_custom_tokens.py — Mở rộng vocab PhoBERT với token domain-specific
=====================================================================================
Mục đích: thêm ~1000 token trading (thuật ngữ chuyên biệt) vào tokenizer,
LƯU THÀNH BẢN RIÊNG (không đụng vào "vinai/phobert-base" gốc trên HuggingFace
cache), để project gốc không bị ảnh hưởng nếu không dùng custom tokenizer.

────────────────────────────────────────────────────────────────────────────
TẠI SAO KHÔNG ĐỘNG VÀO TOKENIZER GỐC:

PhoBERT tokenizer gốc được dùng chung bởi mọi nơi gọi load_tokenizer() mặc
định. Nếu sửa trực tiếp, MỌI checkpoint cũ (train trước khi thêm token) sẽ
LỆCH vocab — token cũ có thể bị đẩy ID khác, embedding học được trở nên vô
nghĩa khi load lại. Lưu bản riêng vào thư mục cục bộ (custom_tokenizer/)
đảm bảo:
    - Tokenizer gốc không đổi, các checkpoint cũ vẫn load tokenizer gốc bình
      thường nếu cần.
    - Bản custom là một artifact ĐỘC LẬP, version riêng, dùng cho round
      train MỚI (train lại từ đầu — đúng theo yêu cầu của bạn).

────────────────────────────────────────────────────────────────────────────
CÁCH DÙNG:

    python scripts/add_custom_tokens.py \
        --new-tokens-file my_trading_tokens.txt \
        --output-dir custom_tokenizer

File my_trading_tokens.txt: mỗi dòng một token, ví dụ:

    đường_MA
    kháng_cự
    phá_vỡ
    EURUSD
    RSI
    MACD
    nến_doji
    ...

Sau khi chạy xong, thư mục `custom_tokenizer/` chứa đầy đủ file tokenizer
(vocab.txt, tokenizer_config.json, ...) — load lại y hệt cách load một
tokenizer HuggingFace bình thường, qua AutoTokenizer.from_pretrained(path).
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
from transformers import AutoTokenizer


def add_custom_tokens(
    base_tokenizer_name : str,
    new_tokens           : list[str],
    output_dir            : str,
) -> AutoTokenizer:
    """
    Load tokenizer gốc, thêm token mới, lưu thành bản riêng.

    Returns: tokenizer đã mở rộng (để caller có thể dùng ngay nếu cần).
    """
    print(f"Loading base tokenizer: {base_tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

    vocab_size_before = len(tokenizer)
    print(f"Vocab size TRƯỚC khi thêm: {vocab_size_before}")

    # Lọc bỏ token đã tồn tại sẵn — add_tokens() sẽ bỏ qua chúng tự động,
    # nhưng lọc trước để log rõ ràng số token THỰC SỰ mới.
    existing_vocab = set(tokenizer.get_vocab().keys())
    truly_new = [t for t in new_tokens if t not in existing_vocab]
    already_exist = [t for t in new_tokens if t in existing_vocab]

    if already_exist:
        print(f"\n{len(already_exist)} token đã tồn tại sẵn trong vocab gốc, bỏ qua:")
        print(f"  {already_exist[:10]}{'...' if len(already_exist) > 10 else ''}")

    print(f"\nThêm {len(truly_new)} token mới...")
    num_added = tokenizer.add_tokens(truly_new)

    vocab_size_after = len(tokenizer)
    print(f"Vocab size SAU khi thêm: {vocab_size_after}")
    print(f"Số token thực sự được thêm: {num_added}")

    assert vocab_size_after == vocab_size_before + num_added, \
        "Số lượng vocab không khớp — kiểm tra lại danh sách token đầu vào"

    # ── Lưu thành bản tokenizer ĐỘC LẬP ──────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✓ Đã lưu custom tokenizer vào: {output_dir}/")
    print(f"  (Tokenizer gốc '{base_tokenizer_name}' KHÔNG bị ảnh hưởng)")

    return tokenizer


def load_tokens_from_file(filepath: str) -> list[str]:
    """Đọc danh sách token từ file text, mỗi dòng một token."""
    with open(filepath, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Thêm custom token (trading) vào PhoBERT tokenizer"
    )
    parser.add_argument(
        "--base-tokenizer", type=str, default="vinai/phobert-base",
        help="Tokenizer gốc dùng làm nền",
    )
    parser.add_argument(
        "--new-tokens-file", type=str, required=True,
        help="File text chứa danh sách token mới, mỗi dòng một token",
    )
    parser.add_argument(
        "--output-dir", type=str, default="custom_tokenizer",
        help="Thư mục lưu tokenizer đã mở rộng",
    )
    args = parser.parse_args()

    new_tokens = load_tokens_from_file(args.new_tokens_file)
    print(f"Đọc {len(new_tokens)} token từ {args.new_tokens_file}\n")

    add_custom_tokens(
        base_tokenizer_name = args.base_tokenizer,
        new_tokens           = new_tokens,
        output_dir            = args.output_dir,
    )


if __name__ == "__main__":
    main()
