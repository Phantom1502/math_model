"""
scripts/prepare_sft_data.py — Chuẩn hoá GSM8K sang format SFT
=================================================================
Convert GSM8K (HuggingFace: openai/gsm8k, config "main") sang format
đơn giản, tiếng Anh, tối ưu cho model max_seq=512:

    Problem: {question}
    Solution:
    Step 1: {step 1}
    Step 2: {step 2}
    ...
    Answer: {number}

Không dùng tag XML (<problem>, <solution>) — với model nhỏ, tag chỉ tốn
thêm token mà không cần thiết. Ranh giới đã đủ rõ bằng "\n" + từ khóa
cố định (Problem: / Solution: / Step n: / Answer:).

Output: 2 file JSONL — train.jsonl, test.jsonl. Mỗi dòng:
    {"prompt": "...", "completion": "...", "answer": "72"}

    prompt     : phần KHÔNG tính loss khi SFT (mask bằng -100)
    completion : phần TÍNH loss khi SFT
    answer     : đáp số đã normalize (dùng để so khớp ground truth ở Stage 2)

Chạy:
    python scripts/prepare_sft_data.py --output-dir data/gsm8k_sft
"""

import argparse
import json
import os
import re
import sys

from datasets import load_dataset

# Cho phép import tokenizer.py từ app/model/ khi chạy script này từ app/model/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tokenizer import VietnameseTokenizer


# Bỏ calculator annotation dạng <<48/2=24>> — trace nội bộ GSM8K, không cần
_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def _clean_line(line: str) -> str:
    """Bỏ annotation <<...>> và khoảng trắng thừa."""
    return _CALC_ANNOTATION_RE.sub("", line).strip()


def parse_gsm8k_sample(question: str, raw_answer: str) -> dict:
    """
    Parse 1 sample GSM8K thô sang format chuẩn hoá.

    raw_answer gốc có dạng:
        Natalia sold clips to 48 of her friends in April...
        <<48/2=24>>
        Natalia sold 48+24 = <<48+24=72>>72 clips...
        #### 72

    Returns: {"prompt", "completion", "answer", "n_steps"}
    """
    lines = [l for l in raw_answer.strip().split("\n") if l.strip()]

    # Dòng cuối luôn là "#### <answer>"
    final_line = lines[-1]
    assert final_line.startswith("####"), f"Không tìm thấy '####' trong: {raw_answer!r}"
    answer = final_line.replace("####", "").strip()

    # Các dòng còn lại → steps (bỏ annotation, bỏ dòng rỗng sau khi clean)
    step_lines = [_clean_line(l) for l in lines[:-1]]
    step_lines = [l for l in step_lines if l]

    prompt     = f"Problem: {question.strip()}\nSolution:\n"
    step_text  = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(step_lines))
    completion = f"{step_text}\nAnswer: {answer}"

    return {
        "prompt"    : prompt,
        "completion": completion,
        "answer"    : answer,
        "n_steps"   : len(step_lines),
    }


def convert_split(split_name: str) -> list[dict]:
    print(f"\nLoading GSM8K split='{split_name}'...")
    ds = load_dataset("openai/gsm8k", "main", split=split_name)

    samples = []
    n_skipped = 0
    for row in ds:
        try:
            sample = parse_gsm8k_sample(row["question"], row["answer"])
            samples.append(sample)
        except (AssertionError, IndexError):
            n_skipped += 1

    print(f"  ✓ {len(samples)} samples parsed OK  |  {n_skipped} skipped (lỗi parse)")
    return samples


def print_stats(samples: list[dict], name: str):
    n_steps = [s["n_steps"] for s in samples]
    lens    = [len(s["prompt"]) + len(s["completion"]) for s in samples]

    print(f"\n  [{name}] n={len(samples)}")
    print(f"    steps/sample : min={min(n_steps)}  max={max(n_steps)}  "
          f"avg={sum(n_steps)/len(n_steps):.1f}")
    print(f"    chars/sample : min={min(lens)}  max={max(lens)}  "
          f"avg={sum(lens)/len(lens):.0f}  "
          f"(ước lượng thô ~{max(lens)//3} token cho sample dài nhất)")


def filter_by_length(
    samples   : list[dict],
    tokenizer : VietnameseTokenizer,
    max_seq   : int,
    name      : str,
    margin    : int = 2,   # chừa chỗ cho bos/eos khi add_special_tokens=True lúc train
) -> list[dict]:
    """
    Tokenize thật bằng tokenizer của project, bỏ sample nào
    len(prompt_ids) + len(completion_ids) + margin > max_seq.

    Dùng encode_batch để tránh gọi tokenizer từng sample một (chậm với 7K+ mẫu).
    """
    prompts     = [s["prompt"] for s in samples]
    completions = [s["completion"] for s in samples]

    prompt_ids_list     = tokenizer.encode_batch(prompts,     add_special_tokens=False)
    completion_ids_list = tokenizer.encode_batch(completions, add_special_tokens=False)

    kept, dropped_lens = [], []
    for sample, p_ids, c_ids in zip(samples, prompt_ids_list, completion_ids_list):
        total = len(p_ids) + len(c_ids) + margin
        if total <= max_seq:
            sample["n_tokens"] = total
            kept.append(sample)
        else:
            dropped_lens.append(total)

    n_dropped = len(samples) - len(kept)
    print(f"\n  [{name}] Lọc theo max_seq={max_seq} (margin={margin}):")
    print(f"    giữ lại : {len(kept)}/{len(samples)}")
    if n_dropped:
        print(f"    bỏ      : {n_dropped}  "
              f"(token dài nhất bị bỏ: {max(dropped_lens)}, "
              f"ngắn nhất bị bỏ: {min(dropped_lens)})")
    else:
        print(f"    bỏ      : 0")

    return kept


def save_jsonl(samples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn hoá GSM8K sang format SFT step-by-step")
    parser.add_argument("--output-dir",     type=str, default="data/gsm8k_sft")
    parser.add_argument("--tokenizer-path", type=str, default="custom_tokenizer",
                         help="Path tới tokenizer đã train (từ scripts/train_tokenizer.py)")
    parser.add_argument("--max-seq",        type=int, default=512,
                         help="max_seq của model — sample vượt ngưỡng này sẽ bị loại")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_samples = convert_split("train")
    test_samples  = convert_split("test")

    print_stats(train_samples, "train")
    print_stats(test_samples, "test")

    print(f"\nLoading tokenizer từ '{args.tokenizer_path}' để lọc theo max_seq={args.max_seq}...")
    tokenizer = VietnameseTokenizer(pretrained_name=args.tokenizer_path)

    train_samples = filter_by_length(train_samples, tokenizer, args.max_seq, "train")
    test_samples  = filter_by_length(test_samples,  tokenizer, args.max_seq, "test")

    save_jsonl(train_samples, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(test_samples,  os.path.join(args.output_dir, "test.jsonl"))

    print(f"\n{'='*60}")
    print(f"  DONE — ví dụ 1 sample train:")
    print(f"{'='*60}")
    example = train_samples[0]
    print(example["prompt"] + example["completion"])
    print(f"\n  answer   : {example['answer']}")
    print(f"  n_steps  : {example['n_steps']}")


if __name__ == "__main__":
    main()