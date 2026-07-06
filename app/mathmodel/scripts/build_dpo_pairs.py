"""
scripts/build_dpo_pairs.py — Stage 3a: Build preference pairs từ nhãn PRM
=============================================================================
Đọc output Stage 2 (label_prm_data.py), với mỗi bài toán có >=2 lời giải:
    - Sắp xếp theo prm_score giảm dần
    - chosen = lời giải điểm cao nhất, rejected = lời giải điểm thấp nhất
    - CHỈ giữ cặp nếu (score_chosen - score_rejected) >= min_gap
      (tránh cặp gần bằng nhau, nhiễu label — theo đúng spec Stage 3)

LƯU Ý QUAN TRỌNG: không dùng điểm TUYỆT ĐỐI (vd "phải có >50% step correct")
để quyết định pair — với model 100M mới SFT, hầu hết lời giải có prm_score
thấp (thống kê Stage 2 thực tế: chỉ 4.6% final answer đúng, 93% step bị
gán incorrect). Điều DPO cần là KHÁC BIỆT TƯƠNG ĐỐI giữa các lời giải CÙNG
một bài toán — 2 lời giải đều sai đáp số cuối vẫn có thể tạo pair hợp lệ
nếu 1 cái sai muộn hơn/ít bước sai hơn cái kia.

Output: data/gsm8k_dpo/pairs.jsonl — mỗi dòng:
    {"prompt": "...", "chosen": "...", "rejected": "...",
     "chosen_score": 0.0, "rejected_score": 0.0, "ground_truth_answer": "..."}

Chạy:
    python scripts/build_dpo_pairs.py \
        --input data/gsm8k_prm/labeled.jsonl \
        --output data/gsm8k_dpo/pairs.jsonl \
        --min-gap 0.15
"""

import argparse
import json
import os


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved → {path}")


def build_completion(steps: list[str], final_answer: str) -> str:
    """
    Build lại completion SẠCH từ dữ liệu đã parse (steps, final_answer) —
    KHÔNG dùng text gốc model sinh ra. Lý do: generate_batch() (Stage 2)
    không early-stop từng sequence riêng khi gặp eos giữa batch, nên text
    gốc có thể dính token rác sau "Answer: X". Build lại từ structured data
    đảm bảo chosen/rejected khớp CHÍNH XÁC format đã học ở SFT.
    """
    step_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
    return f"{step_text}\nAnswer: {final_answer}"


def build_pairs(rows: list[dict], min_gap: float = 0.15) -> list[dict]:
    pairs = []
    n_no_variance = 0   # bài chỉ có 0-1 lời giải hợp lệ (parse hỏng hết/gần hết)
    n_too_close   = 0   # có ≥2 lời giải nhưng gap điểm quá nhỏ, bỏ để tránh nhiễu

    for row in rows:
        solutions = row.get("solutions", [])
        if len(solutions) < 2:
            n_no_variance += 1
            continue

        ranked = sorted(solutions, key=lambda s: s["prm_score"], reverse=True)
        best, worst = ranked[0], ranked[-1]
        gap = best["prm_score"] - worst["prm_score"]

        if gap < min_gap:
            n_too_close += 1
            continue

        prompt = f"Problem: {row['problem']}\nSolution:\n"
        pairs.append({
            "prompt"              : prompt,
            "chosen"              : build_completion(best["steps"],  best["final_answer"]),
            "rejected"            : build_completion(worst["steps"], worst["final_answer"]),
            "chosen_score"        : best["prm_score"],
            "rejected_score"      : worst["prm_score"],
            "ground_truth_answer" : row["ground_truth_answer"],
        })

    print(f"\n  Tổng bài toán           : {len(rows)}")
    print(f"  Bỏ (<2 lời giải hợp lệ) : {n_no_variance}")
    print(f"  Bỏ (gap < {min_gap})      : {n_too_close}")
    print(f"  Pairs tạo được          : {len(pairs)}")

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Stage 3a — Build preference pairs từ nhãn PRM")
    parser.add_argument("--input",   type=str, default="data/gsm8k_prm/labeled.jsonl")
    parser.add_argument("--output",  type=str, default="data/gsm8k_dpo/pairs.jsonl")
    parser.add_argument("--min-gap", type=float, default=0.15,
                         help="Khoảng cách prm_score tối thiểu giữa chosen/rejected để giữ cặp")
    args = parser.parse_args()

    rows  = load_jsonl(args.input)
    pairs = build_pairs(rows, min_gap=args.min_gap)

    if not pairs:
        print("\n  ⚠ KHÔNG tạo được pair nào — thử giảm --min-gap, hoặc gán nhãn thêm bài "
              "(n_problems ở Stage 2 hiện còn nhỏ).")
        return

    save_jsonl(pairs, args.output)

    gaps = [p["chosen_score"] - p["rejected_score"] for p in pairs]
    print(f"\n  Gap trung bình : {sum(gaps)/len(gaps):.3f}")
    print(f"  Gap min / max  : {min(gaps):.3f} / {max(gaps):.3f}")

    example = pairs[0]
    print(f"\n{'='*60}\n  Ví dụ 1 pair:\n{'='*60}")
    print(f"[PROMPT]\n{example['prompt']}")
    print(f"[CHOSEN]   (score={example['chosen_score']:.2f})\n{example['chosen']}")
    print(f"\n[REJECTED] (score={example['rejected_score']:.2f})\n{example['rejected']}")


if __name__ == "__main__":
    main()