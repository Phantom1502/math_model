"""
scripts/label_prm_data.py — Stage 2: Gán nhãn step-level bằng Math-Shepherd
================================================================================
Với mỗi bài toán:
    1. Model (đã SFT) tự sinh N lời giải độc lập (sampling, temperature > 0).
    2. Với mỗi lời giải, tách thành các step theo "Step n: ...".
    3. Với mỗi step i, cắt lời giải tại đó, cho model SINH TIẾP k lần
       (Monte Carlo rollout) → so khớp "Answer: X" cuối cùng với đáp số đúng.
    4. Tỷ lệ đúng trong k lần = soft label cho step i:
           ratio >= 0.6  → "correct"
           ratio <= 0.2  → "incorrect"
           còn lại       → "uncertain"

CẢNH BÁO COMPUTE: generate() trong generate.py KHÔNG có KV-cache (forward
lại toàn bộ sequence mỗi token mới) và chỉ chạy batch_size=1. Số lần gọi
generate() cho 1 bài toán ≈ n_solutions * (1 + avg_steps * k) — tăng rất
nhanh theo n_problems. MẶC ĐỊNH cố tình để nhỏ (50 bài) để chạy thử trước,
tăng dần bằng --n-problems khi đã chắc pipeline chạy đúng.

Input : data/gsm8k_sft/train.jsonl (output của prepare_sft_data.py —
        dùng field "prompt" và "answer", KHÔNG dùng "completion" vì ta cần
        lời giải MODEL TỰ SINH, không phải ground-truth completion).
Output: data/gsm8k_prm/labeled.jsonl — format khớp spec Stage 2:
    {
      "problem": "...",
      "ground_truth_answer": "...",
      "solutions": [
        {"steps": [...], "step_labels": [...], "step_ratios": [...],
         "prm_score": 0.0, "final_answer": "..."}
      ]
    }

Chạy:
    python scripts/label_prm_data.py \
        --checkpoint checkpoints_sft/sft_best.pt \
        --train-jsonl data/gsm8k_sft/train.jsonl \
        --n-problems 50 --n-solutions 4 --k 4
"""

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from generate import load_model_for_inference, generate, generate_batch


# ══════════════════════════════════════════════════════════════════════════
# Parse output model sinh ra (format Step n: / Answer: đã học ở SFT)
# ══════════════════════════════════════════════════════════════════════════

_STEP_RE   = re.compile(r"^Step\s*\d+\s*:\s*(.+)$")
_ANSWER_RE = re.compile(r"Answer\s*:\s*([\-\d][\d,\.\$]*)")
_PROMPT_RE = re.compile(r"^Problem:\s*(.*)\nSolution:\s*\n?$", re.DOTALL)


def normalize_answer(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    """Lấy giá trị sau 'Answer:' CUỐI CÙNG xuất hiện (phòng model lặp nhiều lần)."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    return normalize_answer(matches[-1])


def extract_steps(text: str) -> list[str]:
    steps = []
    for line in text.split("\n"):
        m = _STEP_RE.match(line.strip())
        if m:
            steps.append(m.group(1).strip())
    return steps


def extract_question(prompt: str) -> str:
    m = _PROMPT_RE.match(prompt.strip())
    return m.group(1).strip() if m else prompt.strip()


def build_partial_prompt(base_prompt: str, steps: list[str], upto: int) -> str:
    """base_prompt đã kết thúc bằng 'Solution:\\n' — nối thêm step 1..upto."""
    step_text = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps[:upto])) + "\n"
    return base_prompt + step_text


# ══════════════════════════════════════════════════════════════════════════
# Driver chính
# ══════════════════════════════════════════════════════════════════════════

def label_dataset(
    model, tokenizer, cfg,
    problems            : list[dict],
    n_solutions         : int   = 4,
    k                   : int   = 4,
    max_new_solution    : int   = 200,
    max_new_rollout     : int   = 100,
    temperature_solution: float = 0.8,
    temperature_rollout : float = 0.8,
) -> list[dict]:
    results = []
    t0 = time.time()

    for idx, row in enumerate(problems):
        prompt       = row["prompt"]
        ground_truth = normalize_answer(row["answer"])
        question     = extract_question(prompt)

        solutions_out = []
        full_texts = generate_batch(
            model, tokenizer, cfg, prompt,
            batch_size=n_solutions,
            max_new=max_new_solution, temperature=temperature_solution,
            top_k=50, top_p=0.95,
        )
        for full_text in full_texts:
            steps        = extract_steps(full_text)
            final_answer = extract_answer(full_text)

            if not steps or final_answer is None:
                continue  # sinh hỏng hoàn toàn (không parse được), bỏ qua lời giải này

            step_labels, step_ratios = [], []
            for i in range(len(steps)):
                if i == len(steps) - 1:
                    is_correct = (final_answer == ground_truth)
                    step_labels.append("correct" if is_correct else "incorrect")
                    step_ratios.append(1.0 if is_correct else 0.0)
                    continue

                partial = build_partial_prompt(prompt, steps, i + 1)
                conts = generate_batch(
                    model, tokenizer, cfg, partial,
                    batch_size=k,
                    max_new=max_new_rollout, temperature=temperature_rollout,
                    top_k=50, top_p=0.95,
                )
                n_correct = 0
                for cont in conts:
                    ans = extract_answer(cont)
                    if ans is not None and ans == ground_truth:
                        n_correct += 1
                ratio = n_correct / k
                step_ratios.append(ratio)
                if ratio >= 0.6:
                    step_labels.append("correct")
                elif ratio <= 0.2:
                    step_labels.append("incorrect")
                else:
                    step_labels.append("uncertain")

            score_map = {"correct": 1.0, "uncertain": 0.5, "incorrect": 0.0}
            prm_score = sum(score_map[l] for l in step_labels) / len(step_labels)

            solutions_out.append({
                "steps"       : steps,
                "step_labels" : step_labels,
                "step_ratios" : step_ratios,
                "prm_score"   : prm_score,
                "final_answer": final_answer,
            })

        results.append({
            "problem"            : question,
            "ground_truth_answer": ground_truth,
            "solutions"          : solutions_out,
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(problems):
            elapsed = time.time() - t0
            rate    = elapsed / (idx + 1)
            eta     = rate * (len(problems) - idx - 1)
            print(f"  [{idx+1}/{len(problems)}] "
                  f"{elapsed:.0f}s trôi qua | ~{rate:.1f}s/bài | ETA ~{eta/60:.1f} phút")

    return results


def save_jsonl(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 — Math-Shepherd step-level labeling")
    parser.add_argument("--checkpoint",  type=str, required=True,
                         help="Checkpoint SFT (vd checkpoints_sft/sft_best.pt)")
    parser.add_argument("--train-jsonl", type=str, default="data/gsm8k_sft/train.jsonl")
    parser.add_argument("--output",      type=str, default="data/gsm8k_prm/labeled.jsonl")
    parser.add_argument("--n-problems",  type=int, default=50,
                         help="Số bài toán dùng để gán nhãn — ĐỂ NHỎ khi chạy thử lần đầu")
    parser.add_argument("--n-solutions", type=int, default=4,
                         help="Số lời giải model tự sinh cho mỗi bài")
    parser.add_argument("--k",           type=int, default=4,
                         help="Số rollout Monte Carlo cho mỗi step trung gian")
    parser.add_argument("--max-new-solution", type=int, default=200)
    parser.add_argument("--max-new-rollout",  type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    print(f"Loading model từ {args.checkpoint} ...")
    model, tokenizer, cfg = load_model_for_inference(args.checkpoint)

    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        all_problems = [json.loads(l) for l in f if l.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(all_problems)
    problems = all_problems[:args.n_problems]

    # Ước lượng số lần GỌI model (không phải batch item) — sau khi batch hoá,
    # đây mới là con số quyết định tốc độ thật, không phải tổng continuation.
    avg_steps_guess = 4  # ước lượng thô để cảnh báo, không ảnh hưởng logic
    est_model_calls = args.n_problems * (1 + (avg_steps_guess - 1))
    print(f"\nƯớc lượng số lần gọi model (đã batch hoá theo n_solutions/k): ~{est_model_calls:,} "
          f"({args.n_problems} bài × (1 lần sinh N lời giải + ~{avg_steps_guess-1} step rollout))")
    print(f"Mỗi lần gọi xử lý batch={max(args.n_solutions, args.k)} sequence song song — "
          f"nhanh hơn nhiều so với gọi generate() tuần tự.\n")

    results = label_dataset(
        model, tokenizer, cfg, problems,
        n_solutions=args.n_solutions, k=args.k,
        max_new_solution=args.max_new_solution,
        max_new_rollout=args.max_new_rollout,
        temperature_solution=args.temperature,
        temperature_rollout=args.temperature,
    )

    save_jsonl(results, args.output)

    # Thống kê nhanh để biết tín hiệu có ý nghĩa không (spec: rủi ro "mọi rollout đều sai")
    n_sol_total   = sum(len(r["solutions"]) for r in results)
    n_correct_sol = sum(1 for r in results for s in r["solutions"] if s["final_answer"] == r["ground_truth_answer"])
    label_counts  = {"correct": 0, "incorrect": 0, "uncertain": 0}
    for r in results:
        for s in r["solutions"]:
            for l in s["step_labels"]:
                label_counts[l] += 1

    print(f"\n{'='*60}")
    print(f"  THỐNG KÊ NHANH")
    print(f"{'='*60}")
    print(f"  Số bài            : {len(results)}")
    print(f"  Tổng lời giải     : {n_sol_total} (bỏ qua lời giải parse hỏng)")
    print(f"  Lời giải đúng đáp số final : {n_correct_sol}/{n_sol_total} "
          f"({100*n_correct_sol/max(n_sol_total,1):.1f}%)")
    print(f"  Step labels       : {label_counts}")
    if label_counts["correct"] == 0 or label_counts["incorrect"] == 0:
        print(f"\n  ⚠ CẢNH BÁO: chỉ có 1 loại nhãn xuất hiện — không đủ tương phản để")
        print(f"    build preference pairs ở Stage 3. Cân nhắc fallback heuristic")
        print(f"    (kiểm tra phép tính bằng calculator) như đã ghi trong spec.")


if __name__ == "__main__":
    main()