"""
scripts/label_selfcorrect_data.py — Stage 2b (pipeline riêng): 3-tier data
cho self-correction training
=============================================================================
THAM CHIẾU: Self_correction_training.md, mục 2.6-2.7.

KHÁC với scripts/label_prm_data.py (Math-Shepherd, giữ nguyên làm tham khảo
— KHÔNG bị sửa bởi file này):
    - label_prm_data.py : Monte Carlo rollout (k lần/step) để ƯỚC LƯỢNG xem
                           1 step "có khả năng" đúng không → tốn ~50-230
                           lần gọi model/bài.
    - File này           : có GROUND TRUTH đầy đủ từng bước (từ GSM8K, đã
                           có sẵn trong train.jsonl) → DIFF trực tiếp theo
                           GIÁ TRỊ để tìm bước sai đầu tiên, KHÔNG cần đoán
                           bằng Monte Carlo → CHỈ 1 LẦN gọi model/bài (bản
                           thân rollout gốc — không cần generate thêm gì).

    → Rẻ hơn ~50-230 lần so với Math-Shepherd → chạy được TOÀN BỘ train
      set, không chỉ vài chục bài.

THIẾT KẾ tier_wrong_fixed (SỬA TẠI CHỖ, không generate tiếp):
    Thay vì để model tự generate phần tiếp theo sau marker "Wait" (cách này
    tốn thêm 1 lần gọi model VÀ chỉ có ~fix_retries cơ hội ra đúng đáp số),
    ta SỬA NGAY TẠI bước sai bằng câu đúng lấy từ ground truth, rồi nối
    tiếp CÁC BƯỚC CÒN LẠI CŨNG TỪ ground truth luôn — không cần generate gì
    thêm, đáp số CHẮC CHẮN đúng 100% vì toàn bộ phần sau điểm sửa đều là GT.

    Ví dụ (x=5, y=10, x+y=?), model sinh sai "thay x = 4":
        Step 1: thay x = 4. Wait, that's wrong — it should be: thay x = 5.
        Step 2: x + y = 5 + 10 = 15.
        Answer: 15

3 CẠM BẪY từ tài liệu — xử lý thế nào:
    1. So SỐ (state), không so text thô — extract_step_value() lấy số sau
       dấu '=' trong mỗi step, so giá trị chứ không so nguyên câu.
    2. Alignment bằng DP (Needleman-Wunsch), không theo index cứng — model
       có thể gộp/tách step khác granularity với ground truth.
    3. Tránh lệch văn phong khi ghép nối — ĐIỂM NỐI DUY NHẤT nằm NGAY TẠI
       bước sai, có "Wait, that's wrong" đánh dấu tường minh (chuyển giọng
       văn ở đây là CÓ CHỦ ĐÍCH, hợp lý). Từ đó trở đi dùng THUẦN văn bản
       GT (không xen kẽ model/GT nhiều lần như bản trước) — giảm số điểm
       nối lệch phong cách xuống còn đúng 1, và điểm đó được flag rõ ràng.
       Đánh đổi: prefix (model tự viết) và correction+phần sau (GT) có thể
       tham chiếu đại lượng hơi khác cách gọi — chấp nhận được ở quy mô demo.

Output mỗi bài (nếu rollout ban đầu SAI — bài đã đúng ngay thì bỏ qua,
không có gì để "sửa"): 3 tier hoàn chỉnh, sẵn sàng cho bước train sau
(Plackett-Luce / PRO listwise — viết ở script khác, KHÔNG nằm trong file
này) — mỗi dòng:
    {
      "prompt": "...",
      "ground_truth_answer": "...",
      "divergence_step_idx": 2,
      "tier_correct_straight": "Step 1: ...\\nAnswer: X",   # từ GT, miễn phí
      "tier_wrong_raw"       : "Step 1: ...\\nAnswer: Y",   # rollout gốc, miễn phí
      "tier_wrong_fixed"     : "Step 1: ...\\nWait...\\nAnswer: X" # từ GT, miễn phí, đáp số LUÔN đúng
    }

Chạy:
    python scripts/label_selfcorrect_data.py \
        --checkpoint checkpoints_sft/sft_best.pt \
        --train-jsonl data/gsm8k_sft/train.jsonl \
        --output data/gsm8k_selfcorrect/tiers.jsonl \
        --n-problems -1
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from generate import load_model_for_inference, generate


# ══════════════════════════════════════════════════════════════════════════
# Parse text — độc lập với label_prm_data.py (tách hẳn 2 pipeline)
# ══════════════════════════════════════════════════════════════════════════

_STEP_RE   = re.compile(r"^Step\s*\d+\s*:\s*(.+)$")
_ANSWER_RE = re.compile(r"Answer\s*:\s*([\-\d][\d,\.\$]*)")
_VALUE_RE  = re.compile(r"=\s*\$?(-?[\d,]+\.?\d*)")
_ANY_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def normalize_answer(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "").rstrip(".")


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text)
    return normalize_answer(matches[-1]) if matches else None


def extract_steps(text: str) -> list[str]:
    steps = []
    for line in text.split("\n"):
        m = _STEP_RE.match(line.strip())
        if m:
            steps.append(m.group(1).strip())
    return steps


def extract_step_value(step_text: str) -> str | None:
    """
    'State' mà 1 step tạo ra = giá trị số cuối cùng sau dấu '=' trong step đó
    (vd "16 eggs/day * 3 = 48 eggs/day" → "48"). Đây là điểm mấu chốt để so
    SỐ thay vì so TEXT — tránh cạm bẫy #1 trong tài liệu tham chiếu.

    Fallback nếu step không có '=' (câu diễn giải thuần, không phép tính):
    lấy số cuối cùng xuất hiện trong câu.
    """
    matches = _VALUE_RE.findall(step_text)
    if matches:
        return normalize_answer(matches[-1])
    fallback = _ANY_NUM_RE.findall(step_text)
    return normalize_answer(fallback[-1]) if fallback else None


def extract_question(prompt: str) -> str:
    m = re.match(r"^Problem:\s*(.*)\nSolution:\s*\n?$", prompt.strip(), re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


def build_step_text(steps: list[str], start_num: int = 1) -> str:
    return "\n".join(f"Step {start_num + i}: {s}" for i, s in enumerate(steps))


# ══════════════════════════════════════════════════════════════════════════
# Alignment — Needleman-Wunsch đơn giản trên chuỗi GIÁ TRỊ (không phải text)
# Tránh cạm bẫy #2: model có thể gộp/tách step khác granularity với GT.
# ══════════════════════════════════════════════════════════════════════════

def align_sequences(seq_a: list, seq_b: list) -> list[tuple]:
    """
    Trả về list các cặp (i, j) theo alignment tối ưu giữa seq_a và seq_b.
    i hoặc j có thể là None (gap — 1 bên có step mà bên kia không có step
    tương ứng, do khác granularity).
    """
    n, m = len(seq_a), len(seq_b)
    MATCH, MISMATCH, GAP = 1, -1, -1

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + (MATCH if seq_a[i - 1] == seq_b[j - 1] else MISMATCH)
            up   = dp[i - 1][j] + GAP
            left = dp[i][j - 1] + GAP
            dp[i][j] = max(diag, up, left)

    i, j = n, m
    alignment = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (MATCH if seq_a[i - 1] == seq_b[j - 1] else MISMATCH):
            alignment.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + GAP:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1
    alignment.reverse()
    return alignment


def find_divergence(model_values: list, gt_values: list) -> tuple[int, int]:
    """
    Trả về (divergence_idx, aligned_gt_idx) — index (theo model_steps) của
    step ĐẦU TIÊN có giá trị KHÁC với GT được alignment ánh xạ tới.

    Nếu mọi cặp matched đều bằng nhau (không tìm được mismatch rõ ràng theo
    alignment) nhưng final answer đã biết là SAI (điều kiện gọi hàm này) —
    coi lỗi nằm ở STEP CUỐI của model (phép tính/kết luận cuối sai dù các
    bước trung gian "trông" khớp GT).
    """
    alignment = align_sequences(model_values, gt_values)

    for i, j in alignment:
        if i is not None and j is not None and model_values[i] != gt_values[j]:
            return i, j

    last_i = len(model_values) - 1
    aligned_j = next((j for i, j in alignment if i == last_i and j is not None), len(gt_values) - 1)
    return last_i, aligned_j


# ══════════════════════════════════════════════════════════════════════════
# Sinh 3-tier cho 1 bài toán
# ══════════════════════════════════════════════════════════════════════════

def build_completion(steps: list[str], final_answer: str) -> str:
    return f"{build_step_text(steps)}\nAnswer: {final_answer}"


def process_one_problem(
    model, tokenizer, cfg,
    row              : dict,
    max_new_rollout  : int   = 200,
    temperature      : float = 0.8,
) -> dict:
    """
    Trả về dict record (xem docstring đầu file) nếu rollout SAI (có gì để
    sửa). Trả về {"status": ...} nếu:
        - rollout parse hỏng hoàn toàn
        - rollout ĐÃ ĐÚNG NGAY (không có bước sai để dạy phục hồi — GT đã
          đóng vai "đúng-thẳng" sẵn, không cần thêm gì cho bài này)

    CHỈ 1 LẦN gọi model (generate() cho rollout) — tier_wrong_fixed được
    xây HOÀN TOÀN từ template + ground truth, không generate thêm (xem
    docstring đầu file — "SỬA TẠI CHỖ", không phải "generate tiếp").
    """
    prompt    = row["prompt"]
    question  = extract_question(prompt)
    gt_answer = normalize_answer(row["answer"])
    gt_steps  = extract_steps(row["completion"])

    # ── LẦN GỌI MODEL DUY NHẤT ────────────────────────────────────────────
    rollout_text = generate(
        model, tokenizer, cfg, prompt,
        max_new=max_new_rollout, temperature=temperature,
        top_k=50, top_p=0.95, new_token_only=True,
    )
    model_steps = extract_steps(rollout_text)
    model_final = extract_answer(rollout_text)

    if not model_steps or model_final is None:
        return {"status": "parse_failed"}

    if model_final == gt_answer:
        return {"status": "already_correct"}

    # ── Localize bằng diff giá trị (KHÔNG Monte Carlo) ───────────────────
    model_values = [extract_step_value(s) for s in model_steps]
    gt_values    = [extract_step_value(s) for s in gt_steps]
    div_idx, gt_idx = find_divergence(model_values, gt_values)

    correct_prefix = model_steps[:div_idx]      # model tự viết, giữ nguyên
    wrong_step     = model_steps[div_idx]       # model tự viết, bước sai
    corrected_step = gt_steps[gt_idx] if 0 <= gt_idx < len(gt_steps) else wrong_step
    remaining_gt   = gt_steps[gt_idx + 1:] if 0 <= gt_idx < len(gt_steps) else []

    # ── Tier "sai-trần": chính rollout gốc, MIỄN PHÍ ─────────────────────
    tier_wrong_raw = build_completion(model_steps, model_final)

    # ── Tier "sai+sửa": SỬA TẠI CHỖ — 1 điểm nối DUY NHẤT (ngay tại bước
    #    sai, có "Wait" đánh dấu tường minh), phần còn lại 100% từ GT ─────
    prefix_text     = build_step_text(correct_prefix, start_num=1) if correct_prefix else ""
    corrected_line  = (
        f"Step {div_idx + 1}: {wrong_step} "
        f"Wait, that's wrong — it should be: {corrected_step}"
    )
    remaining_text  = build_step_text(remaining_gt, start_num=div_idx + 2) if remaining_gt else ""

    tier_wrong_fixed = "\n".join(
        part for part in [prefix_text, corrected_line, remaining_text] if part
    ) + f"\nAnswer: {gt_answer}"

    return {
        "status"                : "ok",
        "problem"               : question,
        "prompt"                : prompt,
        "ground_truth_answer"   : gt_answer,
        "divergence_step_idx"   : div_idx,
        "n_model_steps"         : len(model_steps),
        "tier_correct_straight" : row["completion"],   # từ GT — miễn phí, có sẵn
        "tier_wrong_raw"        : tier_wrong_raw,
        "tier_wrong_fixed"      : tier_wrong_fixed,     # từ GT — miễn phí, đáp số LUÔN đúng
    }


# ══════════════════════════════════════════════════════════════════════════
# Driver
# ══════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2b — Diff-based 3-tier self-correction data")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--train-jsonl", type=str, default="data/gsm8k_sft/train.jsonl")
    parser.add_argument("--output",      type=str, default="data/gsm8k_selfcorrect/tiers.jsonl")
    parser.add_argument("--n-problems",  type=int, default=-1,
                         help="-1 = dùng TOÀN BỘ train.jsonl (chỉ 1 lần gọi model/bài nên khả thi)")
    parser.add_argument("--max-new-rollout", type=int, default=200)
    parser.add_argument("--temperature",     type=float, default=0.8)
    args = parser.parse_args()

    print(f"Loading model từ {args.checkpoint} ...")
    model, tokenizer, cfg = load_model_for_inference(args.checkpoint)

    all_rows = load_jsonl(args.train_jsonl)
    rows = all_rows if args.n_problems <= 0 else all_rows[:args.n_problems]
    print(f"Xử lý {len(rows)}/{len(all_rows)} bài (1 lần gọi model/bài — không cần Monte Carlo, "
          f"không cần generate lần 2)\n")

    results = []
    n_parse_failed, n_already_correct, n_ok = 0, 0, 0
    t0 = time.time()

    for idx, row in enumerate(rows):
        out = process_one_problem(
            model, tokenizer, cfg, row,
            max_new_rollout=args.max_new_rollout,
            temperature=args.temperature,
        )

        if out["status"] == "parse_failed":
            n_parse_failed += 1
        elif out["status"] == "already_correct":
            n_already_correct += 1
        else:
            n_ok += 1
            results.append(out)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t0
            rate    = elapsed / (idx + 1)
            eta     = rate * (len(rows) - idx - 1)
            print(f"  [{idx+1}/{len(rows)}] {elapsed:.0f}s | ~{rate:.1f}s/bài | "
                  f"ETA ~{eta/60:.1f} phút | ok={n_ok} correct_sẵn={n_already_correct} "
                  f"parse_hỏng={n_parse_failed}")

    save_jsonl(results, args.output)

    print(f"\n{'='*60}")
    print(f"  THỐNG KÊ")
    print(f"{'='*60}")
    print(f"  Tổng bài xử lý          : {len(rows)}")
    print(f"  Đã đúng ngay (bỏ qua)   : {n_already_correct} ({100*n_already_correct/len(rows):.1f}%)")
    print(f"  Parse hỏng (bỏ qua)     : {n_parse_failed}")
    print(f"  Có 3-tier hợp lệ        : {n_ok}")
    if results:
        avg_div = sum(r["divergence_step_idx"] for r in results) / len(results)
        avg_steps = sum(r["n_model_steps"] for r in results) / len(results)
        print(f"  Vị trí sai trung bình   : step {avg_div:.1f}/{avg_steps:.1f} (tính từ 0)")


if __name__ == "__main__":
    main()