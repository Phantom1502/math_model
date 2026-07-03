"""
benchmark_hellaswag.py — Đánh giá MemoryLM trên HellaSwag
===========================================================
Zero-shot evaluation: model chọn completion có avg log-prob/token cao nhất.

Lưu ý:
    HellaSwag gốc là tiếng Anh — model tiếng Việt sẽ cho accuracy thấp
    (~25% = random baseline). Dùng để theo dõi TREND qua các checkpoint,
    không phải để so sánh tuyệt đối với GPT-2/LLaMA.

Usage:
    python benchmark_hellaswag.py checkpoints/chunk_10.pt
    python benchmark_hellaswag.py ckpt_10.pt ckpt_50.pt --n-samples 200
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset

from model import causal_mask


# ══════════════════════════════════════════════════════════════════════════
# Core
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def avg_logprob_per_token(model, tokenizer, prompt, completion, device, max_seq=512) -> float:
    prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    if not completion_ids:
        return float("-inf")

    full_ids = prompt_ids + completion_ids
    if len(full_ids) > max_seq:
        keep = max_seq - len(completion_ids)
        if keep <= 0:
            return float("-inf")
        full_ids = prompt_ids[-keep:] + completion_ids

    ids_t     = torch.tensor([full_ids], dtype=torch.long, device=device)
    T         = ids_t.size(1)
    logits    = model(ids_t, attn_mask=causal_mask(T, device))
    log_probs = F.log_softmax(logits[0], dim=-1)

    n_prompt = T - len(completion_ids)
    total_lp = sum(
        log_probs[n_prompt + i - 1, tok_id].item()
        for i, tok_id in enumerate(completion_ids)
        if n_prompt + i - 1 >= 0
    )
    return total_lp / len(completion_ids)


# ══════════════════════════════════════════════════════════════════════════
# Load HellaSwag
# ══════════════════════════════════════════════════════════════════════════

def _load_hellaswag(split="validation", n_samples=None) -> list[dict]:
    ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))

    return [
        {
            "ctx"    : f"{row['activity_label']}: {row['ctx']}",
            "endings": row["endings"],
            "label"  : int(row["label"]),
        }
        for row in ds
    ]


# ══════════════════════════════════════════════════════════════════════════
# Evaluate
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_hellaswag(model, tokenizer, cfg, n_samples=500, split="validation", verbose=True) -> dict:
    device  = next(model.parameters()).device
    max_seq = cfg.model.max_seq
    model.eval()

    items   = _load_hellaswag(split=split, n_samples=n_samples)
    n_total = len(items)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  HELLASWAG  ({split}, n={n_total})")
        print(f"  Random baseline: 25.0%")
        print(f"{'═'*60}")

    n_correct = 0
    for i, item in enumerate(items):
        scores = [
            avg_logprob_per_token(model, tokenizer, item["ctx"], ending, device, max_seq)
            for ending in item["endings"]
        ]
        if scores.index(max(scores)) == item["label"]:
            n_correct += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i+1:>4}/{n_total}] acc so far: {n_correct/(i+1)*100:.1f}%")

    accuracy = n_correct / n_total

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Accuracy : {accuracy*100:.2f}%")
        print(f"  Correct  : {n_correct}/{n_total}")
        print(f"  vs random: {(accuracy - 0.25)*100:+.2f}%")
        print(f"{'═'*60}\n")

    return {
        "accuracy"       : accuracy,
        "n_correct"      : n_correct,
        "n_total"        : n_total,
        "random_baseline": 0.25,
        "delta_vs_random": accuracy - 0.25,
    }


def hellaswag_score_for_run_all(model, tokenizer, cfg, n_samples=200) -> float:
    return run_hellaswag(model, tokenizer, cfg, n_samples=n_samples, verbose=False)["accuracy"]


# ══════════════════════════════════════════════════════════════════════════
# Compare checkpoints
# ══════════════════════════════════════════════════════════════════════════

def compare_hellaswag(checkpoint_paths, n_samples=200, verbose=False) -> None:
    from generate import load_model_for_inference

    rows = []
    for path in checkpoint_paths:
        print(f"\n── Loading {path} ──")
        model, tokenizer, cfg = load_model_for_inference(path)
        r = run_hellaswag(model, tokenizer, cfg, n_samples=n_samples, verbose=verbose)
        r["checkpoint"] = path
        rows.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'═'*60}")
    print(f"  {'CHECKPOINT':<30} {'ACC':>7}  {'vs RND':>7}")
    print(f"{'─'*60}")
    for r in rows:
        name = r["checkpoint"].split("/")[-1]
        print(f"  {name:<30} {r['accuracy']*100:>6.2f}%  {r['delta_vs_random']*100:>+6.2f}%")
    print(f"{'═'*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--split",     type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    if len(args.checkpoints) == 1:
        from generate import load_model_for_inference
        model, tokenizer, cfg = load_model_for_inference(args.checkpoints[0])
        run_hellaswag(model, tokenizer, cfg, n_samples=args.n_samples, split=args.split, verbose=True)
    else:
        compare_hellaswag(args.checkpoints, n_samples=args.n_samples, verbose=args.verbose)