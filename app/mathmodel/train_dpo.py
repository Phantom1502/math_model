"""
train_dpo.py — Entry point chạy Stage 3b: DPO trên preference pairs
===========================================================
Điều kiện trước khi chạy:
    1. Đã có checkpoint SFT (Stage 1) — vd checkpoints_sft/sft_best.pt
    2. Đã chạy scripts/build_dpo_pairs.py → có data/gsm8k_dpo/pairs.jsonl

Usage:
    python train_dpo.py \
        --sft-ckpt checkpoints_sft/sft_best.pt \
        --pairs-jsonl data/gsm8k_dpo/pairs.jsonl \
        --n-epochs 10 --beta 0.1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import ModelConfig, get_100m_config
from tokenizer import load_tokenizer
from model import build_model
from dpo_dataset import make_dpo_dataloader
from trainer import run_dpo


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3b — DPO trên preference pairs")
    p.add_argument("--sft-ckpt",    type=str, required=True,
                   help="Checkpoint SFT (Stage 1) — KHÔNG phải checkpoint DPO cũ")
    p.add_argument("--pairs-jsonl", type=str, default="data/gsm8k_dpo/pairs.jsonl")
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-6,
                   help="LR cho DPO — NHỎ HƠN NHIỀU so với SFT (5e-5). "
                        "Dataset preference cực nhỏ (13 pairs) rất dễ overfit/drift "
                        "nếu lr quá lớn — xem bảng rủi ro trong spec.")
    p.add_argument("--beta",        type=float, default=0.1,
                   help="Hệ số beta trong DPO loss — càng nhỏ càng cho phép "
                        "policy lệch xa ref, càng lớn càng giữ sát ref")
    p.add_argument("--save-dir",    type=str,   default="checkpoints_dpo")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_100m_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")

    # ── Đọc model_cfg từ checkpoint SFT — đảm bảo kiến trúc khớp ─────────────
    ckpt_raw = torch.load(args.sft_ckpt, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        cfg.model = ModelConfig(**ckpt_raw["model_cfg"])
        print(f"  model_cfg: d_model={cfg.model.d_model}, n_layers={cfg.model.n_layers}, "
              f"max_seq={cfg.model.max_seq}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg — dùng config mặc định.")

    # ── Hyperparameter riêng cho DPO ─────────────────────────────────────────
    cfg.train.lr                   = args.lr
    cfg.train.batch_size           = args.batch_size
    cfg.train.warmup_steps         = 5      # dataset quá nhỏ, warmup dài vô nghĩa
    cfg.train.lr_decay_cycle_steps = 100
    cfg.train.save_dir             = args.save_dir

    # ── Tokenizer — PHẢI cùng tokenizer đã dùng lúc SFT/pretrain ─────────────
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── Model — build kiến trúc, weight thật load trong run_dpo() ────────────
    model = build_model(cfg)
    print(f"Total params: {model.num_params() / 1e6:.1f}M")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = make_dpo_dataloader(
        args.pairs_jsonl, tokenizer, batch_size=args.batch_size,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    run_dpo(
        cfg, model, tokenizer, train_loader,
        n_epochs=args.n_epochs, beta=args.beta,
        init_checkpoint=args.sft_ckpt,
    )


if __name__ == "__main__":
    main()