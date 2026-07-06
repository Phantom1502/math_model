"""
train_sft.py — Entry point chạy Stage 1: SFT trên GSM8K
===========================================================
Điều kiện trước khi chạy:
    1. Đã có pretrained checkpoint (model biết đọc/viết cơ bản).
    2. Đã chạy scripts/prepare_sft_data.py → có data/gsm8k_sft/{train,test}.jsonl
       (test.jsonl KHÔNG dùng ở đây — giữ nguyên cho Stage 4).

Usage:
    python train_sft.py \
        --pretrained-ckpt checkpoints/chunk_50.pt \
        --train-jsonl data/gsm8k_sft/train.jsonl \
        --n-epochs 3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import ModelConfig, get_100m_config
from tokenizer import load_tokenizer
from model import build_model
from sft_dataset import make_sft_dataloaders
from trainer import run_sft


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 — SFT format CoT trên GSM8K")
    p.add_argument("--pretrained-ckpt", type=str, required=True,
                   help="Checkpoint pretrained hiện có (lấy model_cfg + load weight khởi tạo)")
    p.add_argument("--train-jsonl", type=str, default="data/gsm8k_sft/train.jsonl")
    p.add_argument("--n-epochs",    type=int,   default=3)
    p.add_argument("--batch-size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=5e-5,
                   help="LR cho SFT — nên nhỏ hơn lr lúc pretrain (tránh phá vỡ kiến thức đã học)")
    p.add_argument("--val-ratio",   type=float, default=0.03)
    p.add_argument("--save-dir",    type=str,   default="checkpoints_sft")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_100m_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")

    # ── Đọc model_cfg từ checkpoint pretrained — đảm bảo kiến trúc khớp ──────
    ckpt_raw = torch.load(args.pretrained_ckpt, map_location=device)
    if "model_cfg" in ckpt_raw and ckpt_raw["model_cfg"] is not None:
        cfg.model = ModelConfig(**ckpt_raw["model_cfg"])
        print(f"  model_cfg từ checkpoint: d_model={cfg.model.d_model}, "
              f"n_layers={cfg.model.n_layers}, max_seq={cfg.model.max_seq}")
    else:
        print("  CẢNH BÁO: checkpoint không có model_cfg — dùng config mặc định, "
              "có thể KHÔNG khớp kiến trúc lúc pretrain → load_state_dict sẽ lỗi.")

    # ── Hyperparameter train riêng cho giai đoạn SFT ─────────────────────────
    cfg.train.lr                   = args.lr
    cfg.train.batch_size           = args.batch_size
    cfg.train.warmup_steps         = 50
    cfg.train.lr_decay_cycle_steps = 2000
    cfg.train.save_dir             = args.save_dir
    cfg.train.eval_every           = 200
    cfg.train.save_every           = 500

    # ── Tokenizer — PHẢI cùng tokenizer đã dùng lúc pretrain checkpoint này ──
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── Model — build đúng kiến trúc theo model_cfg, weight thật load trong run_sft ──
    model = build_model(cfg)
    print(f"Total params: {model.num_params() / 1e6:.1f}M")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = make_sft_dataloaders(
        args.train_jsonl, tokenizer,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    # ── Train ────────────────────────────────────────────────────────────
    run_sft(
        cfg, model, tokenizer,
        train_loader, val_loader,
        n_epochs=args.n_epochs,
        init_checkpoint=args.pretrained_ckpt,
    )


if __name__ == "__main__":
    main()