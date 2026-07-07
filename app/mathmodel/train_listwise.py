"""
train_listwise.py — Entry point chạy Stage 3c: Plackett-Luce/PRO listwise training
===========================================================
Điều kiện trước khi chạy:
    1. Có checkpoint để khởi tạo — thường là SFT (Stage 1, vd
       checkpoints_sft/sft_best.pt), nhưng CŨNG dùng được checkpoint listwise
       từ vòng trước (--sft-ckpt checkpoints_listwise/listwise_best.pt) nếu
       đang chạy VÒNG LẶP tự cải thiện (label_selfcorrect_data.py → train
       → dùng checkpoint mới label lại → train tiếp). Tên tham số giữ
       "--sft-ckpt" vì trường hợp phổ biến nhất là Stage 1, nhưng về bản
       chất chỉ là "checkpoint khởi tạo cho CẢ π_θ lẫn π_ref".
    2. Đã chạy scripts/label_selfcorrect_data.py → có data/gsm8k_selfcorrect/tiers.jsonl
       (KHÔNG phải data/gsm8k_dpo/pairs.jsonl — đó là pipeline DPO riêng,
       chỉ có 2 mức, không dùng được cho listwise 3 mức ở đây)

CÓ VAL SPLIT + EVAL ĐỊNH KỲ (mỗi epoch) — quan trọng vì dataset thường nhỏ
(vài trăm bài), rất dễ overfit nếu chỉ nhìn train loss. So sánh train_loss
vs VAL_loss mỗi epoch để biết còn đang học tổng quát hay đã bắt đầu học
thuộc lòng — đặc biệt cần chú ý nếu đang lặp nhiều vòng trên cùng 1 tập nhỏ.

Usage:
    python train_listwise.py \
        --sft-ckpt checkpoints_sft/sft_best.pt \
        --tiers-jsonl data/gsm8k_selfcorrect/tiers.jsonl \
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
from listwise_dataset import make_listwise_dataloaders
from trainer import run_listwise


def parse_args():
    p = argparse.ArgumentParser(description="Stage 3c — Plackett-Luce/PRO listwise training")
    p.add_argument("--sft-ckpt",     type=str, required=True,
                   help="Checkpoint SFT (Stage 1) — KHÔNG phải checkpoint DPO/listwise cũ")
    p.add_argument("--tiers-jsonl",  type=str, default="data/gsm8k_selfcorrect/tiers.jsonl")
    p.add_argument("--n-epochs",     type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-6,
                   help="LR nhỏ giống DPO — vẫn là fine-tune tiếp trên tập preference nhỏ")
    p.add_argument("--beta",         type=float, default=0.1)
    p.add_argument("--val-ratio",    type=float, default=0.1,
                   help="Tỷ lệ tách val từ tiers.jsonl — cao hơn SFT (0.03) vì dataset nhỏ")
    p.add_argument("--save-dir",     type=str,   default="checkpoints_listwise")
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

    # ── Hyperparameter riêng cho listwise ─────────────────────────────────────
    cfg.train.lr                   = args.lr
    cfg.train.batch_size           = args.batch_size
    cfg.train.warmup_steps         = 5
    cfg.train.lr_decay_cycle_steps = 100
    cfg.train.save_dir             = args.save_dir

    # ── Tokenizer — PHẢI cùng tokenizer đã dùng lúc SFT/pretrain ─────────────
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ── Model — build kiến trúc, weight thật load trong run_listwise() ───────
    model = build_model(cfg)
    print(f"Total params: {model.num_params() / 1e6:.1f}M")

    # ── Data — cần 3 tier/bài (tier_correct_straight/tier_wrong_fixed/tier_wrong_raw) ──
    train_loader, val_loader = make_listwise_dataloaders(
        args.tiers_jsonl, tokenizer, batch_size=args.batch_size,
        val_ratio=args.val_ratio,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    run_listwise(
        cfg, model, tokenizer, train_loader, val_loader,
        n_epochs=args.n_epochs, beta=args.beta,
        init_checkpoint=args.sft_ckpt,
    )


if __name__ == "__main__":
    main()