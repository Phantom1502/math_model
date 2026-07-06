"""
train.py — Entry point chạy pretrain
========================================
Usage trên Colab:

    !pip install transformers datasets accelerate -q
    !python train.py

Hoặc trong notebook cell:

    from config import get_100m_config
    from train import main
    main(get_100m_config())
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import get_100m_config, get_small_config, get_110m_config
from tokenizer import load_tokenizer
from dataset import ChunkedWikiLoader, ChunkedVTSNLPLoader, ChunkedParquetLoader, ChunkedMixLoader
from model import build_model
from trainer import run_pretrain


def main(cfg=None, start_chunk: int = 0, reset_lr_for_new_round: bool = False):
    """
    Args:
        cfg        : Config object
        start_chunk: chunk muốn bắt đầu/tiếp tục train (0 = từ đầu dataset).
        reset_lr_for_new_round: True khi muốn lr/scheduler mới có hiệu lực
                   thật sự (tạo lại optimizer, bỏ qua lr cũ trong checkpoint).

    ────────────────────────────────────────────────────────────────────────
    NGUỒN DỮ LIỆU:

    Wikipedia (mặc định):
        cfg.data.source = "wikipedia"

    VTSNLP curated:
        cfg.data.source = "vtsnlp"
        cfg.data.vtsnlp_domains = ["Science", "Books_and_Literature"]  # optional

    Local parquet (sách, corpus nội bộ):
        cfg.data.source           = "parquet"
        cfg.data.parquet_path     = "data/books.parquet"
        cfg.data.parquet_text_col = "text"   # đổi nếu cột tên khác

    Lọc parquet theo metadata (filter_fn không serializable — khởi tạo
    loader thủ công rồi truyền vào run_pretrain qua data_loader_gen):
        from dataset import ChunkedParquetLoader
        loader = ChunkedParquetLoader(
            cfg, tokenizer, "data/books.parquet",
            filter_fn=lambda s: s.get("genre") == "Lịch sử",
        )
        run_pretrain(cfg, model, tokenizer, data_loader_gen=loader)

    ────────────────────────────────────────────────────────────────────────
    TRAIN ROUND MỚI VỚI CONFIG KHÁC:

    Trường hợp 1 — chỉ đổi hyperparameter train, giữ nguyên kiến trúc:
        cfg.train.resume_from = "checkpoints/chunk_33.pt"
        cfg.train.lr = 1e-4
        main(cfg, start_chunk=34, reset_lr_for_new_round=True)

    Trường hợp 2 — đổi kiến trúc (num_slots, d_model, use_memory, ...):
        cfg.train.resume_from = None   # KHÔNG thể resume — phải train lại
        main(cfg, start_chunk=0)
    ────────────────────────────────────────────────────────────────────────
    """
    if cfg is None:
        cfg = get_small_config()

    torch.manual_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\n── Loading tokenizer (PhoBERT) ──")
    tokenizer = load_tokenizer(cfg)
    cfg.model.vocab_size = tokenizer.vocab_size
    print(f"Vocab size     : {tokenizer.vocab_size}")
    print(f"strict_chart_mode: {tokenizer.strict_chart_mode}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    model = build_model(cfg)
    print(f"Total params     : {model.num_params()/1e6:.1f}M")
    print(f"Trainable params : {model.num_params(trainable_only=True)/1e6:.1f}M")

    # ── Data loader (incremental, RAM-safe) ──────────────────────────────────
    print("\n── Setting up incremental data loader ──")
    print(f"Source: {cfg.data.source} | Chunk size: {cfg.data.chunk_size} sample/chunk")
    if start_chunk > 0:
        print(f"Bắt đầu từ chunk {start_chunk} (skip {start_chunk * cfg.data.chunk_size:,} sample đầu)")

    if cfg.data.source == "vtsnlp":
        if cfg.data.vtsnlp_domains:
            print(f"Lọc domain: {cfg.data.vtsnlp_domains}")
        data_loader_gen = ChunkedVTSNLPLoader(
            cfg, tokenizer, start_chunk=start_chunk, domains=cfg.data.vtsnlp_domains
        )

    elif cfg.data.source == "wikipedia":
        data_loader_gen = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)

    elif cfg.data.source == "parquet":
        if not cfg.data.parquet_path:
            raise ValueError(
                "cfg.data.source='parquet' nhưng cfg.data.parquet_path chưa được đặt.\n"
                "Ví dụ: cfg.data.parquet_path = 'data/books.parquet'\n"
                "Nếu cần lọc theo metadata (author, genre, ...), khởi tạo "
                "ChunkedParquetLoader trực tiếp với filter_fn thay vì dùng main()."
            )
        print(f"Parquet: {cfg.data.parquet_path} | cột text: '{cfg.data.parquet_text_col}'")
        data_loader_gen = ChunkedParquetLoader(
            cfg, tokenizer,
            parquet_path=cfg.data.parquet_path,
            text_col    =cfg.data.parquet_text_col,
            start_chunk =start_chunk,
        )

    elif cfg.data.source == "mix":
        data_loader_gen = ChunkedMixLoader(cfg, tokenizer, start_chunk=start_chunk)

    else:
        raise ValueError(
            f"cfg.data.source='{cfg.data.source}' không hợp lệ — "
            f"chỉ hỗ trợ 'wikipedia', 'vtsnlp', hoặc 'parquet'"
        )

    # ── Train ──────────────────────────────────────────────────────────────
    print("\n── Starting pretraining ──")
    trainer = run_pretrain(cfg, model, tokenizer, data_loader_gen,
                            start_chunk=start_chunk,
                            reset_lr_for_new_round=reset_lr_for_new_round)

    return trainer


if __name__ == "__main__":
    main()