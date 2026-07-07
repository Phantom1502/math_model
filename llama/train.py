"""
app/llama/train.py — Entry point pretrain cho nhánh HF LlamaForCausalLM
============================================================================
Usage (Colab, chạy từ trong app/llama/):

    !pip install "transformers>=4.38" datasets accelerate -q
    !python train.py

Hoặc trong notebook cell:

    from config import Config
    from train import main
    main(Config())

Resume — khác bản cũ: resume_from trỏ tới THƯ MỤC (không phải file .pt),
vì checkpoint giờ lưu bằng model.save_pretrained():

    cfg.train.resume_from = "checkpoints_llama/step_5000"
    main(cfg)   # start_chunk tự đọc từ train_state.json trong thư mục đó

Đổi nguồn dữ liệu — giống hệt convention cũ (xem app/memlm/train.py):
    cfg.data.source = "wikipedia" | "vtsnlp" | "parquet" | "mix"

Riêng nguồn "parquet"/"mix" TỰ ĐỘNG convert price token cũ ("O_512 H_..")
sang định dạng mới ("<px_O_512>") trước khi tokenize — không cần sửa lại
dữ liệu đã sinh từ app/utils/chart/*.
"""

import torch
from transformers import LlamaForCausalLM

from config import Config
from tokenizer import load_llama_tokenizer, convert_legacy_price_tokens
from model import build_model, num_params
from dataset import (
    ChunkedWikiLoader, ChunkedVTSNLPLoader, ChunkedParquetLoader, ChunkedMixLoader,
)
from trainer import LlamaPretrainTrainer, save_checkpoint, load_train_state, load_optimizer_scheduler


def main(cfg: Config = None, start_chunk: int = 0):
    if cfg is None:
        cfg = Config()

    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.train.device = device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("\n── Loading tokenizer (BPE + price token thật) ──")
    tokenizer = load_llama_tokenizer(cfg)
    print(f"Vocab size: {len(tokenizer):,}")

    # ── Model ────────────────────────────────────────────────────────────
    print("\n── Building model ──")
    if cfg.train.resume_from:
        print(f"Loading model từ checkpoint: {cfg.train.resume_from}")
        model = LlamaForCausalLM.from_pretrained(cfg.train.resume_from)
        state = load_train_state(cfg.train.resume_from)
        if start_chunk == 0:
            start_chunk = state["chunk_idx"]
    else:
        model = build_model(cfg, tokenizer)

    print(f"Total params: {num_params(model)/1e6:.1f}M")

    # ── Data loader ──────────────────────────────────────────────────────
    print(f"\n── Data loader: source={cfg.data.source} | chunk_size={cfg.data.chunk_size} ──")
    if start_chunk > 0:
        print(f"Bắt đầu từ chunk {start_chunk}")

    # Chỉ cần convert price token cho nguồn liên quan tới dữ liệu trading
    # (parquet/mix) — wikipedia/vtsnlp là text thường, không cần transform.
    text_transform = convert_legacy_price_tokens if cfg.data.source in ("parquet", "mix") else None

    if cfg.data.source == "wikipedia":
        loader = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)

    elif cfg.data.source == "vtsnlp":
        if cfg.data.vtsnlp_domains:
            print(f"Lọc domain: {cfg.data.vtsnlp_domains}")
        loader = ChunkedVTSNLPLoader(
            cfg, tokenizer, start_chunk=start_chunk, domains=cfg.data.vtsnlp_domains,
        )

    elif cfg.data.source == "parquet":
        if not cfg.data.parquet_path:
            raise ValueError(
                "cfg.data.source='parquet' nhưng cfg.data.parquet_path chưa được đặt."
            )
        print(f"Parquet: {cfg.data.parquet_path} | cột text: '{cfg.data.parquet_text_col}'")
        loader = ChunkedParquetLoader(
            cfg, tokenizer, parquet_path=cfg.data.parquet_path,
            text_col=cfg.data.parquet_text_col, start_chunk=start_chunk,
            text_transform=text_transform,
        )

    elif cfg.data.source == "mix":
        loader = ChunkedMixLoader(cfg, tokenizer, start_chunk=start_chunk, text_transform=text_transform)

    else:
        raise ValueError(f"cfg.data.source='{cfg.data.source}' không hợp lệ")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = LlamaPretrainTrainer(cfg, model, tokenizer)

    if cfg.train.resume_from:
        state = load_train_state(cfg.train.resume_from)
        trainer.global_step   = state["global_step"]
        trainer.best_val_loss = state["val_loss"] or float("inf")
        load_optimizer_scheduler(cfg.train.resume_from, trainer.optimizer, trainer.scheduler, device)
        print(f"Resuming từ step {trainer.global_step}, chunk {start_chunk}")

    # ── Train loop ───────────────────────────────────────────────────────
    print("\n── Starting pretraining ──")
    for train_loader, val_loader in loader:
        chunk_idx = loader.chunk_count

        print(f"\n{'='*60}\nCHUNK {chunk_idx}\n{'='*60}")
        val_loss = trainer.train_one_chunk(train_loader, val_loader, chunk_idx)

        save_checkpoint(
            f"{cfg.train.save_dir}/chunk_{chunk_idx}", trainer.model, tokenizer,
            trainer.optimizer, trainer.scheduler, trainer.global_step, chunk_idx, val_loss,
        )

    print("\n✓ Pretraining hoàn tất toàn bộ dataset.")
    return trainer


if __name__ == "__main__":
    main()