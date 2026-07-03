"""
trainer/pretrain.py — Pretraining
===================================
"""

import torch
from .base import BaseTrainer
from utils import save_checkpoint, load_checkpoint
from utils.checkpoint import hf_upload_latest


class PretrainTrainer(BaseTrainer):
    """Kế thừa nguyên BaseTrainer — pretrain dùng cross-entropy mặc định."""
    pass


def run_pretrain(cfg, model, tokenizer, data_loader_gen=None, start_chunk: int = 0,
                 reset_lr_for_new_round: bool = False):

    trainer = PretrainTrainer(cfg, model, tokenizer)

    # Resume checkpoint
    if cfg.train.resume_from:
        state = load_checkpoint(
            cfg.train.resume_from, trainer.model,
            trainer.optimizer, trainer.scheduler, trainer.device,
        )
        trainer.global_step   = state["global_step"]
        trainer.best_val_loss = state["val_loss"] or float("inf")

        if start_chunk == 0:
            start_chunk = state["chunk_idx"]

        print(f"Resuming từ step {trainer.global_step}, chunk {start_chunk}")

        if reset_lr_for_new_round:
            print(f"reset_lr_for_new_round=True: tạo lại optimizer lr={cfg.train.lr}")
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay,
                betas=(0.9, 0.95),
            )
            trainer._setup_scheduler()
            for _ in range(trainer.global_step):
                trainer.scheduler.step()

    # Tạo data loader nếu chưa có
    if data_loader_gen is None:
        if cfg.data.source == "mix":
            from dataset import ChunkedMixLoader
            data_loader_gen = ChunkedMixLoader(cfg, tokenizer, start_chunk=start_chunk)
        elif cfg.data.source == "wikipedia":
            from dataset import ChunkedWikiLoader
            data_loader_gen = ChunkedWikiLoader(cfg, tokenizer, start_chunk=start_chunk)
        elif cfg.data.source == "vtsnlp":
            from dataset import ChunkedVTSNLPLoader
            data_loader_gen = ChunkedVTSNLPLoader(
                cfg, tokenizer, start_chunk=start_chunk,
                domains=cfg.data.vtsnlp_domains,
            )
        elif cfg.data.source == "parquet":
            from dataset import ChunkedParquetLoader
            data_loader_gen = ChunkedParquetLoader(
                cfg, tokenizer,
                parquet_path=cfg.data.parquet_path,
                text_col    =cfg.data.parquet_text_col,
                start_chunk =start_chunk,
            )
        else:
            raise ValueError(f"cfg.data.source='{cfg.data.source}' không hợp lệ")

    hf_repo_id = cfg.train.hf_repo_id
    hf_token   = cfg.train.hf_token

    # Train loop
    for train_loader, val_loader in data_loader_gen:
        chunk_idx = data_loader_gen.chunk_count

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx}")
        print(f"{'='*60}")

        val_loss   = trainer.train_one_chunk(train_loader, val_loader, chunk_idx)
        chunk_path = f"{cfg.train.save_dir}/chunk_{chunk_idx}.pt"

        save_checkpoint(
            chunk_path,
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx, val_loss,
            model_cfg=cfg.model,
        )

        if hf_repo_id:
            hf_upload_latest(chunk_path, repo_id=hf_repo_id, token=hf_token)

    print("\n✓ Pretraining hoàn tất toàn bộ dataset.")
    return trainer