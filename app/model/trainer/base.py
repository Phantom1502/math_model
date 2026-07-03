"""
trainer/base.py — Logic chung cho mọi loại training (pretrain / SFT / DPO)
============================================================================
Chứa:
    - optimizer / scheduler setup (cosine annealing with warm restarts)
    - gradient accumulation + mixed precision
    - evaluate loop
    - benchmark hook

Subclass override compute_loss() nếu cần loss khác cross-entropy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import causal_mask
from utils import TrainLogger, log_eval, log_bench, save_checkpoint
from benchmark import run_all


class BaseTrainer:
    def __init__(self, cfg, model, tokenizer):
        self.cfg       = cfg
        self.model     = model
        self.tokenizer = tokenizer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.device.type == "cuda" and cfg.train.mixed_precision)
        )

        self.global_step   = 0
        self.chunk_idx     = 0
        self.best_val_loss = float("inf")

        self.logger = TrainLogger(log_every=cfg.train.log_every)
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Warmup tuyến tính → cosine decay theo chu kỳ (SGDR)."""
        warmup      = self.cfg.train.warmup_steps
        cycle_steps = self.cfg.train.lr_decay_cycle_steps
        min_ratio   = self.cfg.train.lr_min_ratio

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            step_in_cycle = (step - warmup) % cycle_steps
            progress      = step_in_cycle / max(cycle_steps, 1)
            cosine        = 0.5 * (1 + math.cos(math.pi * progress))
            return min_ratio + (1 - min_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mặc định: cross-entropy next-token prediction. Override ở subclass."""
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def train_one_batch(self, batch: dict, accum_step: int) -> float:
        ids    = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        B, T   = ids.shape
        mask   = causal_mask(T, self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            logits = self.model(ids, attn_mask=mask)
            loss   = self.compute_loss(logits, labels)

        self.scaler.scale(loss / self.cfg.train.grad_accum).backward()

        if (accum_step + 1) % self.cfg.train.grad_accum == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        self.model.eval()
        total, n = 0.0, 0

        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids    = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            B, T   = ids.shape
            mask   = causal_mask(T, self.device)
            logits = self.model(ids, attn_mask=mask)
            total += self.compute_loss(logits, labels).item()
            n     += 1

        self.model.train()
        return total / max(n, 1)

    def train_one_chunk(self, train_loader, val_loader, chunk_idx: int) -> float:
        self.model.train()
        self.chunk_idx = chunk_idx

        for epoch in range(self.cfg.train.epochs_per_chunk):
            for accum_step, batch in enumerate(train_loader):
                loss = self.train_one_batch(batch, accum_step)
                self.logger.update(loss)

                if self.logger.should_log():
                    lr = self.scheduler.get_last_lr()[0]
                    self.logger.flush(step=self.global_step, lr=lr)

                if self.global_step > 0 and self.global_step % self.cfg.train.eval_every == 0:
                    val_loss = self.evaluate(val_loader)
                    log_eval(val_loss, step=self.global_step)

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            f"{self.cfg.train.save_dir}/best.pt",
                            self.model, self.optimizer, self.scheduler,
                            self.global_step, self.chunk_idx, val_loss,
                            model_cfg=self.cfg.model,
                        )

                    bench = run_all(self.model, self.tokenizer, self.cfg, verbose=False, step=self.global_step)
                    self.model.train()
                    log_bench(bench, step=self.global_step)

                if self.global_step > 0 and self.global_step % self.cfg.train.save_every == 0:
                    save_checkpoint(
                        f"{self.cfg.train.save_dir}/step_{self.global_step}.pt",
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, self.chunk_idx,
                        model_cfg=self.cfg.model,
                    )

        val_loss = self.evaluate(val_loader)
        log_eval(val_loss, step=self.global_step, prefix="  [Chunk end] ")

        bench = run_all(self.model, self.tokenizer, self.cfg, verbose=False, step=self.global_step)
        log_bench(bench, step=self.global_step)
        return val_loss