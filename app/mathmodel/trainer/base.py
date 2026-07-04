"""
trainer/base.py — Logic chung cho mọi loại training (pretrain / SFT / DPO)
============================================================================
Chứa:
    - optimizer / scheduler setup (cosine annealing with warm restarts)
    - gradient accumulation + mixed precision
    - evaluate loop
    - benchmark hook

Subclass override compute_loss() nếu cần loss khác cross-entropy.

CHANGELOG (fix): loss trước đây dùng reduction="mean" theo từng micro-batch,
khiến các micro-batch được weight sai khi cộng dồn gradient (xem
BUGFIX_grad_accum_loss_weighting.md). Nay đổi sang reduction="sum" và chuẩn
hóa theo tổng số token hợp lệ của CẢ cửa sổ accumulation.
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
        """Mặc định: cross-entropy next-token prediction, reduction='sum'.

        QUAN TRỌNG: dùng reduction="sum" thay vì "mean" mặc định. Việc chuẩn
        hóa theo số token hợp lệ được thực hiện ở cấp cửa sổ accumulation
        (xem train_one_batch / _run_accum_window), không phải ở đây — vì
        "mean" tại đây chỉ trung bình theo số token của MỘT micro-batch,
        gây sai lệch trọng số khi cộng dồn gradient qua nhiều micro-batch
        có số token hợp lệ khác nhau (do padding / ignore_index).
        Override ở subclass nếu cần loss khác, nhưng giữ nguyên reduction="sum".
        """
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    def train_one_batch(
        self,
        batch: dict,
        total_valid_tokens: int,
        is_last_in_window: bool,
    ) -> float:
        """Chạy forward + backward cho MỘT micro-batch trong một cửa sổ accumulation.

        Args:
            batch: micro-batch hiện tại.
            total_valid_tokens: tổng số token hợp lệ (!= -100) trên TOÀN cửa
                sổ accumulation (tất cả micro-batch sẽ được cộng dồn trước khi
                optimizer step), không phải chỉ của batch này.
            is_last_in_window: True nếu đây là micro-batch cuối cùng của cửa
                sổ hiện tại → sẽ trigger optimizer step.

        Returns:
            Loss trung bình theo token (để log), không phải loss tổng.
        """
        ids    = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        T      = ids.shape[1]
        mask   = causal_mask(T, self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            logits   = self.model(ids, attn_mask=mask)
            loss_sum = self.compute_loss(logits, labels)  # reduction="sum"

        # Chuẩn hóa theo tổng token của CẢ cửa sổ accumulation, không phải
        # theo grad_accum (số bước) và không phải theo token của riêng batch này.
        self.scaler.scale(loss_sum / total_valid_tokens).backward()

        if is_last_in_window:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return loss_sum.item() / total_valid_tokens

    def _run_accum_window(self, batches: list) -> None:
        """Chạy trọn 1 cửa sổ gradient accumulation (N micro-batch → 1 optimizer step).

        N thường bằng cfg.train.grad_accum, ngoại trừ cửa sổ cuối cùng của mỗi
        epoch có thể ngắn hơn nếu số micro-batch còn lại không chia hết.
        """
        total_valid_tokens = sum(
            (b["labels"] != -100).sum().item() for b in batches
        )
        # Tránh chia cho 0 trong trường hợp hiếm gặp cả cửa sổ toàn token pad.
        total_valid_tokens = max(total_valid_tokens, 1)

        for i, batch in enumerate(batches):
            is_last = (i == len(batches) - 1)
            loss = self.train_one_batch(batch, total_valid_tokens, is_last)
            self.logger.update(loss)

            if self.logger.should_log():
                lr = self.scheduler.get_last_lr()[0]
                self.logger.flush(step=self.global_step, lr=lr)

    def _maybe_eval_and_save(self, val_loader) -> None:
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

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches: int = 50) -> float:
        """Val loss trung bình theo token trên toàn bộ các batch được lấy mẫu."""
        self.model.eval()
        total_loss_sum, total_tokens = 0.0, 0

        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            ids    = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            T      = ids.shape[1]
            mask   = causal_mask(T, self.device)
            logits = self.model(ids, attn_mask=mask)

            total_loss_sum += self.compute_loss(logits, labels).item()
            total_tokens   += (labels != -100).sum().item()

        self.model.train()
        return total_loss_sum / max(total_tokens, 1)

    def train_one_chunk(self, train_loader, val_loader, chunk_idx: int) -> float:
        self.model.train()
        self.chunk_idx = chunk_idx
        grad_accum = self.cfg.train.grad_accum

        for epoch in range(self.cfg.train.epochs_per_chunk):
            accum_buffer = []

            for batch in train_loader:
                accum_buffer.append(batch)

                if len(accum_buffer) == grad_accum:
                    self._run_accum_window(accum_buffer)
                    accum_buffer = []
                    self._maybe_eval_and_save(val_loader)

            # Cửa sổ cuối epoch: nếu số batch còn lại không chia hết grad_accum,
            # vẫn chạy nốt (cửa sổ ngắn hơn) để không bỏ sót dữ liệu/gradient.
            if accum_buffer:
                self._run_accum_window(accum_buffer)
                accum_buffer = []
                self._maybe_eval_and_save(val_loader)

        val_loss = self.evaluate(val_loader)
        log_eval(val_loss, step=self.global_step, prefix="  [Chunk end] ")

        bench = run_all(self.model, self.tokenizer, self.cfg, verbose=False, step=self.global_step)
        log_bench(bench, step=self.global_step)
        return val_loss