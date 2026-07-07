"""
app/llama/trainer.py — Training loop pretrain cho HF LlamaForCausalLM
=========================================================================
Giữ interface tương tự app/memlm/trainer/ (log/eval/checkpoint/benchmark hook)
nhưng gọi model theo convention HF chuẩn:

    out  = model(input_ids=ids, attention_mask=mask, use_cache=False)
    loss = cross_entropy(out.logits, labels)   # tự tính tay, KHÔNG truyền
                                                # labels=... vào forward()

Vì sao KHÔNG truyền labels=... thẳng vào model(): input_ids/labels ở đây đã
được dataset.py SHIFT SẴN (input_ids = doc[:-1], labels = doc[1:], cùng độ
dài seg_len) — giống hệt convention cũ của app/memlm. Nếu truyền labels=...
vào LlamaForCausalLM.forward(), HF sẽ TỰ SHIFT THÊM MỘT LẦN NỮA nội bộ
(shift_logits = logits[..., :-1, :], shift_labels = labels[..., 1:]) → double
shift → loss sai hoàn toàn. Do đó luôn lấy logits thô rồi tự cross_entropy.

Checkpoint: dùng model.save_pretrained()/tokenizer.save_pretrained() chuẩn
HF (KHÔNG phải torch.save(state_dict) thô như bản cũ) — cho phép load lại
bằng LlamaForCausalLM.from_pretrained(path) trực tiếp, và dùng ngay được
cho bước SFT kế tiếp (TRL/Trainer đều load model theo cách này).
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import run_all as run_llama_benchmark


# ══════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════

class TrainLogger:
    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.loss_sum  = 0.0
        self.count     = 0
        self.t_start   = time.time()

    def update(self, loss_value: float):
        self.loss_sum += loss_value
        self.count    += 1

    def should_log(self) -> bool:
        return self.count >= self.log_every

    def flush(self, step: int, lr: float, prefix: str = "") -> dict:
        avg     = self.loss_sum / max(self.count, 1)
        ppl     = math.exp(min(avg, 20))
        elapsed = time.time() - self.t_start

        print(f"{prefix}Step {step:>7} | loss: {avg:.4f} | ppl: {ppl:.2f} | "
              f"lr: {lr:.2e} | {elapsed:.0f}s")

        result = {"loss": avg, "ppl": ppl, "elapsed": elapsed}
        self.loss_sum, self.count, self.t_start = 0.0, 0, time.time()
        return result


def log_eval(val_loss: float, step: int = None, prefix: str = "  -- "):
    ppl = math.exp(min(val_loss, 20))
    step_str = f"Step {step} | " if step is not None else ""
    print(f"{prefix}{step_str}Val loss: {val_loss:.4f} | ppl: {ppl:.2f}")


def log_bench(bench: dict, step: int = None, prefix: str = "  -- "):
    step_str = f"Step {step} | " if step is not None else ""
    bench_str = " | ".join(f"{k}: {v:.4f}" for k, v in bench.items())
    print(f"{prefix}{step_str}{bench_str}")


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint — format HF chuẩn (save_pretrained), KHÁC bản cũ
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, tokenizer=None, optimizer=None, scheduler=None,
                     global_step: int = 0, chunk_idx: int = 0, val_loss=None):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)

    train_state = {"global_step": global_step, "chunk_idx": chunk_idx, "val_loss": val_loss}
    with open(os.path.join(path, "train_state.json"), "w") as f:
        json.dump(train_state, f)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

    print(f"  ✓ Saved checkpoint (HF format) → {path}")


def load_train_state(path) -> dict:
    state_path = os.path.join(path, "train_state.json")
    if not os.path.exists(state_path):
        return {"global_step": 0, "chunk_idx": 0, "val_loss": None}
    with open(state_path) as f:
        return json.load(f)


def load_optimizer_scheduler(path, optimizer=None, scheduler=None, device="cpu"):
    opt_path = os.path.join(path, "optimizer.pt")
    sch_path = os.path.join(path, "scheduler.pt")
    if optimizer is not None and os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    if scheduler is not None and os.path.exists(sch_path):
        scheduler.load_state_dict(torch.load(sch_path, map_location=device))


# ══════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════

class LlamaPretrainTrainer:
    def __init__(self, cfg, model, tokenizer):
        self.cfg       = cfg
        self.model     = model
        self.tokenizer = tokenizer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
        )
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95),
        )

        self.use_bf16 = (
            cfg.train.bf16 and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda" and cfg.train.mixed_precision and not self.use_bf16),
        )

        self.global_step   = 0
        self.chunk_idx      = 0
        self.best_val_loss   = float("inf")

        self.logger = TrainLogger(log_every=cfg.train.log_every)
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Warmup tuyến tính → cosine decay theo chu kỳ (SGDR) — giữ nguyên convention cũ."""
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

    def _autocast(self):
        if self.use_bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return torch.amp.autocast(
            "cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
        )

    def train_one_batch(self, batch: dict, accum_step: int) -> float:
        ids    = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        mask   = batch["attention_mask"].to(self.device)

        with self._autocast():
            out  = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            loss = self.compute_loss(out.logits, labels)

        scaled = loss / self.cfg.train.grad_accum
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (accum_step + 1) % self.cfg.train.grad_accum == 0:
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
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
            mask   = batch["attention_mask"].to(self.device)
            out    = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            total += self.compute_loss(out.logits, labels).item()
            n     += 1

        self.model.train()
        return total / max(n, 1)

    def train_one_chunk(self, train_loader, val_loader, chunk_idx: int) -> float:
        self.model.train()
        self.chunk_idx = chunk_idx

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
                        f"{self.cfg.train.save_dir}/best", self.model, self.tokenizer,
                        self.optimizer, self.scheduler, self.global_step, self.chunk_idx, val_loss,
                    )

                bench = run_llama_benchmark(self.model, self.tokenizer, self.cfg, verbose=False)
                self.model.train()
                log_bench(bench, step=self.global_step)

            if self.global_step > 0 and self.global_step % self.cfg.train.save_every == 0:
                save_checkpoint(
                    f"{self.cfg.train.save_dir}/step_{self.global_step}", self.model,
                    self.tokenizer, self.optimizer, self.scheduler, self.global_step, self.chunk_idx,
                )

        val_loss = self.evaluate(val_loader)
        log_eval(val_loss, step=self.global_step, prefix="  [Chunk end] ")

        bench = run_llama_benchmark(self.model, self.tokenizer, self.cfg, verbose=False)
        log_bench(bench, step=self.global_step)
        return val_loss