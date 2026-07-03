"""
utils/logger.py — Logging chuẩn cho training
================================================
Sửa bug đã phát hiện: loss log trước đây tính sai vì cộng dồn
loss của nhiều segment nhưng chia cho số gradient-step.
Logger này tách riêng "segment count" và "optimizer step count".
"""

import math
import time


class TrainLogger:
    """
    Theo dõi loss CHÍNH XÁC theo segment, không lẫn với optimizer step.

    Usage:
        logger = TrainLogger(log_every=100)
        for segment in segments:
            loss = ...
            logger.update(loss.item())
            if logger.should_log():
                logger.flush(step=global_step, lr=current_lr)
    """

    def __init__(self, log_every: int = 100):
        self.log_every    = log_every
        self.loss_sum      = 0.0
        self.count          = 0
        self.t_start          = time.time()

    def update(self, loss_value: float):
        """Gọi sau mỗi segment forward/backward (loss CHƯA chia grad_accum)."""
        self.loss_sum += loss_value
        self.count    += 1

    def should_log(self) -> bool:
        return self.count >= self.log_every

    def avg_loss(self) -> float:
        return self.loss_sum / max(self.count, 1)

    def flush(self, step: int, lr: float, prefix: str = "") -> dict:
        """In log và reset counter. Trả về dict các giá trị để caller dùng thêm nếu cần."""
        avg  = self.avg_loss()
        ppl  = math.exp(min(avg, 20))
        elapsed = time.time() - self.t_start

        print(
            f"{prefix}Step {step:>7} | "
            f"loss: {avg:.4f} | ppl: {ppl:.2f} | "
            f"lr: {lr:.2e} | {elapsed:.0f}s"
        )

        result = {"loss": avg, "ppl": ppl, "elapsed": elapsed}

        self.loss_sum = 0.0
        self.count    = 0
        self.t_start  = time.time()

        return result


def log_eval(val_loss: float, step: int = None, prefix: str = "  -- "):
    ppl = math.exp(min(val_loss, 20))
    step_str = f"Step {step} | " if step is not None else ""
    print(f"{prefix}{step_str}Val loss: {val_loss:.4f} | ppl: {ppl:.2f}")
    
def log_bench(bench: dict, step: int = None, prefix: str = "  -- "):
    step_str = f"Step {step} | " if step is not None else ""
    bench_str = " | ".join(f"{k}: {v:.4f}" for k, v in bench.items())
    print(f"{prefix}{step_str}{bench_str}")
