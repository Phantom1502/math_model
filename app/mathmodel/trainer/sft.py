"""
trainer/sft.py — Supervised Fine-Tuning (Stage 1)
====================================================
Khác PretrainTrainer: dataset SFT nhỏ (~7K sample, load hết vào RAM một
lần), không cần logic CHUNK STREAMING của BaseTrainer.train_one_chunk
(logic đó sinh ra để xử lý dataset lớn không load hết vào RAM được).

Vì vậy KHÔNG dùng train_one_chunk() — tự viết loop epoch riêng. Vẫn tái
dùng _run_accum_window() (gom nhiều micro-batch thành 1 cửa sổ gradient
accumulation, tự tính total_valid_tokens cho cả cửa sổ rồi mới gọi
train_one_batch) — KHÔNG gọi train_one_batch() trực tiếp vì hàm đó cần
total_valid_tokens của CẢ cửa sổ (không phải của riêng 1 batch) để chuẩn
hóa loss đúng (xem comment trong trainer/base.py).

KHÔNG gọi benchmark.run_all(): bộ benchmark đó là các câu hỏi TIẾNG VIỆT
(semantic/entity/fact/ood), không đo được tiến độ SFT tiếng Anh trên
GSM8K — gọi vào sẽ tốn compute mà cho số liệu không phản ánh đúng việc
đang làm. Đo tiến độ SFT bằng val_loss (đã đủ vì đây là SFT, chưa phải
Stage 4 nơi cần đo accuracy giải toán thật) — muốn xem chất lượng CoT
trực tiếp thì dùng generate.py với checkpoint sft_best.pt.
"""

from .base import BaseTrainer
from utils import log_eval, save_checkpoint, load_checkpoint


class SFTTrainer(BaseTrainer):
    """Kế thừa nguyên BaseTrainer — compute_loss mặc định (CE, ignore_index=-100)
    đã đúng vì labels đã được SFTDataset mask sẵn phần prompt."""
    pass


def _maybe_save_best(trainer: SFTTrainer, val_loss: float, cfg) -> bool:
    """Lưu sft_best.pt nếu val_loss hiện tại thấp nhất từ trước tới giờ.
    Tách riêng vì cần gọi ở CẢ 2 chỗ: giữa epoch (theo eval_every) và
    cuối mỗi epoch (dataset SFT nhỏ nên số step/epoch có thể ít hơn
    eval_every rất nhiều — nếu chỉ check ở nhánh eval_every, best sẽ
    gần như không bao giờ được lưu với dataset nhỏ)."""
    if val_loss < trainer.best_val_loss:
        trainer.best_val_loss = val_loss
        save_checkpoint(
            f"{cfg.train.save_dir}/sft_best.pt",
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx=0, val_loss=val_loss,
            model_cfg=cfg.model,
        )
        return True
    return False


def _run_window_correct_log(trainer: SFTTrainer, batches: list) -> None:
    """
    Giống hệt BaseTrainer._run_accum_window() ở phần backward/optimizer step
    (KHÔNG đổi gì — gradient vẫn đúng, không cần sửa, không đụng base.py).

    KHÁC ở chỗ: KHÔNG dùng thẳng giá trị loss train_one_batch() trả về để
    log. train_one_batch() trả về loss_sum_của_batch_NÀY / total_valid_tokens
    _của_CẢ_cửa_sổ — đúng cho backward (tổng theo cửa sổ ra gradient chuẩn),
    nhưng SAI khi log trực tiếp: con số này bị "pha loãng" theo tỷ lệ
    (tổng token cả cửa sổ ÷ token riêng batch này) — với grad_accum=64, số
    hiển thị nhỏ hơn thật ~64 lần, khiến loss train trông như hội tụ về 0
    trong khi thực ra không phải vậy.

    Sửa: nhân NGƯỢC lại với total_valid_tokens (khôi phục loss_sum gốc của
    riêng batch này), rồi chia cho SỐ TOKEN CỦA RIÊNG BATCH ĐÓ — ra đúng
    loss trung bình/token của batch này, log giá trị ĐÃ SỬA này.
    """
    total_valid_tokens = sum((b["labels"] != -100).sum().item() for b in batches)
    total_valid_tokens = max(total_valid_tokens, 1)

    for i, batch in enumerate(batches):
        is_last = (i == len(batches) - 1)
        returned_loss = trainer.train_one_batch(batch, total_valid_tokens, is_last)

        own_valid_tokens = max((batch["labels"] != -100).sum().item(), 1)
        true_loss = returned_loss * total_valid_tokens / own_valid_tokens   # = loss_sum_batch_này / own_valid_tokens

        trainer.logger.update(true_loss)
        if trainer.logger.should_log():
            lr = trainer.scheduler.get_last_lr()[0]
            trainer.logger.flush(step=trainer.global_step, lr=lr)


def run_sft(
    cfg,
    model,
    tokenizer,
    train_loader,
    val_loader,
    n_epochs       : int = 3,
    init_checkpoint: str = None,
) -> SFTTrainer:
    """
    Args:
        init_checkpoint: path checkpoint PRETRAINED (không phải checkpoint SFT
                         cũ nếu resume) — chỉ load model weight, KHÔNG load
                         optimizer/scheduler, vì SFT là giai đoạn train mới
                         với lr/scheduler riêng (thường nhỏ hơn lr pretrain).
    """
    trainer = SFTTrainer(cfg, model, tokenizer)

    if init_checkpoint:
        print(f"\nKhởi tạo model từ pretrained checkpoint: {init_checkpoint}")
        load_checkpoint(init_checkpoint, trainer.model, device=trainer.device)
        print("  (chỉ load model weight — optimizer/scheduler bắt đầu MỚI cho SFT)")

    print(f"\n{'='*60}")
    print(f"  SFT — {n_epochs} epoch(s) trên {len(train_loader.dataset)} samples")
    print(f"{'='*60}")

    grad_accum = cfg.train.grad_accum
    val_loss   = float("inf")

    for epoch in range(n_epochs):
        trainer.model.train()
        print(f"\n── Epoch {epoch + 1}/{n_epochs} ──")

        accum_buffer = []

        for batch in train_loader:
            accum_buffer.append(batch)

            if len(accum_buffer) == grad_accum:
                trainer._run_accum_window(accum_buffer)
                accum_buffer = []

                if trainer.global_step > 0 and trainer.global_step % cfg.train.eval_every == 0:
                    val_loss = trainer.evaluate(val_loader)
                    log_eval(val_loss, step=trainer.global_step)
                    _maybe_save_best(trainer, val_loss, cfg)

                if trainer.global_step > 0 and trainer.global_step % cfg.train.save_every == 0:
                    save_checkpoint(
                        f"{cfg.train.save_dir}/sft_step_{trainer.global_step}.pt",
                        trainer.model, trainer.optimizer, trainer.scheduler,
                        trainer.global_step, chunk_idx=0,
                        model_cfg=cfg.model,
                    )

        # Cửa sổ cuối epoch: số batch còn lại có thể ngắn hơn grad_accum,
        # vẫn chạy nốt để không bỏ sót dữ liệu/gradient (giống train_one_chunk).
        if accum_buffer:
            trainer._run_accum_window(accum_buffer)
            accum_buffer = []

        val_loss = trainer.evaluate(val_loader)
        log_eval(val_loss, step=trainer.global_step, prefix=f"  [Epoch {epoch + 1} end] ")
        if _maybe_save_best(trainer, val_loss, cfg):
            print(f"  ✓ Cập nhật sft_best.pt (val_loss={val_loss:.4f})")

    save_checkpoint(
        f"{cfg.train.save_dir}/sft_final.pt",
        trainer.model, trainer.optimizer, trainer.scheduler,
        trainer.global_step, chunk_idx=0, val_loss=val_loss,
        model_cfg=cfg.model,
    )
    print(f"\n✓ SFT hoàn tất. Checkpoint cuối: {cfg.train.save_dir}/sft_final.pt")
    print(f"  Checkpoint tốt nhất (val_loss thấp nhất): {cfg.train.save_dir}/sft_best.pt")
    return trainer