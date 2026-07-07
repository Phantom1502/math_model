"""
trainer/listwise.py — Plackett-Luce / PRO Ranking Loss (Stage 3c)
=============================================================================
THAM CHIẾU: Self_correction_training.md, mục 2.5.

Mở rộng DPO (so sánh CẶP) sang XẾP HẠNG 3 MỨC trong MỘT lần tính loss, dùng
công thức Plackett-Luce (nền tảng của PRO — Preference Ranking Optimization):

    P(A > B > C) = [score(A) / (score(A)+score(B)+score(C))]
                 × [score(B) / (score(B)+score(C))]
    loss = -log P(đúng thứ tự)

Với r(y) = β * (logπ_θ(y|x) - logπ_ref(y|x))  — CÙNG công thức reward như
DPO — score(y) = exp(r(y)). Tính bằng logsumexp để ổn định số học, KHÔNG
tính score trực tiếp (dễ overflow vì là exp của sum log-prob cả sequence).

Thứ hạng CỐ ĐỊNH theo dữ liệu từ scripts/label_selfcorrect_data.py:
    A = tier_correct_straight  (rank 1 — tốt nhất, từ ground truth)
    B = tier_wrong_fixed       (rank 2 — sai nhưng nhận ra + sửa được)
    C = tier_wrong_raw         (rank 3 — sai và không sửa, rollout gốc)

Tái dùng NGUYÊN sequence_logprob() từ trainer/dpo.py — công thức log-prob
giống hệt DPO, chỉ khác cách tổng hợp thành loss (listwise thay vì pairwise
best-vs-worst, không bỏ phí tier ở giữa như DPO 2 pairs rời rạc).
"""

import copy
import torch

from .base import BaseTrainer
from .dpo import sequence_logprob
from utils import save_checkpoint, load_checkpoint


class ListwiseTrainer(BaseTrainer):
    def __init__(self, cfg, model, tokenizer, beta: float = 0.1):
        super().__init__(cfg, model, tokenizer)
        self.beta = beta

        # π_ref: deepcopy TRƯỚC khi train — model lúc này đã load weight SFT
        # ở run_listwise() (xem dưới), nên ref chính xác là bản SFT.
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def compute_listwise_loss(self, batch: dict) -> dict:
        rewards = []   # rewards[0]=A (correct_straight), [1]=B (wrong_fixed), [2]=C (wrong_raw)

        for r in range(3):
            ids    = batch[f"rank{r}_input_ids"].to(self.device)
            labels = batch[f"rank{r}_labels"].to(self.device)

            logp_policy = sequence_logprob(self.model, ids, labels)
            with torch.no_grad():
                logp_ref = sequence_logprob(self.ref_model, ids, labels)

            rewards.append(self.beta * (logp_policy - logp_ref))   # (B,)

        # log P(A>B>C) = [r_A - logsumexp(r_A,r_B,r_C)] + [r_B - logsumexp(r_B,r_C)]
        stack_abc = torch.stack(rewards, dim=0)          # (3, B)
        stack_bc  = torch.stack(rewards[1:], dim=0)       # (2, B)

        log_p1 = rewards[0] - torch.logsumexp(stack_abc, dim=0)
        log_p2 = rewards[1] - torch.logsumexp(stack_bc,  dim=0)
        log_p  = log_p1 + log_p2

        loss = -log_p.mean()

        # Diagnostic: % bài mà policy xếp ĐÚNG thứ tự reward A>B>C
        order_accuracy = ((rewards[0] > rewards[1]) & (rewards[1] > rewards[2])).float().mean()

        return {"loss": loss, "order_accuracy": order_accuracy.item()}

    def train_one_batch_listwise(self, batch: dict) -> dict:
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            out = self.compute_listwise_loss(batch)

        self.scaler.scale(out["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

        return {"loss": out["loss"].item(), "order_accuracy": out["order_accuracy"]}

    @torch.no_grad()
    def evaluate_listwise(self, val_loader) -> dict:
        """Tính avg_loss/avg_order_acc trên val set — KHÔNG backward, KHÔNG
        update ref_model/policy. Dùng để phát hiện overfit (so với train loss)."""
        self.model.eval()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for batch in val_loader:
            out = self.compute_listwise_loss(batch)
            total_loss += out["loss"].item()
            total_acc  += out["order_accuracy"]
            n_batches  += 1

        self.model.train()
        n_batches = max(n_batches, 1)
        return {"loss": total_loss / n_batches, "order_accuracy": total_acc / n_batches}


def run_listwise(
    cfg,
    model,
    tokenizer,
    train_loader,
    val_loader,
    n_epochs       : int   = 10,
    beta           : float = 0.1,
    init_checkpoint: str   = None,
) -> ListwiseTrainer:
    """
    Args:
        init_checkpoint: checkpoint SFT — PHẢI load weight vào `model` TRƯỚC
                         khi khởi tạo ListwiseTrainer (deepcopy làm ref_model
                         xảy ra trong __init__), giống hệt lý do ở run_dpo().
    """
    if init_checkpoint:
        print(f"\nKhởi tạo model từ checkpoint: {init_checkpoint}")
        load_checkpoint(init_checkpoint, model, device=cfg.train.device)

    trainer = ListwiseTrainer(cfg, model, tokenizer, beta=beta)
    print("  ✓ π_ref = bản đóng băng của model NGAY SAU khi load checkpoint ở trên")

    print(f"\n{'='*60}")
    print(f"  Listwise (Plackett-Luce) — {n_epochs} epoch(s) trên "
          f"{len(train_loader.dataset)} bài train / {len(val_loader.dataset)} bài val, beta={beta}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        trainer.model.train()
        ep_loss, ep_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            out = trainer.train_one_batch_listwise(batch)
            ep_loss   += out["loss"]
            ep_acc    += out["order_accuracy"]
            n_batches += 1

            print(f"  Step {trainer.global_step:>4} | loss: {out['loss']:.4f} | "
                  f"order_acc: {out['order_accuracy']*100:5.1f}%")

        val_out = trainer.evaluate_listwise(val_loader)
        print(f"── Epoch {epoch+1}/{n_epochs} end | "
              f"train_loss: {ep_loss/n_batches:.4f} | train_order_acc: {ep_acc/n_batches*100:.1f}% | "
              f"VAL_loss: {val_out['loss']:.4f} | VAL_order_acc: {val_out['order_accuracy']*100:.1f}% ──")

        if val_out["loss"] < trainer.best_val_loss:
            trainer.best_val_loss = val_out["loss"]
            save_checkpoint(
                f"{cfg.train.save_dir}/listwise_best.pt",
                trainer.model, trainer.optimizer, trainer.scheduler,
                trainer.global_step, chunk_idx=0, val_loss=val_out["loss"],
                model_cfg=cfg.model,
            )
            print(f"  ✓ Cập nhật listwise_best.pt (val_loss={val_out['loss']:.4f})")
        else:
            print(f"  (val_loss không cải thiện — best hiện tại: {trainer.best_val_loss:.4f}, "
                  f"có thể là dấu hiệu bắt đầu overfit nếu train_loss vẫn tiếp tục giảm)")

        save_checkpoint(
            f"{cfg.train.save_dir}/listwise_epoch_{epoch+1}.pt",
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx=0, val_loss=val_out["loss"],
            model_cfg=cfg.model,
        )

    save_checkpoint(
        f"{cfg.train.save_dir}/listwise_final.pt",
        trainer.model, trainer.optimizer, trainer.scheduler,
        trainer.global_step, chunk_idx=0,
        model_cfg=cfg.model,
    )
    print(f"\n✓ Listwise training hoàn tất.")
    print(f"  Checkpoint cuối       : {cfg.train.save_dir}/listwise_final.pt")
    print(f"  Checkpoint tốt nhất (val_loss thấp nhất): {cfg.train.save_dir}/listwise_best.pt")
    return trainer