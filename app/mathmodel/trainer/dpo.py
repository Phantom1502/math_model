"""
trainer/dpo.py — Direct Preference Optimization (Stage 3b)
=============================================================================
L_DPO = -log σ( β * [ (logπ_θ(chosen|x) - logπ_ref(chosen|x))
                     - (logπ_θ(rejected|x) - logπ_ref(rejected|x)) ] )

π_ref : bản copy ĐÓNG BĂNG của model SFT — khởi tạo 1 lần, KHÔNG update.
π_θ   : model đang train — khởi tạo từ CÙNG checkpoint SFT.

KHÔNG dùng train_one_chunk/train_one_batch của BaseTrainer: loss DPO tính
trên TOÀN SEQUENCE (1 scalar/sample, dùng SUM log-prob), khác cấu trúc
per-token CE mà BaseTrainer.compute_loss giả định (cross-entropy + chuẩn
hóa theo số token của cửa sổ accumulation — không áp dụng được cho DPO).
Viết loop training riêng, tái dùng optimizer/scheduler/scaler đã setup sẵn
trong BaseTrainer.__init__.

Dataset cực nhỏ (13 pairs ở demo này) — không cần grad accumulation, mỗi
batch = 1 optimizer step, giữ code đơn giản dễ đọc. Không dùng TrainLogger
(thiết kế cho loss/ppl kiểu CE) — in trực tiếp loss/preference-accuracy/
margin, là các chỉ số phù hợp hơn để theo dõi DPO.
"""

import copy
import torch
import torch.nn.functional as F

from .base import BaseTrainer
from model import causal_mask
from utils import save_checkpoint, load_checkpoint


def sequence_logprob(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Tổng (SUM, không phải trung bình) log-prob của TOÀN sequence, chỉ tính
    trên vị trí labels != -100 (phần response — đã mask sẵn từ
    sft_dataset._build_example, tái dùng nguyên cho DPO).

    Khác benchmark.avg_logprob_per_token (chia theo độ dài) — công thức DPO
    gốc dùng SUM log-prob của toàn response, không normalize theo độ dài.

    Returns: tensor (B,)
    """
    T = input_ids.shape[1]
    mask = causal_mask(T, input_ids.device)
    logits = model(input_ids, attn_mask=mask)
    log_probs = F.log_softmax(logits, dim=-1)

    valid = (labels != -100)
    safe_labels = labels.clone()
    safe_labels[~valid] = 0   # tránh gather lỗi index âm/-100, sẽ bị nhân 0 ngay dưới

    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    gathered = gathered * valid
    return gathered.sum(dim=1)  # (B,)


class DPOTrainer(BaseTrainer):
    def __init__(self, cfg, model, tokenizer, beta: float = 0.1):
        super().__init__(cfg, model, tokenizer)
        self.beta = beta

        # π_ref: deepcopy TRƯỚC khi train bắt đầu — model lúc này đã được
        # load weight SFT ở run_dpo() (xem dưới), nên ref chính xác là bản SFT.
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def compute_dpo_loss(self, batch: dict) -> dict:
        c_ids    = batch["chosen_input_ids"].to(self.device)
        c_labels = batch["chosen_labels"].to(self.device)
        r_ids    = batch["rejected_input_ids"].to(self.device)
        r_labels = batch["rejected_labels"].to(self.device)

        logp_c_policy = sequence_logprob(self.model, c_ids, c_labels)
        logp_r_policy = sequence_logprob(self.model, r_ids, r_labels)

        with torch.no_grad():
            logp_c_ref = sequence_logprob(self.ref_model, c_ids, c_labels)
            logp_r_ref = sequence_logprob(self.ref_model, r_ids, r_labels)

        pi_logratios  = logp_c_policy - logp_r_policy
        ref_logratios = logp_c_ref    - logp_r_ref
        logits = self.beta * (pi_logratios - ref_logratios)

        loss = -F.logsigmoid(logits).mean()

        # pref_accuracy: % pair mà policy "nới rộng" khoảng cách ưu tiên
        # chosen>rejected SO VỚI ref — chỉ số chuẩn dùng để theo dõi DPO,
        # KHÔNG phải accuracy giải toán (đo việc đó ở Stage 4 riêng).
        pref_accuracy = (logits > 0).float().mean()
        margin        = (pi_logratios - ref_logratios).mean()

        return {"loss": loss, "pref_accuracy": pref_accuracy.item(), "margin": margin.item()}

    def train_one_batch_dpo(self, batch: dict) -> dict:
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
            out = self.compute_dpo_loss(batch)

        self.scaler.scale(out["loss"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

        return {"loss": out["loss"].item(), "pref_accuracy": out["pref_accuracy"], "margin": out["margin"]}


def run_dpo(
    cfg,
    model,
    tokenizer,
    train_loader,
    n_epochs       : int   = 10,
    beta           : float = 0.1,
    init_checkpoint: str   = None,
) -> DPOTrainer:
    """
    Args:
        init_checkpoint: checkpoint SFT (KHÔNG phải checkpoint DPO cũ nếu
                         resume) — PHẢI load weight vào `model` TRƯỚC khi
                         khởi tạo DPOTrainer, vì DPOTrainer.__init__ deepcopy
                         `model` hiện tại để làm ref_model — nếu load sau,
                         ref_model sẽ là bản random-init thay vì SFT.
    """
    if init_checkpoint:
        print(f"\nKhởi tạo model từ checkpoint SFT: {init_checkpoint}")
        load_checkpoint(init_checkpoint, model, device=cfg.train.device)

    trainer = DPOTrainer(cfg, model, tokenizer, beta=beta)
    print("  ✓ π_ref = bản đóng băng của model NGAY SAU khi load checkpoint SFT ở trên")

    print(f"\n{'='*60}")
    print(f"  DPO — {n_epochs} epoch(s) trên {len(train_loader.dataset)} pairs, beta={beta}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        trainer.model.train()
        ep_loss, ep_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            out = trainer.train_one_batch_dpo(batch)
            ep_loss   += out["loss"]
            ep_acc    += out["pref_accuracy"]
            n_batches += 1

            print(f"  Step {trainer.global_step:>4} | loss: {out['loss']:.4f} | "
                  f"pref_acc: {out['pref_accuracy']*100:5.1f}% | margin: {out['margin']:+.3f}")

        print(f"── Epoch {epoch+1}/{n_epochs} end | avg_loss: {ep_loss/n_batches:.4f} | "
              f"avg_pref_acc: {ep_acc/n_batches*100:.1f}% ──")

        save_checkpoint(
            f"{cfg.train.save_dir}/dpo_epoch_{epoch+1}.pt",
            trainer.model, trainer.optimizer, trainer.scheduler,
            trainer.global_step, chunk_idx=0,
            model_cfg=cfg.model,
        )

    save_checkpoint(
        f"{cfg.train.save_dir}/dpo_final.pt",
        trainer.model, trainer.optimizer, trainer.scheduler,
        trainer.global_step, chunk_idx=0,
        model_cfg=cfg.model,
    )
    print(f"\n✓ DPO hoàn tất. Checkpoint cuối: {cfg.train.save_dir}/dpo_final.pt")
    return trainer