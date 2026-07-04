# Bug Report: Sai lệch trọng số loss khi dùng Gradient Accumulation

## Tóm tắt

Khi thay đổi cấu hình `micro_batch_size` / `grad_accum` nhưng giữ nguyên **effective batch size** (`micro_batch_size × grad_accum = const`), kết quả training vẫn khác nhau đáng kể (dao động loss/gradient noise, độ ổn định training). Nguyên nhân: `compute_loss` dùng `reduction="mean"` của `F.cross_entropy`, khiến mỗi micro-batch được weight sai khi cộng dồn gradient qua các bước accumulation.

## Môi trường phát hiện

- File: `trainer/base.py`
- Hàm liên quan: `BaseTrainer.compute_loss`, `BaseTrainer.train_one_batch`
- Quan sát: micro-batch = 8, grad_accum = 64 → acc thấp hơn, training nhiễu hơn so với micro-batch = 32, grad_accum = 16 (cùng effective batch = 512).

## Nguyên nhân gốc

```python
def compute_loss(self, logits, labels):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
```

`F.cross_entropy` mặc định `reduction="mean"` — chia loss cho **số token hợp lệ trong micro-batch hiện tại**, không phải theo tổng effective batch.

```python
self.scaler.scale(loss / self.cfg.train.grad_accum).backward()
```

Gradient cuối cùng sau khi accumulate tương đương:

```
grad ∝ (1 / grad_accum) * Σ_i  mean_loss_i
     = (1 / grad_accum) * Σ_i  ( Σ token_loss trong batch_i / num_valid_tokens_i )
```

Đây là **trung bình của các trung bình**, gán trọng số bằng nhau (`1/grad_accum`) cho mỗi micro-batch bất kể số token hợp lệ (`num_valid_tokens_i`) thực tế khác nhau (do padding / `ignore_index=-100`).

### Vì sao ảnh hưởng thay đổi theo micro-batch size

- `micro_batch_size` càng nhỏ → phương sai của `num_valid_tokens_i` giữa các micro-batch càng cao (mẫu nhỏ, ít trung bình hóa nội bộ) → gradient cuối lệch trọng số nhiều hơn giữa các lần cộng dồn.
- `micro_batch_size` càng lớn → mỗi micro-batch tự trung bình hóa qua nhiều sample hơn → `num_valid_tokens_i` ổn định hơn giữa các bước accumulate → gradient "sạch" hơn.

→ Cùng effective batch size nhưng cấu hình micro-batch khác nhau cho kết quả training khác nhau một cách hệ thống, không phải nhiễu ngẫu nhiên.

## Ảnh hưởng

- Gradient bị lệch trọng số theo token mỗi khi độ dài sequence trong batch không đồng đều (padding biến thiên).
- Ảnh hưởng tăng dần khi `grad_accum` lớn và `micro_batch_size` nhỏ.
- Không gây lỗi runtime, không thể hiện qua log loss thông thường — chỉ biểu hiện gián tiếp qua độ ổn định training / accuracy cuối cùng, nên rất dễ bị bỏ qua khi debug.

## Cách khắc phục

Đổi `reduction="sum"` và chuẩn hóa theo **tổng số token hợp lệ trên toàn effective batch**, thay vì chia đều theo `grad_accum`.

```python
def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
```

```python
def train_one_batch(self, batch, accum_step, total_valid_tokens):
    ids    = batch["input_ids"].to(self.device)
    labels = batch["labels"].to(self.device)
    mask   = causal_mask(ids.shape[1], self.device)

    with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda" and self.cfg.train.mixed_precision)):
        logits   = self.model(ids, attn_mask=mask)
        loss_sum = self.compute_loss(logits, labels)  # reduction="sum"

    self.scaler.scale(loss_sum / total_valid_tokens).backward()

    if (accum_step + 1) % self.cfg.train.grad_accum == 0:
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

    return loss_sum.item() / total_valid_tokens
```

`total_valid_tokens` = tổng số token khác `-100` trên toàn bộ cửa sổ accumulation (tất cả micro-batch trong 1 lần optimizer step), không phải chỉ một micro-batch.

### Hai cách tính `total_valid_tokens`

1. **Pass đếm trước (chính xác nhất):** trước khi vào loop accumulate, duyệt qua các micro-batch sắp dùng và cộng `(labels != -100).sum()`. Cách này được HuggingFace `Trainer` và Unsloth dùng.
2. **Ước lượng nhanh (kém chính xác hơn):** nếu độ dài sequence tương đối đồng đều / ít padding, có thể dùng gần đúng `grad_accum * micro_batch_size * seq_len`, chấp nhận sai số nhỏ để tránh thêm 1 pass.

### Triển khai đầy đủ trong vòng lặp training

`train_loader` được duyệt tuần tự từng micro-batch một, nên để đếm trước `total_valid_tokens` cho *cả cửa sổ accumulation*, cần gom (buffer) đủ `grad_accum` micro-batch lại trước khi tính, thay vì tính rải rác theo `accum_step % grad_accum` như bản cũ. Điểm trigger optimizer step cũng đổi theo: dựa vào việc buffer đã đầy (hoặc hết dữ liệu ở cuối epoch), thay vì dựa vào chỉ số tuyệt đối `accum_step`. Chi tiết code xem trong `trainer/base.py` đã cập nhật (`train_one_batch`, `_run_accum_window`, `train_one_chunk`).

**Lưu ý quan trọng về cửa sổ cuối epoch:** nếu `len(train_loader) % grad_accum != 0`, cửa sổ accumulation cuối cùng của epoch sẽ ngắn hơn bình thường. Điều này không gây sai lệch trọng số vì `total_valid_tokens` vẫn được tính đúng theo đúng số micro-batch thực tế có trong cửa sổ đó — chỉ là effective batch size của riêng bước optimizer step cuối epoch nhỏ hơn các bước khác, tương tự "drop_last=False" trong DataLoader thông thường.

**Lưu ý về memory:** buffer chỉ giữ dữ liệu thô (`input_ids`, `labels`) của `grad_accum` micro-batch, không giữ activation/gradient — vì `backward()` vẫn được gọi ngay trong `train_one_batch` cho từng micro-batch một (không đợi cả cửa sổ mới backward). Overhead memory thêm không đáng kể so với bản cũ.

## Việc cần làm (checklist)

- [ ] Đổi `reduction` trong `compute_loss` từ `mean` → `sum`.
- [ ] Thêm cơ chế đếm `total_valid_tokens` theo cửa sổ accumulation (trong dataloader hoặc trong `train_one_chunk`).
- [ ] Sửa `train_one_batch` để nhận `total_valid_tokens` và chuẩn hóa loss theo giá trị này thay vì `grad_accum`.
- [ ] Cập nhật lại giá trị trả về của `train_one_batch` để log đúng loss trung bình theo token (tránh log sai do đổi sang `sum`).
- [ ] Kiểm tra các subclass override `compute_loss` (SFT/DPO) có bị ảnh hưởng tương tự không.
- [ ] Chạy lại thử nghiệm so sánh (8, 64) vs (32, 16) sau khi fix để xác nhận kết quả hội tụ giống nhau.

---

# Changelog

## [Unreleased] — Fix: gradient accumulation loss weighting bug

### Fixed
- **`trainer/base.py`**: `compute_loss` trước đây dùng `reduction="mean"` (mặc định của `F.cross_entropy`), khiến mỗi micro-batch trong gradient accumulation được weight đều theo `1/grad_accum` bất kể số token hợp lệ thực tế khác nhau giữa các micro-batch (do padding / `ignore_index=-100`). Điều này làm cho các cấu hình `(micro_batch_size, grad_accum)` khác nhau cho cùng effective batch size vẫn cho kết quả training khác nhau một cách hệ thống — cấu hình micro-batch nhỏ hơn (accumulate nhiều bước hơn) bị nhiễu gradient nhiều hơn.
- Đổi `compute_loss` sang `reduction="sum"`, chuẩn hóa loss theo tổng số token hợp lệ trên toàn effective batch thay vì chia đều theo số bước accumulation trong `train_one_batch`.

### Changed
- `train_one_batch` nhận thêm tham số `total_valid_tokens` (tổng token hợp lệ của cả cửa sổ accumulation) để chuẩn hóa loss chính xác trước khi `backward()`.
- Giá trị loss trả về từ `train_one_batch` được tính lại (`loss_sum / total_valid_tokens`) để log đúng loss trung bình theo token, tránh log sai lệch do đổi `reduction`.

### Impact
- Kết quả training với các cấu hình `(micro_batch_size, grad_accum)` khác nhau nhưng cùng effective batch size sẽ hội tụ nhất quán hơn, không còn phụ thuộc vào việc chia nhỏ batch như thế nào.
- Không ảnh hưởng đến effective batch size, learning rate schedule, hay optimizer step frequency — chỉ ảnh hưởng đến cách trọng số hóa loss/gradient giữa các micro-batch.