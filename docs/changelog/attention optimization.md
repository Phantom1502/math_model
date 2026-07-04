# Changelog: Tối ưu hoá `SelfAttentionRoPE` + Gradient Checkpointing

## 1. Bối cảnh

Refactor `SelfAttentionRoPE` (attention thủ công) sang dùng `F.scaled_dot_product_attention` (SDPA/FlashAttention), đồng thời bọc `TransformerBlock` bằng `torch.utils.checkpoint`. Mục tiêu: giảm VRAM để có thể tăng batch size khi train trên GPU giới hạn (Colab T4/A100/L4).

---

## 2. Các thay đổi chính

### 2.1. Thay matmul thủ công bằng `F.scaled_dot_product_attention`

**Trước:**
```python
scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
if attn_mask is not None:
    scores = scores + attn_mask
weights = self.dropout(F.softmax(scores, dim=-1))
out = torch.matmul(weights, v)
```

**Sau:**
```python
out = F.scaled_dot_product_attention(
    query=q, key=k, value=v,
    attn_mask=attn_mask,
    dropout_p=dropout_p,
    is_causal=False,
)
```

**Nguyên nhân:**
- SDPA dùng kernel Flash-Attention/mem-efficient (khi backend hỗ trợ) → không cần vật chất hoá ma trận `scores` kích thước `(B, h, T, T)` trong VRAM, giảm đáng kể bộ nhớ đỉnh, đặc biệt với sequence dài.
- Về mặt toán học, scale mặc định của SDPA là `1/sqrt(d_head)`, đúng bằng `self.scale` cũ → kết quả tương đương, không đổi hành vi model.
- `attn_mask` dạng additive-float vẫn tương thích trực tiếp, không cần đổi convention masking ở nơi gọi.

### 2.2. Thêm Gradient Checkpointing ở `TransformerBlock`

```python
def _forward_block(hidden_states):
    attn_out = self.attention(hidden_states, freqs_cis, attn_mask)
    x_residual = hidden_states + attn_out
    out = x_residual + self.feed_forward(x_residual)
    return out
return checkpoint(_forward_block, x, use_reentrant=False)
```

**Nguyên nhân:**
- Xoá activation trung gian trong forward, tính lại (recompute) khi backward → đổi thời gian train (+25–35%) lấy VRAM.
- Kết hợp với SDPA, tổng VRAM tiết kiệm được dùng để **tăng batch size**, giúp gradient ước lượng ổn định hơn và tận dụng GPU tốt hơn.
- `use_reentrant=False` được chọn đúng: backend non-reentrant tự lưu/khôi phục RNG state (dropout mask nhất quán giữa forward gốc và forward recompute), không cần liệt kê tường minh input trong `checkpoint(...)`.

---

## 3. Lỗi phát hiện trong bản refactor và cách sửa

### 3.1. 🔴 `dropout_p` bị "đóng băng" tại `__init__` — dropout không tắt khi eval

**Lỗi:**
```python
self.dropout_p = dropout if self.training else 0.0   # chạy 1 lần lúc khởi tạo
```
`self.training` luôn là `True` tại thời điểm `__init__` chạy, bất kể sau này gọi `model.eval()`. `eval()`/`train()` chỉ đổi cờ `self.training`, không chạy lại `__init__`.

**Hậu quả:** dropout vẫn áp dụng lúc inference/eval → validation loss bị nhiễu, output không deterministic, ảnh hưởng đến các quyết định dựa trên validation (early-stopping, chọn best checkpoint, LR scheduler theo plateau...).

**Sửa:** chỉ lưu `self.dropout = dropout` (giá trị thô), kiểm tra `self.training` **động** ngay trong `forward`:
```python
dropout_p = self.dropout if self.training else 0.0
```

### 3.2. 🔴 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` hard-code cục bộ trong attention

**Vấn đề:**
- Hard-code `"cuda"` → crash nếu chạy trên CPU/MPS.
- Output SDPA bên trong `with autocast(...)` là `bfloat16`; ra khỏi `with`, `self.Wo(out)` chạy với `Wo.weight` dạng `float32` → dễ raise `RuntimeError` do dtype mismatch, trừ khi toàn bộ training loop bên ngoài đã bọc autocast toàn cục (lúc đó autocast cục bộ này trở thành dư thừa).
- Mâu thuẫn với comment gốc ("T4 dùng float16, A100/L4 dùng bfloat16") nhưng code chỉ hard-code `bfloat16`.

**Sửa (khuyến nghị):** bỏ hẳn autocast cục bộ, quản lý mixed-precision ở training loop (bọc toàn bộ `model(x)`), đảm bảo mọi module (RMSNorm, Linear, FFN, Attention) đồng nhất dtype.

Nếu vẫn muốn autocast cục bộ để ép chạy Flash kernel mà không sửa training loop, cần ép dtype output về lại dtype gốc trước khi vào `Wo`:
```python
in_dtype = q.dtype
with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                          dropout_p=dropout_p, is_causal=False)
out = out.to(in_dtype)
```

---

## 4. Tương thích với checkpoint đã train trước đó

- Tên submodule (`Wq, Wk, Wv, Wo, norm`) và shape parameter **không đổi** → `load_state_dict()` hoạt động bình thường, không cần train lại từ đầu.
- `nn.Dropout` module cũ vốn không có parameter, việc thay bằng `float dropout_p` không ảnh hưởng đến việc load checkpoint.
- Optimizer state (Adam m/v) resume bình thường vì shape param giữ nguyên.

### Rủi ro khi resume cần lưu ý
- **Bắt buộc fix mục 3.2** trước khi resume — nếu không sẽ crash ngay step đầu (dtype mismatch) khi training loop chưa có autocast toàn cục.
- **Loss có thể spike nhẹ** ở vài trăm step đầu sau resume do đổi thuật toán kernel (SDPA dùng online-softmax/tiling khác matmul thủ công) và/hoặc đổi precision — không phải divergence nghiêm trọng, thường tự ổn định lại. Nếu spike lớn, có thể giảm LR tạm thời rồi tăng lại theo schedule.
- Nếu đã từng dùng validation loss (đo lúc `model.eval()`) từ **trước khi fix mục 3.1** để quyết định early-stopping/chọn checkpoint, các số liệu đó không đáng tin cậy — chỉ nên tính từ sau khi patch dropout.

---

## 5. Tăng batch size nhờ VRAM tiết kiệm được — các điểm cần khớp lại

VRAM tiết kiệm từ SDPA + checkpointing chủ yếu đến từ giảm activation, cho phép tăng batch size. Khi thực hiện điều này lúc resume, cần lưu ý:

1. **LR nên tăng theo batch size** (linear scaling rule cho SGD, hoặc sqrt-scaling cho Adam/AdamW) nếu tăng batch size vật lý mà không giữ effective batch size cố định. Nên tăng dần, tránh nhảy LR đột ngột giữa quá trình resume.
2. **Phân biệt batch size vật lý và effective batch size** (= batch size vật lý × gradient accumulation steps). Cách an toàn nhất: tăng batch size vật lý và **giảm accumulation steps tương ứng** để giữ effective batch size không đổi — không cần đổi LR/schedule.
3. **Optimizer state (Adam m/v) không gắn với batch size cụ thể**, resume vẫn hợp lệ, nhưng cần vài trăm step để thích nghi lại với gradient noise mới nếu batch size thực sự đổi.
4. **Checkpointing chỉ giảm VRAM cho activation**, không giảm VRAM cho optimizer state/model weights. Với model lớn (VRAM bị optimizer state chiếm nhiều), mức tăng batch size khả dụng có thể ít hơn kỳ vọng — nên tăng dần và theo dõi `nvidia-smi` thực tế thay vì nhảy thẳng lên batch size lớn.
5. Khuyến nghị: tăng batch size **dần dần** (x1.5–2 mỗi lần) và giữ effective batch size cố định trong lần thử đầu tiên, để tách riêng ảnh hưởng của "đổi attention kernel" và "đổi batch size" khi theo dõi loss curve.

---

## 6. Tóm tắt

| Hạng mục | Trạng thái |
|---|---|
| SDPA thay matmul thủ công | ✅ Đúng hướng, tương đương toán học |
| Gradient checkpointing | ✅ Trade-off hợp lý (VRAM ↔ compute) |
| Bug dropout_p tính tĩnh ở `__init__` | 🔴 Đã xác định, cần patch (kiểm tra `self.training` động trong `forward`) |
| Autocast cục bộ hard-code `cuda`/`bfloat16` | 🔴 Đã xác định, cần patch (bỏ hoặc ép dtype output) |
| Khả năng continue-train từ checkpoint cũ | ✅ Khả thi sau khi patch 2 lỗi trên, không cần train lại từ đầu |
| Tăng batch size nhờ VRAM tiết kiệm | ✅ Hợp lý, cần khớp lại LR/accumulation steps khi resume |