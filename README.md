# Spec: Dạy model tự viết suy luận toán học qua SFT → PRM → DPO

## 1. Bối cảnh hiện tại

- Model: tự pretrain, framework PyTorch thuần (không dùng HuggingFace `transformers`/`trl`).
- Năng lực hiện tại: đọc/viết cơ bản ổn, giải toán còn yếu (chưa có format suy luận từng bước rõ ràng, độ chính xác thấp).
- Mục tiêu dự án: **demo cá nhân để hiểu luồng** SFT → gán nhãn PRM → RL/DPO, có kết quả cuối là model cải thiện được khả năng giải toán và có thể quan sát được quá trình suy luận (CoT) của nó.

## 2. Mục tiêu tổng thể (definition of done)

- [ ] Model sinh được lời giải có cấu trúc từng bước rõ ràng cho bài toán word-problem cơ bản (cộng/trừ/nhân/chia nhiều bước, kiểu GSM8K).
- [ ] Có một cơ chế chấm điểm từng bước (PRM) hoạt động được, không cần gọi API ngoài liên tục.
- [ ] Model sau DPO có accuracy cao hơn model chỉ-SFT trên cùng một test set giữ riêng (held-out).
- [ ] Có công cụ (script/log) để "nhìn" được CoT của model qua các checkpoint, so sánh trước/sau từng giai đoạn.

Không mục tiêu (out of scope):
- Không cần đạt SOTA hay so sánh với model lớn.
- Không cần PPO/GRPO đầy đủ ở lần đầu — DPO là đủ để chứng minh luồng.
- Không cần multi-task ngoài toán (giữ phạm vi hẹp: word problems số học).

## 3. Kiến trúc pipeline (4 giai đoạn)

```
[Stage 0] Model pretrained hiện có
      │
[Stage 1] SFT — dạy format CoT
      │
[Stage 2] Self-sampling + gán nhãn step-level (PRM data)
      │
[Stage 3] Build preference pairs → DPO training
      │
[Stage 4] Evaluation + quan sát CoT
```

---

### Stage 1 — SFT (dạy format, chưa cần giỏi toán)

**Input**: dataset word-problem có lời giải từng bước.
- Nguồn gợi ý: GSM8K (tiếng Anh, ~7.5K train) hoặc dịch sang tiếng Việt nếu muốn CoT tiếng Việt. Quy mô vài nghìn mẫu là đủ cho SFT format.

**Format chuẩn hoá** (cố định để các stage sau dùng chung):
```
<problem>
{đề bài}
</problem>
<solution>
Bước 1: {nội dung bước 1}
Bước 2: {nội dung bước 2}
...
Đáp số: {giá trị số}
</solution>
```
Lý do dùng tag rõ ràng thay vì chỉ "Bước n:" như bản demo trước: dễ parse bằng regex/string split trong PyTorch thuần, không phụ thuộc model tuân thủ định dạng tự nhiên ngôn ngữ.

**Việc cần làm**:
- Viết script chuẩn hoá dataset thô → format trên.
- Loss: standard next-token cross-entropy, chỉ tính loss trên phần `<solution>...</solution>` (mask phần `<problem>` nếu muốn tiết kiệm signal, tuỳ chọn).
- Output: checkpoint `model_sft`.

**Tiêu chí xong stage**: model sinh đúng cấu trúc tag + "Bước n:" + "Đáp số:" với tỷ lệ cao (>90% well-formed outputs) trên tập validation, bất kể đúng/sai về mặt toán học.

---

### Stage 2 — Gán nhãn step-level (PRM labels)

Đây là phần thay thế cho "gọi Gemini chấm" ở bản demo trước, chuyển sang **tự động, không cần API ngoài** — hợp với model 100M-500M mới học vì nó sẽ sai nhiều, tạo đủ tín hiệu tương phản.

**Phương pháp: Math-Shepherd (Monte Carlo rollout)**

Với mỗi bài toán trong tập train:
1. Model sinh lời giải đầy đủ (chia thành các bước theo tag `Bước n:`).
2. Với mỗi bước thứ `i`, cắt lời giải tại đó (giữ bước 1..i), cho model **tiếp tục sinh** phần còn lại `k` lần (ví dụ k=4-8) với sampling temperature > 0.
3. Với mỗi lần tiếp tục, so khớp "Đáp số" cuối cùng với đáp số đúng (ground truth có sẵn trong dataset).
4. Tỷ lệ đúng trong `k` lần = điểm "soft label" cho bước `i`.
   - Ngưỡng gợi ý: `correct` nếu tỷ lệ ≥ 0.6, `incorrect` nếu ≤ 0.2, còn lại `uncertain`.

**Chi phí compute**: đây là bước tốn compute nhất (mỗi bước × k rollouts × số bài). Với model 100M-500M, ước lượng ổn trên 1 GPU tầm trung nếu giới hạn:
- Số bài dùng để gán nhãn: 1,000–3,000 bài (không cần toàn bộ train set).
- k = 4–6 rollouts/bước, giới hạn max_new_tokens ngắn (chỉ cần đủ sinh hết phần còn lại của 1 lời giải, không cần dài).

**Output**: dataset dạng
```json
{
  "problem": "...",
  "steps": ["Bước 1: ...", "Bước 2: ...", ...],
  "step_labels": ["correct", "correct", "incorrect", ...],
  "final_answer": "...",
  "ground_truth": "..."
}
```

**Fallback nếu model quá yếu** (mọi rollout đều sai → không phân biệt được bước tốt/xấu): dùng thêm heuristic đơn giản — so khớp phép tính trong mỗi bước bằng cách tự parse biểu thức số học và kiểm tra bằng calculator (không cần LLM ngoài), coi bước nào tính sai số học là `incorrect` chắc chắn. Đây là lưới an toàn rẻ, không cần dựa vào Monte Carlo.

---

### Stage 3 — Build preference pairs + DPO

**Build pairs**:
- Với mỗi bài toán có ≥2 lời giải đã sinh (từ Stage 2), so sánh:
  - Lời giải có nhiều bước `correct` hơn / điểm PRM trung bình cao hơn → `chosen`.
  - Lời giải có điểm thấp hơn rõ rệt (đặc biệt nếu đáp số sai) → `rejected`.
- Chỉ giữ cặp có khoảng cách điểm đủ lớn (tránh cặp gần bằng nhau, nhiễu label).
- Format cặp: `(problem, chosen_full_solution, rejected_full_solution)` — dùng full-sequence DPO trước (đơn giản hơn step-level DPO, đủ để demo ý tưởng).

**DPO loss** (tự implement bằng PyTorch thuần, không cần `trl`):
```
L_DPO = -log σ( β * [ (logπ_θ(chosen|x) - logπ_ref(chosen|x))
                     - (logπ_θ(rejected|x) - logπ_ref(rejected|x)) ] )
```
- `π_ref` = model sau Stage 1 (SFT), đóng băng (frozen), không update.
- `π_θ` = model đang train, khởi tạo từ cùng checkpoint SFT.
- β gợi ý khởi điểm: 0.1.
- Cần: hàm tính log-prob của một chuỗi cho trước dưới model (sum log-prob theo token, chỉ trên phần response, mask phần prompt) — đây là phần logic chính cần viết vì bạn tự viết framework.

**Output**: checkpoint `model_dpo`.

---

### Stage 4 — Evaluation

- **Test set giữ riêng** (không dùng ở Stage 1/2/3): đo accuracy (đáp số đúng/sai) của `model_sft` vs `model_dpo`.
- **Quan sát CoT**: log lời giải đầy đủ của cùng một bộ câu hỏi mẫu qua từng checkpoint (pretrained → sft → dpo), lưu lại để so sánh chất lượng lập luận theo thời gian (không chỉ theo accuracy).
- Optional: dùng lại artifact PRM-visualizer đã có ở phần trước (chỉ cần đổi endpoint gọi API sang serve local model của bạn qua REST, thay vì gọi Claude) để trực quan hoá từng bước của `model_dpo`.

## 4. Format dữ liệu chốt (dùng xuyên suốt các stage)

```json
// Sample sau Stage 2, input cho Stage 3
{
  "problem": "string",
  "ground_truth_answer": "string (số, đã normalize)",
  "solutions": [
    {
      "steps": ["string", "..."],
      "step_labels": ["correct" | "incorrect" | "uncertain", "..."],
      "prm_score": 0.0,
      "final_answer": "string"
    }
  ]
}
```

## 5. Rủi ro / điểm cần cẩn thận

| Rủi ro | Ảnh hưởng | Giảm thiểu |
|---|---|---|
| Model quá yếu ở Stage 2, mọi rollout sai | Không phân biệt được bước tốt/xấu | Dùng fallback heuristic tính toán số học thay Monte Carlo cho các bước sớm |
| DPO overfit trên tập preference nhỏ | Model học "văn phong" thay vì "đúng logic" | Giữ tập preference đa dạng bài toán, không lặp lại 1 bài toán quá nhiều cặp |
| Log-prob tính sai do lỗi masking prompt/response | DPO loss vô nghĩa, train không hội tụ | Unit test riêng cho hàm log-prob trước khi ghép vào loop DPO |
| Reward hacking (nếu sau này thêm judge ngoài) | Model học đánh lừa judge thay vì giải đúng | Ưu tiên Math-Shepherd (bám ground truth) hơn LLM judge cho giai đoạn RL chính |

## 6. Thứ tự triển khai gợi ý

1. Chuẩn hoá dataset + format tag → Stage 1 SFT.
2. Viết hàm log-prob theo response (dùng lại được cho cả Stage 2 lẫn Stage 3).
3. Viết script rollout + Math-Shepherd labeling (Stage 2) trên tập nhỏ trước (~100 bài) để kiểm tra tín hiệu có ý nghĩa trước khi scale lên 1000+.
4. Build preference pairs, viết DPO loss, train.
5. Eval + log CoT so sánh.