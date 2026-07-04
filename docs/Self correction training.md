# Huấn luyện khả năng tự-sửa-lỗi (Self-Correction) cho Reasoning Model
## Từ nguyên nhân → giải pháp đã bàn → tổng kết pipeline

---

## PHẦN 1: NGUYÊN NHÂN GỐC

### 1.1. Quan sát khởi điểm
Reasoning trace của các model lớn (ví dụ Fable 5, DeepSeek-R1...) thường chứa những cụm từ ngắn, mang tính "cảm thán" — "Wait", "Hmm", "GRRR", "aha" — trong lúc giải bài khó. Câu hỏi đặt ra: đây là hiện tượng gì, và có thể chủ động huấn luyện ra nó không?

### 1.2. Giải thích cơ chế (không phải "cảm xúc" theo nghĩa người)
- Những từ này hoạt động như **marker/pointer nén** — đánh dấu một điểm chuyển pha trong quá trình suy luận (phát hiện mâu thuẫn → cần quay lại) — thay vì phải viết lại toàn bộ câu dài mỗi lần.
- Về bản chất autoregressive: model không có "cảm biến" tách rời để tự giám sát. Việc "phát hiện sai" là hệ quả của việc **tường minh viết ra bước tự-kiểm-tra** trong chính chuỗi token, và bước đó quay lại ảnh hưởng đến phân phối xác suất token kế tiếp.
- Có bằng chứng cho một lớp "cảm nhận" sớm hơn ở mức hidden state (probing cho thấy có thể dự đoán đúng/sai *trước khi* model viết token ra) — gợi ý một dạng self-monitoring ẩn, xuất hiện tự nhiên qua huấn luyện chứ không được thiết kế trực tiếp.
- RL với process reward là yếu tố quyết định: nếu điểm nào trong chuỗi mà việc "dừng lại, backtrack" giúp tăng xác suất đúng ở bước cuối, gradient sẽ củng cố xu hướng đó — đây là nguồn gốc của hiện tượng "aha moment" tự phát sinh trong các model như DeepSeek-R1, dù không ai dạy trực tiếp cụm từ đó.

### 1.3. Vấn đề thực tế: "Self-Correction Blind Spot"
Nghiên cứu thực nghiệm (Self-Correction Bench, 2025) chỉ ra: LLM **không tự sửa được lỗi của chính mình**, dù thừa kiến thức để làm việc đó. Nguyên nhân được truy về **thành phần dữ liệu huấn luyện**:
- Corpus SFT tiêu chuẩn (OpenAssistant, UltraFeedback...) gần như toàn completion sạch, hiếm khi chứa đoạn tự-sửa.
- Ngược lại, các model reasoning được RL kỹ (DeepSeek-R1, phi-4-reasoning-plus) có tần suất marker sửa lỗi ("Wait", "But", "However") rất cao → gần như không còn blind spot.
- Bằng chứng thực nghiệm mạnh: chỉ cần **chèn từ "Wait"** ngay sau bước sai (không cần finetune) đã giảm blind spot trung bình 89.3%, tăng độ chính xác trung bình 156%.
- Kết luận: khả năng tự sửa **đã tồn tại tiềm ẩn** trong kiến trúc model, chỉ cần được **kích hoạt tường minh** qua tần suất ví dụ trong dữ liệu huấn luyện — không phải xây từ đầu.

**→ Đây là lý do chính đáng để chủ động thiết kế dữ liệu huấn luyện dạy hành vi tự-sửa, thay vì chờ nó tự phát sinh ngẫu nhiên.**

---

## PHẦN 2: CÁC GIẢI PHÁP ĐÃ BÀN

### 2.1. Ranking 4 mức chất lượng lời giải
Thay vì nhị phân đúng/sai, cần phân biệt rõ 4 trạng thái (quan trọng vì 2 trạng thái "sai" có bản chất khác nhau):

| Mức | Mô tả | Ý nghĩa |
|---|---|---|
| 1 | Đúng hoàn hảo ngay từ đầu | Reward cao nhất |
| 2 | Sai → nhận ra → sửa → đúng | Có metacognition + khả năng sửa |
| 3 | Sai → nhận ra → không sửa được | Có metacognition, thiếu khả năng sửa (cần tách riêng khỏi mức 4, nếu không model không có tín hiệu học riêng kỹ năng *nhận diện*) |
| 4 | Sai → không nhận ra → sai | Thiếu cả hai kỹ năng |

**Lưu ý thiết kế reward quan trọng:**
- Không đặt reward(mức 1) cách quá xa reward(mức 2) — nếu không, model học cách **che giấu sự không chắc chắn** thay vì xử lý minh bạch (đi thẳng tới đáp án dù nội bộ "biết" đang sai, vì lộ ra ngoài bị phạt nặng hơn).
- Reward nên gắn theo **từng bước transition** (dùng process reward model — PRM), không chỉ đặt ở cuối chuỗi, để gradient trỏ đúng vào hành vi cụ thể cần củng cố.

### 2.2. Hai paradigm huấn luyện khác nhau — cần phân biệt rõ
| Paradigm | Cơ chế | Dạy được gì |
|---|---|---|
| **Avoidance** (Step-DPO gốc) | reject = bước sai, chosen = bước đúng (từ cùng prefix đúng) | Tránh đi vào đường sai ngay từ đầu |
| **Recovery** (biến thể được bàn) | reject = bước sai + đi tiếp mù quáng, chosen = bước sai + "Wrong!" + sửa | Phản xạ phát hiện + phục hồi khi *đã* lỡ sai — tổng quát hoá tốt hơn cho lỗi mới, chưa từng thấy |

Cả hai bổ sung cho nhau, không thay thế nhau.

### 2.3. Step-DPO — nền tảng kỹ thuật
- Coi **từng bước suy luận** là đơn vị tối ưu, không phải toàn bộ câu trả lời — giải quyết vấn đề DPO gốc pha loãng reward trên chuỗi dài, khó định vị lỗi.
- Cách xây cặp dữ liệu: rollout on-policy → xác định bước sai đầu tiên → giữ prefix đúng chung → tạo cặp (reject=bước sai, chosen=bước đúng) chỉ tại điểm phân kỳ.
- Phát hiện quan trọng: **self-generated data hiệu quả hơn data người/GPT-4 viết** — vì data ngoài bị lệch phân phối (out-of-distribution) so với lỗi thật của chính model.
- Hạn chế: chỉ xử lý bước sai *đầu tiên*, bỏ qua các bước còn lại → các biến thể sau (Full-Step-DPO, SVPO dùng MCTS, Step-APO dùng advantage estimate) mở rộng để khắc phục.

### 2.4. Bộ 3 cặp dữ liệu để mã hoá đủ ranking 3 mức
```
Cặp 1: reject = sai trần | chosen = sai + "Wrong!" + sửa      → dạy phục hồi
Cặp 2: reject = sai trần | chosen = đúng thẳng                 → dạy tránh sai (Step-DPO gốc)
Cặp 3: reject = sai + sửa | chosen = đúng thẳng                → dạy "sửa được" vẫn kém "đúng ngay"
```
Cặp 3 đặc biệt quan trọng: thiếu nó, model có thể học lệch rằng đường vòng sai-rồi-sửa cũng tốt ngang đúng ngay → dẫn đến xu hướng overthinking không cần thiết.

### 2.5. Vấn đề khi dùng 3 cặp DPO tách rời → Listwise Ranking Loss
- DPO chỉ xử lý so sánh 1-vs-1. Ba cặp train độc lập có thể **xung đột gradient** (cặp 1 và cặp 3 chia sẻ nhánh "sai+sửa" nhưng đóng vai trò ngược nhau) và không đảm bảo tính bắc cầu nhất quán (A>B, B>C nhưng không chắc A>C được tôn trọng).
- **Giải pháp: Plackett-Luce ranking loss (nền tảng của PRO — Preference Ranking Optimization)** — tối ưu xác suất của *cả một thứ tự* (A>B>C) trong một lần tính loss duy nhất, thay vì 3 trận đấu tay đôi rời rạc.
- Trực giác: giống mô hình hoá một cuộc đua ngựa, không phải 3 trận đấu vật riêng lẻ.
- Công thức tổng quát:
  ```
  P(A > B > C) = [score(A)/(score(A)+score(B)+score(C))] × [score(B)/(score(B)+score(C))]
  loss = -log P(đúng thứ tự này)
  ```
- Cần lưu ý thêm: **length bias của DPO** — completion dài hơn (sai+sửa) dễ bị model học nhầm tín hiệu "ngắn = tốt" thay vì "đúng ngay = tốt". Cần length-normalized DPO hoặc SimPO-style normalization.
- Tỷ lệ trộn 3 loại cặp nên thay đổi theo giai đoạn: nghiêng về Cặp 2 (avoidance) trước, tăng dần Cặp 1 & 3 (recovery) sau — giống chiến lược 2-stage của SCoRe (giữ policy ổn định trước, khuyến khích self-correction sau) để tránh reward hacking (model "giả vờ sai" để ăn điểm sửa lỗi).

### 2.6. Tự động hoá việc tìm & tiêm lỗi (giảm chi phí sinh dữ liệu)

**Bài toán con A — Định vị bước sai (localization):**

| Phương pháp | Cơ chế | Chi phí |
|---|---|---|
| Math-Shepherd (brute-force) | Sample T rollout tại *mỗi* bước, đo tỷ lệ ra đáp án đúng | Cao — tỷ lệ thuận độ dài × số sample |
| **OmegaPRM** | Binary search trên cây suy luận (giống `git bisect`) để tìm bước sai đầu tiên, tái sử dụng rollout | Thấp — **75 lần hiệu quả hơn** brute-force ở cùng ngân sách (200K vs 15 triệu điểm dữ liệu) |
| Symbolic checker (Math-Shepherd) | So trực tiếp giá trị model nói ra với ground truth bằng công cụ biểu tượng/quy tắc | Rẻ nhất — không cần gọi model để chấm |
| FOVER | Formal verification (Z3, Isabelle) cho domain hình thức hoá được | Rẻ, chính xác cao, giới hạn ở domain logic/chứng minh |

**Bài toán con B — Sinh phần sửa (injection):**
- Với ground truth có sẵn: dùng **template tự động** ("Wrong! giá trị đúng là X" — X lấy thẳng từ đề bài) — gần như miễn phí.
- Sau đó chỉ cần **một lần generate** (model đóng vai "completer", không phải search) để viết tiếp phần đúng — vì phần khó (tìm lỗi + giá trị đúng) đã giải quyết ở bước A.
- **Pipeline offline, chạy 1 lần:** sinh hàng loạt rollout on-policy → localize lỗi → sinh 3 biến thể mỗi bài → cache xuống đĩa → tái sử dụng qua nhiều epoch train, tránh regenerate lặp lại.

### 2.7. Trường hợp đặc biệt: có ground-truth từng bước đầy đủ (mini math model)
Khi đã có lời giải chuẩn đầy đủ (`step1, step2, ...`) cho mỗi bài, việc localize trở nên rẻ và đơn giản — chỉ cần **diff** thay vì Monte Carlo/MCTS. Nhưng có 3 bẫy kỹ thuật cần tránh:

1. **So sánh text thô → false positive.** Hai step có thể khác chữ nhưng cùng đúng toán học (ví dụ `3+6` vs `6+3`). → Phải parse ra **state/giá trị thực sự** (dùng sympy simplify nếu cần) rồi so giá trị, không so string.
2. **Lệch alignment sau điểm phân kỳ đầu.** Model có thể gộp/tách step khác granularity với ground truth, khiến so theo index cứng gây báo sai hàng loạt. → Dùng **thuật toán alignment kiểu dynamic programming** (giống Needleman-Wunsch) để ánh xạ step model ↔ step ground truth trước khi kết luận.
3. **Lệch văn phong khi ghép nối.** Nếu chosen = (step model tự viết) + (step ground truth copy nguyên) sẽ tạo "đường nối" lệch phong cách, khiến model học nhầm tín hiệu đổi văn phong thay vì sửa lỗi logic. → Ground truth chỉ nên dùng làm **giá trị tham chiếu để chấm**, còn câu chữ nối tiếp để model tự generate (1 lần decode) theo đúng văn phong đang dùng.

**Lợi thế tận dụng được:** vì có ground truth đầy đủ, có thể áp nguyên tắc Step-DPO — chỉ tính loss trên đoạn phân kỳ trở đi, không cần hoàn thiện/tính loss trên toàn bộ đuôi còn lại → vừa rẻ hơn, vừa tránh bẫy lệch văn phong.

---

## PHẦN 3: TỔNG KẾT — PIPELINE ĐỀ XUẤT

```
1. THU THẬP LỖI TỰ NHIÊN (on-policy)
   └─ Model tự giải bài → parse ra từng step

2. ĐỊNH VỊ LỖI (localization)
   ├─ Có ground-truth từng bước? → Diff theo STATE (không theo text) 
   │                                + alignment bằng DP (không theo index cứng)
   └─ Không có?                  → OmegaPRM (binary search + MCTS) 
                                    hoặc symbolic checker nếu domain cho phép

3. SINH DỮ LIỆU 3 MỨC cho mỗi bài
   ├─ Đúng-thẳng      (có sẵn từ ground truth / rollout đúng)
   ├─ Sai-trần        (rollout sai gốc, không cần sinh thêm)
   └─ Sai + "Wrong!" + sửa  (template giá trị đúng + 1 lần model tự generate tiếp,
                             GIỮ văn phong của model, KHÔNG copy nguyên câu ground truth)

4. CACHE TOÀN BỘ dataset xuống đĩa — sinh 1 lần, dùng lại nhiều epoch

5. HUẤN LUYỆN
   ├─ Ưu tiên: Listwise ranking loss (Plackett-Luce / PRO) thay vì 3 DPO pairwise rời rạc
   ├─ Length-normalize để tránh lệch tín hiệu theo độ dài completion
   ├─ Chỉ tính loss trên đoạn phân kỳ (không toàn bộ chuỗi) — rẻ hơn, đúng chuẩn Step-DPO
   └─ Tỷ lệ trộn cặp: nghiêng avoidance trước → tăng dần recovery sau (tránh reward hacking)
```

### Nguyên tắc cốt lõi xuyên suốt
- Lỗi dùng để huấn luyện phải là **lỗi thật của chính model** (on-policy), không phải lỗi người bịa ra — tránh distribution mismatch.
- Tách biệt rõ "tránh sai" và "phục hồi sau sai" — đây là hai kỹ năng khác nhau, cần tín hiệu huấn luyện riêng.
- Luôn có ví dụ âm cho việc "sửa lỗi không cần thiết" — nếu không, model học thói quen tự nghi ngờ vô cớ, gây overthinking.
- Chi phí sinh dữ liệu giảm được bằng cách tách "tìm giá trị đúng" (rẻ, dùng ground truth/checker) khỏi "viết văn bản tiếp theo" (chỉ cần 1 lần decode, không cần search).

---

*Tài liệu tổng hợp từ thảo luận, tham chiếu các công trình: Step-DPO (Lai et al., 2024), Full-Step-DPO (2025), SVPO, Step-APO, Math-Shepherd (Wang et al., 2024), OmegaPRM (Luo et al., 2024), FOVER (Kamoi et al., 2025), Self-Correction Bench (2025), SCoRe (DeepMind), PRO (Preference Ranking Optimization), DeepSeek-R1 (GRPO).*