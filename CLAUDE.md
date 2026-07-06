# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Tổng quan dự án

Demo cá nhân để hiểu luồng **SFT → gán nhãn step-level (PRM) → RL/DPO** cho một
mô hình ngôn ngữ nhỏ (~100-110M tham số, tự pretrain, PyTorch thuần — không
dùng `transformers`/`trl` cho phần training). Mục tiêu là **hiểu cơ chế**, không
phải đạt SOTA. Xem `README.md` ở gốc repo để biết spec đầy đủ 4 giai đoạn.

Model gốc (`MemoryLM`, kiến trúc LLaMA-style: RMSNorm, RoPE, SwiGLU, weight
tying, KV-cache) ban đầu được pretrain cho tiếng Việt (Wikipedia + VTSNLP),
sau đó SFT thêm để học giải toán word-problem tiếng Anh (GSM8K).

## Chạy lệnh — LUÔN từ thư mục gốc repo

Mọi entry-point (`train*.py`, `generate.py`, `benchmark*.py` trong
`app/mathmodel/`, và mọi file trong `app/mathmodel/scripts/`) đều tự chèn
đường dẫn của chính nó vào `sys.path` dựa trên `__file__` — nên chạy được từ
BẤT KỲ thư mục nào, kể cả gốc repo, bằng cả 2 cách:

```bash
# Cách 1 — chạy file trực tiếp
python app/mathmodel/train_sft.py --pretrained-ckpt ...

# Cách 2 — module mode
python -m app.mathmodel.train_sft --pretrained-ckpt ...
```

**LƯU Ý QUAN TRỌNG — đường dẫn data/checkpoint KHÔNG tự động theo `__file__`:**
các giá trị mặc định trong argparse (vd `data/gsm8k_sft/train.jsonl`,
`checkpoints_sft/`, `custom_tokenizer`) được resolve theo **thư mục đang đứng
khi gõ lệnh** (cwd), không phải theo vị trí file `.py`. Nếu chạy từ gốc repo,
PHẢI truyền path đầy đủ tính từ gốc, ví dụ:

```bash
python app/mathmodel/train_sft.py \
    --pretrained-ckpt app/mathmodel/checkpoints/chunk_50.pt \
    --train-jsonl app/mathmodel/data/gsm8k_sft/train.jsonl
```

Đừng dựa vào default argparse khi chạy từ gốc — luôn truyền tường minh.

## Cấu trúc thư mục

```
app/mathmodel/
├── config.py              # Toàn bộ hyperparameter (Config/ModelConfig/DataConfig/TrainConfig/TokenizerConfig)
├── tokenizer.py            # VietnameseTokenizer — BPE base + price-vocab riêng (price vocab KHÔNG liên quan pipeline toán)
├── dataset.py               # Dataset/DataLoader cho PRETRAIN (streaming, chunk lớn)
├── sft_dataset.py            # Dataset cho SFT — đọc prompt/completion jsonl, mask loss phần prompt
├── dpo_dataset.py             # Dataset cho DPO — 2 sequence/sample (chosen/rejected)
├── listwise_dataset.py         # Dataset cho Plackett-Luce — 3 sequence/sample (3-tier)
├── generate.py                  # generate() (KV-cache, single) + generate_batch() (nhiều continuation song song)
├── benchmark.py                  # Benchmark TIẾNG VIỆT (semantic/entity/fact/ood) — cho model PRETRAIN, KHÔNG dùng để đo SFT toán
├── benchmark_hellaswag.py         # HellaSwag zero-shot (tiếng Anh) — theo dõi trend, không so tuyệt đối
├── train.py                       # Entry Stage 0 — pretrain
├── train_sft.py                    # Entry Stage 1 — SFT format CoT trên GSM8K
├── train_dpo.py                     # Entry Stage 3b — DPO (2 mức, pipeline Math-Shepherd)
├── train_listwise.py                 # Entry Stage 3c — Plackett-Luce (3 mức, pipeline diff-based) — ĐANG DÙNG
├── model/                              # Kiến trúc MemoryLM (RMSNorm, RoPE, SwiGLU, KV-cache)
├── trainer/
│   ├── base.py                          # BaseTrainer — optimizer/scheduler/scaler, compute_loss CE mặc định
│   ├── pretrain.py                       # PretrainTrainer — dùng train_one_chunk (streaming lớn)
│   ├── sft.py                             # SFTTrainer — loop epoch riêng, KHÔNG gọi benchmark.py (tiếng Việt, không liên quan)
│   ├── dpo.py                              # DPOTrainer + sequence_logprob() — dùng lại cho cả listwise
│   └── listwise.py                          # ListwiseTrainer — Plackett-Luce, tái dùng sequence_logprob từ dpo.py
├── utils/                                    # checkpoint save/load, TrainLogger
└── scripts/
    ├── train_tokenizer.py                      # Train BPE tokenizer từ đầu (chạy 1 lần)
    ├── add_custom_tokens.py                      # (legacy, dành cho PhoBERT — không dùng trong pipeline toán)
    ├── prepare_sft_data.py                        # GSM8K thô → format Step-by-step, lọc theo max_seq
    ├── label_prm_data.py                           # [PIPELINE A — THAM KHẢO] Math-Shepherd Monte Carlo rollout
    ├── build_dpo_pairs.py                           # [PIPELINE A] preference pairs từ nhãn Math-Shepherd → train_dpo.py
    └── label_selfcorrect_data.py                     # [PIPELINE B — ĐANG DÙNG] diff-based 3-tier → train_listwise.py
```

## HAI pipeline Stage 2/3 song song — KHÔNG trộn lẫn

Repo có **2 cách tạo dữ liệu preference riêng biệt**, cố tình tách rời để giữ
pipeline cũ làm tham khảo:

| | Pipeline A (tham khảo) | Pipeline B (đang dùng) |
|---|---|---|
| Script gán nhãn | `scripts/label_prm_data.py` | `scripts/label_selfcorrect_data.py` |
| Phương pháp | Monte Carlo rollout (k lần/step) | Diff giá trị so với ground truth (không cần rollout) |
| Chi phí | ~50-230 lần gọi model/bài | ~1 lần gọi model/bài |
| Output | `data/gsm8k_prm/labeled.jsonl` (nhiều solution, step_labels rời rạc) | `data/gsm8k_selfcorrect/tiers.jsonl` (đúng 3 tier/bài) |
| Script build training data | `scripts/build_dpo_pairs.py` → `data/gsm8k_dpo/pairs.jsonl` (2 mức) | (không cần bước riêng — tiers.jsonl dùng thẳng) |
| Script train | `train_dpo.py` (DPO pairwise) | `train_listwise.py` (Plackett-Luce 3 mức) — **ưu tiên dùng cái này** |

`label_prm_data.py`/`build_dpo_pairs.py`/`trainer/dpo.py` vẫn hoạt động và
được giữ nguyên vì **có giá trị tham khảo** (cách tiếp cận tổng quát hơn, áp
dụng được cả khi KHÔNG có ground-truth từng bước). Không xoá, không refactor
gộp chung với Pipeline B.

## Quy ước code

- **Comment/docstring: tiếng Việt. Nội dung dữ liệu (prompt GSM8K, format Step/Answer): tiếng Anh** — quyết định có chủ đích vì GSM8K là tiếng Anh và model có `max_seq=512` nhỏ, không đủ chỗ cho lập luận rườm rà.
- **Format completion cố định** (dùng xuyên suốt SFT/DPO/Listwise):
  ```
  Problem: {question}
  Solution:
  Step 1: {...}
  Step 2: {...}
  Answer: {number}
  ```
  Không dùng tag XML (`<problem>`, `<solution>`) — tốn token không cần thiết với model nhỏ.
- **Loss masking**: mọi dataset SFT-style (`sft_dataset.py`, `dpo_dataset.py`, `listwise_dataset.py`) đều dùng chung `_build_example()` trong `sft_dataset.py` — mask `-100` cho phần `prompt`, giữ nguyên phần `completion`. Sửa logic mask thì sửa **một chỗ duy nhất** ở đó.
- **`generate()` vs `generate_batch()`** (`generate.py`): `generate()` cho 1 sequence; `generate_batch()` sinh nhiều continuation SONG SONG cho CÙNG 1 prompt (dùng khi cần nhiều rollout từ cùng 1 điểm — vd Monte Carlo). `generate_batch()` KHÔNG early-stop riêng từng sequence khi gặp `eos` giữa batch (đơn giản hoá có chủ đích, xem docstring trong file).
- **`add_bos`**: mặc định `False` ở mọi nơi gọi `generate()`/`generate_batch()` — đã test thực nghiệm rằng model không nhạy với việc thiếu BOS ở bước inference (xem lịch sử conversation/commit liên quan đến debug format SFT).
- **Checkpoint luôn kèm `model_cfg`** (qua `save_checkpoint(..., model_cfg=cfg.model)`) — mọi script load checkpoint (`generate.py`, `train_sft.py`, `train_dpo.py`, `train_listwise.py`) đều đọc `model_cfg` từ checkpoint để build đúng kiến trúc, KHÔNG dùng config mặc định của script hiện tại (tránh lệch kiến trúc giữa các giai đoạn).
- **Thứ tự load weight khi train DPO/Listwise**: PHẢI load checkpoint SFT vào `model` TRƯỚC khi khởi tạo `DPOTrainer`/`ListwiseTrainer` — vì `__init__` của 2 class này deepcopy `model` hiện tại làm `ref_model` đóng băng. Load sai thứ tự → `ref_model` là bản random-init, phá vỡ toàn bộ ý nghĩa DPO/Plackett-Luce. Xem `run_dpo()`/`run_listwise()` để làm đúng thứ tự này.

## Thứ tự chạy pipeline đầy đủ (Pipeline B — đang dùng)

```bash
# Stage 0 (nếu chưa có pretrained checkpoint)
python app/mathmodel/train.py

# Chuẩn bị data SFT
python app/mathmodel/scripts/prepare_sft_data.py \
    --output-dir app/mathmodel/data/gsm8k_sft \
    --tokenizer-path app/mathmodel/custom_tokenizer

# Stage 1 — SFT
python app/mathmodel/train_sft.py \
    --pretrained-ckpt app/mathmodel/checkpoints/chunk_50.pt \
    --train-jsonl app/mathmodel/data/gsm8k_sft/train.jsonl

# Stage 2b — gán nhãn 3-tier (diff-based, KHÔNG Monte Carlo)
python app/mathmodel/scripts/label_selfcorrect_data.py \
    --checkpoint app/mathmodel/checkpoints_sft/sft_best.pt \
    --train-jsonl app/mathmodel/data/gsm8k_sft/train.jsonl \
    --output app/mathmodel/data/gsm8k_selfcorrect/tiers.jsonl \
    --n-problems -1

# Stage 3c — Plackett-Luce listwise training
python app/mathmodel/train_listwise.py \
    --sft-ckpt app/mathmodel/checkpoints_sft/sft_best.pt \
    --tiers-jsonl app/mathmodel/data/gsm8k_selfcorrect/tiers.jsonl
```

## Việc CHƯA làm (biết trước để không làm trùng)

- **PRM độc lập** (classifier riêng, generalize được cho bài/step chưa từng thấy) — hiện tại cả 2 pipeline chỉ *sinh dữ liệu* rồi feed thẳng vào training policy (DPO/Listwise), KHÔNG có model verifier tồn tại độc lập sau khi train xong.
- **Stage 4** (evaluation trên `test.jsonl` giữ riêng, so `model_sft` vs `model_dpo`/`model_listwise`) — chưa viết script, `test.jsonl` vẫn chưa bị đụng tới.
- Batch hoá multi-problem trong `label_selfcorrect_data.py` (hiện xử lý tuần tự từng bài, `generate()` batch=1 cho rollout) — có thể tối ưu thêm nếu chạy full 7,464 bài quá chậm.