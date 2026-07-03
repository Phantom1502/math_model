"""
sft_dataset.py — Dataset cho SFT (Stage 1) từ file JSONL prompt/completion
=============================================================================
Khác dataset.py (dùng cho pretrain, streaming theo document liên tục cắt
segment cố định): SFTDataset đọc toàn bộ file .jsonl vào RAM một lần
(~7K sample, nhẹ), mỗi sample là một cặp (prompt, completion) ĐỘC LẬP —
không cắt/ghép segment như TokenChunkDataset.

Loss masking:
    tokens        = [bos] + prompt_ids + completion_ids + [eos]
    is_completion = [ F ] + [F]*len(prompt_ids) + [T]*len(completion_ids) + [T]

    input_ids = tokens[:-1]
    labels[i] = tokens[i+1]   nếu is_completion[i+1] == True   (thuộc completion/eos)
              = -100          nếu is_completion[i+1] == False  (thuộc bos/prompt)

Tái dùng nguyên collate_fn() trong dataset.py — nó đã pad labels bằng -100
sẵn, đúng ý nghĩa "ignore_index" mà BaseTrainer.compute_loss dùng mặc định.
KHÔNG cần override compute_loss cho SFT.
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader

from dataset import collate_fn


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_example(tokenizer, prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """Ghép prompt + completion thành 1 sequence, build labels đã mask sẵn."""
    prompt_ids     = tokenizer.encode(prompt,     add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)

    tokens        = [tokenizer.bos_id] + prompt_ids + completion_ids + [tokenizer.eos_id]
    is_completion = [False] + [False] * len(prompt_ids) + [True] * len(completion_ids) + [True]

    input_ids = tokens[:-1]
    labels    = [tok if flag else -100
                 for tok, flag in zip(tokens[1:], is_completion[1:])]

    return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.samples = []
        n_skipped = 0

        for row in rows:
            input_ids, labels = _build_example(tokenizer, row["prompt"], row["completion"])
            if len(input_ids) < 2:
                n_skipped += 1
                continue
            self.samples.append((input_ids, labels))

        if n_skipped:
            print(f"  [SFTDataset] Bỏ qua {n_skipped} sample quá ngắn sau khi encode")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, labels = self.samples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels"   : torch.tensor(labels,    dtype=torch.long),
        }


def split_train_val(
    rows      : list[dict],
    val_ratio : float = 0.03,
    seed      : int   = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Tách val NGAY TRONG train.jsonl — KHÔNG đụng test.jsonl.
    test.jsonl giữ nguyên làm held-out set cho Stage 4 theo đúng spec.
    """
    rows = list(rows)
    rng  = random.Random(seed)
    rng.shuffle(rows)

    n_val      = max(1, int(len(rows) * val_ratio))
    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def make_sft_dataloaders(
    train_jsonl_path: str,
    tokenizer,
    batch_size: int,
    val_ratio : float = 0.03,
    seed      : int   = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Entry point chính: đọc train.jsonl (output prepare_sft_data.py), tự tách
    val nội bộ (KHÔNG dùng test.jsonl), build 2 DataLoader.
    """
    rows = load_jsonl(train_jsonl_path)
    train_rows, val_rows = split_train_val(rows, val_ratio=val_ratio, seed=seed)

    print(f"  [make_sft_dataloaders] train={len(train_rows)}  val={len(val_rows)}  "
          f"(val_ratio={val_ratio}, tách nội bộ từ {train_jsonl_path})")

    train_ds = SFTDataset(train_rows, tokenizer)
    val_ds   = SFTDataset(val_rows,   tokenizer)

    collate = lambda b: collate_fn(b, tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    return train_loader, val_loader