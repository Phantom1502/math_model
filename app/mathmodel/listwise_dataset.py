"""
listwise_dataset.py — Dataset cho Plackett-Luce / PRO (Stage 3c) từ 3-tier data
=============================================================================
Đọc output scripts/label_selfcorrect_data.py (tiers.jsonl). Mỗi bài có 3 mức
xếp hạng CỐ ĐỊNH theo đúng Self_correction_training.md mục 2.4-2.5:

    rank 0 (tốt nhất) : tier_correct_straight — đúng ngay từ đầu
    rank 1            : tier_wrong_fixed      — sai nhưng nhận ra + sửa được
    rank 2 (kém nhất) : tier_wrong_raw        — sai và không sửa

Tái dùng _build_example() từ sft_dataset.py để mask loss (prompt→-100,
completion giữ nguyên) — giống hệt DPODataset, chỉ khác 3 sequence/sample
thay vì 2.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from sft_dataset import _build_example, load_jsonl
from dataset import collate_fn

_RANK_KEYS = ("tier_correct_straight", "tier_wrong_fixed", "tier_wrong_raw")


class ListwiseDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.samples = []
        n_skipped = 0

        for row in rows:
            try:
                items = []
                for key in _RANK_KEYS:
                    ids, labels = _build_example(tokenizer, row["prompt"], row[key])
                    if len(ids) < 2:
                        raise ValueError("quá ngắn sau khi encode")
                    items.append((ids, labels))
                self.samples.append(items)
            except (KeyError, ValueError):
                n_skipped += 1

        if n_skipped:
            print(f"  [ListwiseDataset] Bỏ qua {n_skipped} bài thiếu tier hoặc quá ngắn")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        items = self.samples[idx]
        out = {}
        for r in range(3):
            out[f"rank{r}_input_ids"] = torch.tensor(items[r][0], dtype=torch.long)
            out[f"rank{r}_labels"]    = torch.tensor(items[r][1], dtype=torch.long)
        return out


def collate_listwise(batch: list[dict], pad_id: int) -> dict:
    """Pad riêng từng rank (độ dài khác nhau giữa các rank), tái dùng collate_fn có sẵn."""
    out = {}
    for r in range(3):
        group = [{"input_ids": b[f"rank{r}_input_ids"], "labels": b[f"rank{r}_labels"]} for b in batch]
        collated = collate_fn(group, pad_id)
        out[f"rank{r}_input_ids"] = collated["input_ids"]
        out[f"rank{r}_labels"]    = collated["labels"]
    return out


def make_listwise_dataloader(
    tiers_jsonl_path: str,
    tokenizer,
    batch_size: int,
    shuffle   : bool = True,
) -> DataLoader:
    rows = load_jsonl(tiers_jsonl_path)
    print(f"  [make_listwise_dataloader] {len(rows)} bài từ {tiers_jsonl_path}")

    ds = ListwiseDataset(rows, tokenizer)
    collate = lambda b: collate_listwise(b, tokenizer.pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)