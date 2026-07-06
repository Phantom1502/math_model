"""
dpo_dataset.py — Dataset cho DPO (Stage 3b) từ preference pairs
=============================================================================
Đọc pairs.jsonl (output build_dpo_pairs.py). Mỗi sample gồm 2 sequence
(chosen, rejected) CÙNG chung 1 prompt — tái dùng NGUYÊN _build_example()
từ sft_dataset.py để mask loss (chỉ tính trên phần response, prompt mask
-100) — DPO cần log-prob CHỈ trên phần response, giống hệt lý do SFT mask.

Không tái dùng thẳng SFTDataset vì mỗi sample DPO cần 2 sequence độc lập
(chosen/rejected), không phải 1.
"""

import torch
from torch.utils.data import Dataset, DataLoader

from sft_dataset import _build_example, load_jsonl
from dataset import collate_fn


class DPODataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer):
        self.samples = []
        n_skipped = 0

        for row in rows:
            c_ids, c_labels = _build_example(tokenizer, row["prompt"], row["chosen"])
            r_ids, r_labels = _build_example(tokenizer, row["prompt"], row["rejected"])
            if len(c_ids) < 2 or len(r_ids) < 2:
                n_skipped += 1
                continue
            self.samples.append((c_ids, c_labels, r_ids, r_labels))

        if n_skipped:
            print(f"  [DPODataset] Bỏ qua {n_skipped} pair quá ngắn sau khi encode")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        c_ids, c_labels, r_ids, r_labels = self.samples[idx]
        return {
            "chosen_input_ids"  : torch.tensor(c_ids,    dtype=torch.long),
            "chosen_labels"     : torch.tensor(c_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_ids,    dtype=torch.long),
            "rejected_labels"   : torch.tensor(r_labels, dtype=torch.long),
        }


def collate_dpo(batch: list[dict], pad_id: int) -> dict:
    """Pad chosen và rejected RIÊNG (độ dài có thể khác nhau trong cùng batch),
    tái dùng collate_fn có sẵn cho mỗi nhóm."""
    chosen_batch   = [{"input_ids": b["chosen_input_ids"],   "labels": b["chosen_labels"]}   for b in batch]
    rejected_batch = [{"input_ids": b["rejected_input_ids"], "labels": b["rejected_labels"]} for b in batch]

    chosen   = collate_fn(chosen_batch,   pad_id)
    rejected = collate_fn(rejected_batch, pad_id)

    return {
        "chosen_input_ids"  : chosen["input_ids"],
        "chosen_labels"     : chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_labels"   : rejected["labels"],
    }


def make_dpo_dataloader(
    pairs_jsonl_path: str,
    tokenizer,
    batch_size: int,
    shuffle   : bool = True,
) -> DataLoader:
    rows = load_jsonl(pairs_jsonl_path)
    print(f"  [make_dpo_dataloader] {len(rows)} pairs từ {pairs_jsonl_path}")

    ds = DPODataset(rows, tokenizer)
    collate = lambda b: collate_dpo(b, tokenizer.pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=0)