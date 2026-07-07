"""
app/llama/dataset.py — Incremental streaming loader cho HF tokenizer
=========================================================================
Kiến trúc RAM-safe GIỐNG HỆT app/memlm/dataset.py (chunk theo TOKENIZE_BATCH,
giải phóng batch ngay sau khi tokenize, không load toàn bộ stream vào RAM),
chỉ khác ở tầng tokenize: gọi thẳng tokenizer(texts)["input_ids"] (HF Fast
tokenizer thật) thay vì VietnameseTokenizer.encode_batch() custom.

THÊM: text_transform tuỳ chọn, áp dụng TRƯỚC khi tokenize — dùng khi nguồn
là parquet trading cũ (định dạng "O_512 H_.." cần convert sang
"<px_O_512>" cho khớp price vocab mới, xem tokenizer.py).

THÊM: attention_mask trong collate — LlamaModel dùng attention_mask để biết
phần pad, tự sinh causal mask nội bộ (KHÁC bản cũ phải tự build causal_mask
thủ công và truyền vào model).
"""

import glob
import random
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

TOKENIZE_BATCH = 128


# ══════════════════════════════════════════════════════════════════════════
# Dataset classes — logic giữ nguyên app/memlm/dataset.py
# ══════════════════════════════════════════════════════════════════════════

class TokenChunkDataset(Dataset):
    """Mỗi document cắt thành segment độc lập, shuffle tự do."""

    def __init__(self, documents: list, seg_len: int, min_tail_len: int = 64):
        self.samples = []
        n_skipped    = 0

        for doc in documents:
            if len(doc) < 128:
                n_skipped += 1
                continue

            chunks = []
            n_full = len(doc) // (seg_len + 1)
            for i in range(n_full):
                start = i * (seg_len + 1)
                chunks.append(doc[start: start + seg_len + 1])

            tail = doc[n_full * (seg_len + 1):]
            if len(tail) >= min_tail_len + 1:
                chunks.append(doc[-(seg_len + 1):])

            self.samples.extend(chunks)

        if n_skipped:
            print(f"  [TokenChunkDataset] Bỏ qua {n_skipped} doc ngắn hơn 128 token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


class SequentialDocumentDataset(Dataset):
    """Documents shuffle ngẫu nhiên, segment của cùng document xếp tuần tự."""

    def __init__(self, documents: list, seg_len: int,
                 stride: int = None, shuffle_docs: bool = True):
        stride = stride or seg_len

        doc_list  = [d for d in documents if len(d) >= 128]
        n_skipped = len(documents) - len(doc_list)
        if n_skipped:
            print(f"  [SequentialDocumentDataset] Bỏ qua {n_skipped} doc ngắn hơn 128 token")

        if shuffle_docs:
            random.shuffle(doc_list)

        self.samples = []
        for doc in doc_list:
            windows = []
            start   = 0
            while start + seg_len + 1 <= len(doc):
                windows.append(doc[start: start + seg_len + 1])
                start += stride

            tail_start       = (len(windows) - 1) * stride if windows else 0
            tail_covered_end = tail_start + seg_len + 1
            if tail_covered_end < len(doc) and len(doc) - tail_covered_end >= 64:
                windows.append(doc[-(seg_len + 1):])

            self.samples.extend(windows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


# ══════════════════════════════════════════════════════════════════════════
# Collate — có attention_mask (khác bản cũ)
# ══════════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_id: int = 0):
    max_len        = max(item["input_ids"].size(0) for item in batch)
    input_ids      = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len),         dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L]      = item["input_ids"]
        labels[i, :L]         = item["labels"]
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def make_dataloaders(train_docs, val_docs, cfg, pad_id):
    seg_len = cfg.data.seg_len
    bs      = cfg.train.batch_size
    collate = lambda b: collate_fn(b, pad_id)

    if cfg.data.sequential_mode:
        stride       = cfg.data.window_stride or seg_len
        train_ds     = SequentialDocumentDataset(train_docs, seg_len, stride=stride, shuffle_docs=True)
        val_ds       = TokenChunkDataset(val_docs, seg_len)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, collate_fn=collate)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate)
    else:
        train_ds     = TokenChunkDataset(train_docs, seg_len)
        val_ds       = TokenChunkDataset(val_docs,   seg_len)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate)

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Base loader — tokenize qua HF tokenizer thật
# ══════════════════════════════════════════════════════════════════════════

class _BaseChunkedLoader:
    def __init__(self, cfg, tokenizer, start_chunk: int = 0,
                 text_transform: Optional[Callable[[str], str]] = None):
        self.cfg           = cfg
        self.tokenizer      = tokenizer
        self.text_transform = text_transform    # vd convert_legacy_price_tokens
        self.chunk_size     = cfg.data.chunk_size
        self.seg_len        = cfg.data.seg_len
        self.min_text_len   = cfg.data.min_text_len
        self.val_ratio      = cfg.data.val_ratio
        self.total_chunks   = cfg.train.total_chunks
        self.start_chunk    = start_chunk

        self.raw_stream = self._load_dataset()

        if start_chunk > 0:
            n_skip = start_chunk * self.chunk_size
            print(f"Resume: skip {n_skip:,} sample đầu ({start_chunk} chunk đã train)...")
            self.raw_stream = self.raw_stream.skip(n_skip)

        self.stream_iter = iter(self.raw_stream)
        self.exhausted   = False
        self.chunk_count = start_chunk

    def _load_dataset(self):
        raise NotImplementedError

    def _extract_text(self, sample: dict):
        raise NotImplementedError

    def _tokenize_batch(self, texts: list) -> list:
        """Gọi thẳng HF Fast tokenizer — nhanh hơn nhiều so với Slow tokenizer cũ."""
        if self.text_transform:
            texts = [self.text_transform(t) for t in texts]
        enc = self.tokenizer(texts, add_special_tokens=False)
        return enc["input_ids"]

    def _load_one_chunk(self):
        """Tokenize streaming theo batch nhỏ — giữ RAM peak thấp."""
        documents   = []
        batch_texts = []

        while len(documents) + len(batch_texts) < self.chunk_size:
            try:
                sample = next(self.stream_iter)
            except StopIteration:
                self.exhausted = True
                break

            text = self._extract_text(sample)
            if text is None or len(text) < self.min_text_len:
                continue

            batch_texts.append(text)

            if len(batch_texts) >= TOKENIZE_BATCH:
                token_lists = self._tokenize_batch(batch_texts)
                documents.extend(ids for ids in token_lists if len(ids) >= 2)
                batch_texts = []

        if batch_texts:
            token_lists = self._tokenize_batch(batch_texts)
            documents.extend(ids for ids in token_lists if len(ids) >= 2)

        return documents if documents else None

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        if self.total_chunks > 0 and self.chunk_count >= self.total_chunks:
            raise StopIteration

        documents = self._load_one_chunk()
        if documents is None:
            raise StopIteration

        self.chunk_count += 1

        split_idx  = int(len(documents) * (1 - self.val_ratio))
        train_docs = documents[:split_idx]
        val_docs   = documents[split_idx:] if split_idx < len(documents) else documents[-5:]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        train_loader, val_loader = make_dataloaders(train_docs, val_docs, self.cfg, pad_id)

        mode = "sequential" if self.cfg.data.sequential_mode else "chunked"
        print(
            f"[Chunk {self.chunk_count}] "
            f"docs: {len(documents)} | "
            f"train: {len(train_loader.dataset)} | "
            f"val: {len(val_loader.dataset)} | "
            f"mode: {mode}"
        )
        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Loaders cụ thể
# ══════════════════════════════════════════════════════════════════════════

class ChunkedWikiLoader(_BaseChunkedLoader):
    def _load_dataset(self):
        return load_dataset(
            self.cfg.data.dataset_name, self.cfg.data.dataset_subset,
            split="train", streaming=True,
        )

    def _extract_text(self, sample):
        return sample.get("text", "").strip() or None


class ChunkedVTSNLPLoader(_BaseChunkedLoader):
    HF_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"

    def __init__(self, cfg, tokenizer, start_chunk=0, domains=None, **kw):
        self.domains = set(domains) if domains else None
        super().__init__(cfg, tokenizer, start_chunk=start_chunk, **kw)

    def _load_dataset(self):
        return load_dataset(self.HF_DATASET_NAME, split="train", streaming=True)

    def _extract_text(self, sample):
        if self.domains and sample.get("domain") not in self.domains:
            return None
        return sample.get("text", "").strip() or None


class ChunkedParquetLoader(_BaseChunkedLoader):
    def __init__(self, cfg, tokenizer, parquet_path, text_col="text",
                 start_chunk=0, filter_fn=None, **kw):
        self.parquet_path = parquet_path
        self.text_col     = text_col
        self.filter_fn    = filter_fn
        super().__init__(cfg, tokenizer, start_chunk=start_chunk, **kw)

    def _load_dataset(self):
        return load_dataset(
            "parquet", data_files={"train": self.parquet_path},
            split="train", streaming=True,
        )

    def _extract_text(self, sample):
        if self.filter_fn and not self.filter_fn(sample):
            return None
        text = sample.get(self.text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None


class ChunkedMixLoader(_BaseChunkedLoader):
    """Interleave nhiều source parquet local theo tỷ lệ (giống bản cũ)."""

    def _load_dataset(self):
        from datasets import interleave_datasets

        sources = self.cfg.data.mix_sources
        if not sources:
            raise ValueError("cfg.data.mix_sources trống.")

        names = list(sources.keys())
        probs = [sources[n][1] for n in names]
        if abs(sum(probs) - 1.0) > 1e-3:
            raise ValueError(f"Tổng probabilities = {sum(probs):.4f}, phải = 1.0")

        print(f"  Mix sources ({len(names)}):")
        datasets = []
        for name, prob in zip(names, probs):
            pattern = sources[name][0]
            files   = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"Không tìm thấy file: {pattern}")
            random.shuffle(files)
            print(f"    {name:<14} {prob*100:.0f}%  ({len(files)} files)")
            datasets.append(
                load_dataset("parquet", data_files={"train": files},
                             split="train", streaming=True)
            )

        return interleave_datasets(
            datasets, probabilities=None, seed=42,
            stopping_strategy=self.cfg.data.mix_stopping,
        )

    def _extract_text(self, sample):
        text = sample.get(self.cfg.data.parquet_text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None