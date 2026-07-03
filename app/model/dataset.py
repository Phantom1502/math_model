"""
dataset.py — Incremental loading cho dữ liệu tiếng Việt
============================================================
RAM hạn chế nên không load toàn bộ dataset một lần.
Thay vào đó: load từng chunk N sample, train xong, giải phóng, load chunk tiếp.

Hỗ trợ 4 nguồn dữ liệu:
    ChunkedWikiLoader    — wikimedia/wikipedia
    ChunkedVTSNLPLoader  — VTSNLP/vietnamese_curated_dataset
    ChunkedParquetLoader — file .parquet local (1 source)
    ChunkedMixLoader     — interleave nhiều source parquet theo tỷ lệ

FIX RAM: tokenize streaming theo batch nhỏ (TOKENIZE_BATCH=128)
    Tokenize từng 128 text, giải phóng ngay → peak RAM thấp.
"""

import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


TOKENIZE_BATCH = 128
MIN_DOC_LEN   = 8

# ══════════════════════════════════════════════════════════════════════════
# Dataset classes
# ══════════════════════════════════════════════════════════════════════════

class TokenChunkDataset(Dataset):
    """Mỗi document cắt thành segment độc lập, shuffle tự do."""

    def __init__(self, documents: list[list[int]], seg_len: int, min_tail_len: int = 64):
        self.samples = []
        n_skipped    = 0

        for doc in documents:
            if len(doc) < MIN_DOC_LEN:
                n_skipped += 1
                continue

            chunks = []
            n_full = len(doc) // (seg_len + 1)
            for i in range(n_full):
                start = i * (seg_len + 1)
                chunks.append(doc[start : start + seg_len + 1])

            tail = doc[n_full * (seg_len + 1):]
            if len(tail) >= min_tail_len + 1:
                chunks.append(doc[-(seg_len + 1):])

            self.samples.extend(chunks)

        if n_skipped:
            print(f"  [TokenChunkDataset] Bỏ qua {n_skipped} doc ngắn hơn {MIN_DOC_LEN} token")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids[:-1], "labels": ids[1:]}


class SequentialDocumentDataset(Dataset):
    """
    Documents shuffle ngẫu nhiên, nhưng segment của cùng document xếp tuần tự.
    Dùng DataLoader(shuffle=False).
    """

    def __init__(self, documents: list[list[int]], seg_len: int,
                 stride: int = None, shuffle_docs: bool = True):
        if stride is None:
            stride = seg_len

        doc_list  = [d for d in documents if len(d) >= MIN_DOC_LEN]
        n_skipped = len(documents) - len(doc_list)
        if n_skipped:
            print(f"  [SequentialDocumentDataset] Bỏ qua {n_skipped} doc ngắn hơn {MIN_DOC_LEN} token")

        if shuffle_docs:
            random.shuffle(doc_list)

        self.samples = []
        for doc in doc_list:
            windows = []
            start   = 0
            while start + seg_len + 1 <= len(doc):
                windows.append(doc[start : start + seg_len + 1])
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
# Collate
# ══════════════════════════════════════════════════════════════════════════

def collate_fn(batch, pad_id: int = 0):
    max_len   = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id,  dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100,    dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        labels[i, :L]    = item["labels"]

    return {"input_ids": input_ids, "labels": labels}


# ══════════════════════════════════════════════════════════════════════════
# DataLoader factory
# ══════════════════════════════════════════════════════════════════════════

def make_dataloaders(train_docs, val_docs, cfg, pad_id):
    seg_len = cfg.data.seg_len
    bs      = cfg.train.batch_size
    collate = lambda b: collate_fn(b, pad_id)

    if getattr(cfg.data, "sequential_mode", False):
        stride       = getattr(cfg.data, "window_stride", seg_len)
        train_ds     = SequentialDocumentDataset(train_docs, seg_len, stride=stride, shuffle_docs=True)
        val_ds       = TokenChunkDataset(val_docs, seg_len)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)
    else:
        train_ds     = TokenChunkDataset(train_docs, seg_len)
        val_ds       = TokenChunkDataset(val_docs,   seg_len)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  collate_fn=collate, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0)

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Base loader
# ══════════════════════════════════════════════════════════════════════════

class _BaseChunkedLoader:
    def __init__(self, cfg, tokenizer, start_chunk: int = 0):
        self.cfg          = cfg
        self.tokenizer    = tokenizer
        self.chunk_size   = cfg.data.chunk_size
        self.seg_len      = cfg.data.seg_len
        self.min_text_len = cfg.data.min_text_len
        self.val_ratio    = cfg.data.val_ratio
        self.total_chunks = cfg.train.total_chunks
        self.start_chunk  = start_chunk

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

    def _extract_text(self, sample: dict) -> str | None:
        raise NotImplementedError

    def _load_one_chunk(self) -> list[list[int]] | None:
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
                token_lists = self.tokenizer.encode_batch(batch_texts, add_special_tokens=False)
                documents.extend(ids for ids in token_lists if len(ids) >= 2)
                batch_texts = []

        if batch_texts:
            token_lists = self.tokenizer.encode_batch(batch_texts, add_special_tokens=False)
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

        train_loader, val_loader = make_dataloaders(
            train_docs, val_docs, self.cfg, self.tokenizer.pad_id,
        )

        mode = "sequential" if getattr(self.cfg.data, "sequential_mode", False) else "chunked"
        print(
            f"[Chunk {self.chunk_count}] "
            f"docs: {len(documents)} | "
            f"train: {len(train_loader.dataset)} | "
            f"val: {len(val_loader.dataset)} | "
            f"mode: {mode}"
        )
        return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════

class ChunkedWikiLoader(_BaseChunkedLoader):
    def _load_dataset(self):
        return load_dataset(
            self.cfg.data.dataset_name,
            self.cfg.data.dataset_subset,
            split="train", streaming=True,
        )

    def _extract_text(self, sample):
        return sample.get("text", "").strip() or None


class ChunkedVTSNLPLoader(_BaseChunkedLoader):
    HF_DATASET_NAME = "VTSNLP/vietnamese_curated_dataset"

    def __init__(self, cfg, tokenizer, start_chunk=0, domains=None):
        self.domains = set(domains) if domains else None
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _load_dataset(self):
        return load_dataset(self.HF_DATASET_NAME, split="train", streaming=True)

    def _extract_text(self, sample):
        if self.domains and sample.get("domain") not in self.domains:
            return None
        return sample.get("text", "").strip() or None


class ChunkedParquetLoader(_BaseChunkedLoader):
    def __init__(self, cfg, tokenizer, parquet_path, text_col="text",
                 start_chunk=0, filter_fn=None):
        self.parquet_path = parquet_path
        self.text_col     = text_col
        self.filter_fn    = filter_fn
        super().__init__(cfg, tokenizer, start_chunk=start_chunk)

    def _load_dataset(self):
        return load_dataset(
            "parquet",
            data_files={"train": self.parquet_path},
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
    """
    Interleave nhiều source parquet local theo tỷ lệ định sẵn.

    Ví dụ:
        cfg.data.source = "mix"
        cfg.data.mix.sources = {
            "wiki_vi"  : ("/data/wiki_vi/*.parquet",   0.05),
            "wiki_en"  : ("/data/wiki_en/*.parquet",   0.30),
            "math"     : ("/data/math/*.parquet",       0.10),
            "social_vi": ("/data/social_vi/*.parquet",  0.25),
            "python"   : ("/data/python/*.parquet",     0.30),
        }
        cfg.data.mix.stopping_strategy = "all_exhausted"
    """

    def _load_dataset(self):
        from datasets import interleave_datasets

        mix = self.cfg.data.mix
        if not mix.sources:
            raise ValueError("cfg.data.mix.sources trống.")

        names = list(mix.sources.keys())
        probs = [mix.sources[n][1] for n in names]

        if abs(sum(probs) - 1.0) > 1e-3:
            raise ValueError(f"Tổng probabilities = {sum(probs):.4f}, phải = 1.0")

        print(f"  Mix sources ({len(names)}):")
        datasets = []
        for name, prob in zip(names, probs):
            pattern = mix.sources[name][0]
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
            datasets, probabilities=None,
            seed=42, stopping_strategy=mix.stopping_strategy,
        )

    def _extract_text(self, sample):
        text = sample.get(self.cfg.data.parquet_text_col)
        if not isinstance(text, str):
            return None
        return text.strip() or None