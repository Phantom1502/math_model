"""
config.py — Toàn bộ hyperparameter của project
================================================
Sửa giá trị ở đây, không sửa rải rác trong code.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MixConfig:
    """
    Cấu hình mix nhiều nguồn parquet local theo tỷ lệ.

    sources: dict tên → (glob_path, probability)
        Ví dụ:
            {
                "wiki" : ("data/wiki/*.parquet",  0.20),
                "code" : ("data/code/*.parquet",  0.20),
                "books": ("data/books/*.parquet", 0.40),
                "news" : ("data/news/*.parquet",  0.20),
            }
        Lưu ý: sum(probabilities) phải = 1.0

    stopping_strategy:
        "first_exhausted" — dừng khi source nào hết trước
        "all_exhausted"   — oversample source nhỏ cho đến khi tất cả hết

    shuffle_buffer: số sample giữ trong buffer để shuffle chéo giữa các source
    """
    sources          : dict = field(default_factory=dict)
    stopping_strategy: str  = "first_exhausted"
    shuffle_buffer   : int  = 10_000


@dataclass
class ModelConfig:
    """Kiến trúc model."""
    vocab_size : int   = 64001
    d_model    : int   = 512
    n_heads    : int   = 8
    n_layers   : int   = 8
    max_seq    : int   = 512
    dropout    : float = 0.1
    rope_base  : float = 10000.0


@dataclass
class DataConfig:
    """Cấu hình dữ liệu và cách load incremental."""
    # source: "wikipedia" | "vtsnlp" | "parquet" | "mix"
    source         : str = "wikipedia"

    dataset_name   : str = "wikimedia/wikipedia"
    dataset_subset : str = "20231101.vi"

    vtsnlp_domains : list = None

    # Local parquet
    parquet_path     : str = ""
    parquet_text_col : str = "text"

    # Mix config — chỉ dùng khi source="mix"
    mix: MixConfig = field(default_factory=MixConfig)

    # Sequential / sliding window mode
    sequential_mode : bool = False
    window_stride   : int  = None

    chunk_size     : int   = 10_000
    seg_len        : int   = 512
    min_text_len   : int   = 200
    val_ratio      : float = 0.01
    cache_dir      : str   = "./data_cache"


@dataclass
class TrainConfig:
    """Cấu hình quá trình train."""
    batch_size        : int   = 8
    grad_accum        : int   = 4
    lr                : float = 3e-4
    warmup_steps      : int   = 100
    weight_decay      : float = 0.01
    max_grad_norm     : float = 1.0
    epochs_per_chunk  : int   = 1
    total_chunks      : int   = -1    # -1 = train hết toàn bộ dataset

    # Cosine annealing with warm restarts (SGDR)
    lr_decay_cycle_steps : int   = 10_000
    lr_min_ratio         : float = 0.1

    log_every  : int = 100
    eval_every : int = 500
    save_every : int = 1000

    save_dir    : str           = "./checkpoints"
    resume_from : Optional[str] = None

    device          : str  = "cuda"
    mixed_precision : bool = True

    # HuggingFace Hub upload
    hf_repo_id : Optional[str] = None
    hf_token   : Optional[str] = None


@dataclass
class TokenizerConfig:
    """Cấu hình tokenizer."""
    pretrained_name  : str  = "custom_tokenizer"
    use_fast         : bool = True
    strict_chart_mode: bool = True
    n_price_bins     : int  = 1024


@dataclass
class Config:
    """Gộp tất cả config con."""
    model     : ModelConfig     = field(default_factory=ModelConfig)
    data      : DataConfig      = field(default_factory=DataConfig)
    train     : TrainConfig     = field(default_factory=TrainConfig)
    tokenizer : TokenizerConfig = field(default_factory=TokenizerConfig)
    seed      : int             = 42


def get_default_config() -> Config:
    return Config()


def get_small_config() -> Config:
    """Config nhỏ để test nhanh trên máy yếu / Colab free tier."""
    cfg = Config()
    cfg.model.d_model    = 64
    cfg.model.n_heads    = 2
    cfg.model.n_layers   = 2
    cfg.model.max_seq    = 64
    cfg.data.chunk_size  = 2_000
    cfg.data.seg_len     = 64
    cfg.train.batch_size = 4
    return cfg


def get_100m_config() -> Config:
    """Config ~100M params cho Colab T4."""
    cfg = Config()
    cfg.model.d_model  = 512
    cfg.model.n_heads  = 8
    cfg.model.n_layers = 8
    cfg.model.max_seq  = 512

    cfg.data.chunk_size = 10_000

    cfg.train.lr                   = 3e-4
    cfg.train.warmup_steps         = 200
    cfg.train.lr_decay_cycle_steps = 9_800
    cfg.train.lr_min_ratio         = 0.1
    cfg.train.batch_size           = 32
    cfg.train.grad_accum           = 64
    cfg.train.total_chunks         = -1
    return cfg


def get_110m_config() -> Config:
    """Config ~110M params tối ưu cho Colab T4."""
    cfg = Config()
    cfg.model.d_model  = 512
    cfg.model.n_heads  = 8
    cfg.model.n_layers = 30
    cfg.model.max_seq  = 512

    cfg.data.chunk_size  = 20_000
    cfg.data.seg_len     = 512
    
    # batch 32, grad_accum 64 → effective batch 2048, LR 3e-4, warmup 200 steps, decay cycle 10_000 steps
    # estimated tokens: 32 * 64 * 10_000 * 512 = 10,485,760,000 tokens → ~10B tokens
    cfg.train.lr                    = 3e-4
    cfg.train.warmup_steps          = 200
    cfg.train.lr_decay_cycle_steps  = 10_000   
    cfg.train.lr_min_ratio          = 0.1
    cfg.train.batch_size  = 32
    cfg.train.grad_accum  = 64
    cfg.train.total_chunks          = -1
    return cfg