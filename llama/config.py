"""
app/llama/config.py — Config cho nhánh pretrain dùng HF LlamaForCausalLM chuẩn
==================================================================================
Khác nhánh app/memlm/ (kiến trúc tự viết), nhánh này dùng thẳng
transformers.LlamaConfig + LlamaForCausalLM để:
    - Tương thích native với HF Trainer, TRL (SFTTrainer/DPOTrainer/RewardTrainer)
    - GQA (num_key_value_heads) + KV-cache khi generate/benchmark
    - resize_token_embeddings an toàn khi vocab thay đổi (thêm price token thật)

vocab_size trong LlamaConfig chỉ là PLACEHOLDER — bị ghi đè bởi
model.build_model(cfg, tokenizer) ngay sau khi tokenizer thật được build
(xem tokenizer.py: build_llama_tokenizer). KHÔNG tự sửa tay giá trị này khi
đổi price bin count — sửa cfg.tokenizer.n_price_bins là đủ.
"""

from dataclasses import dataclass, field
from typing import Optional
from transformers import LlamaConfig


@dataclass
class DataConfig:
    # source: "wikipedia" | "vtsnlp" | "parquet" | "mix"
    source           : str  = "wikipedia"

    dataset_name     : str  = "wikimedia/wikipedia"
    dataset_subset   : str  = "20231101.vi"
    vtsnlp_domains   : Optional[list] = None

    # Local parquet (dùng cho nguồn "parquet", vd data trading chart/action/books)
    parquet_path      : str = ""
    parquet_text_col  : str = "text"

    # Mix config — chỉ dùng khi source="mix"
    mix_sources   : dict = field(default_factory=dict)   # {name: (glob_pattern, prob)}
    mix_stopping  : str  = "first_exhausted"              # | "all_exhausted"

    sequential_mode : bool = False
    window_stride   : Optional[int] = None

    chunk_size    : int   = 10_000
    seg_len       : int   = 512
    min_text_len  : int   = 200
    val_ratio     : float = 0.01


@dataclass
class TrainConfig:
    batch_size            : int   = 16
    grad_accum             : int   = 32
    lr                      : float = 3e-4
    warmup_steps             : int   = 500
    weight_decay              : float = 0.01
    max_grad_norm              : float = 1.0

    # Cosine annealing with warm restarts (SGDR) — giữ nguyên convention cũ
    lr_decay_cycle_steps : int   = 50_000
    lr_min_ratio          : float = 0.1

    total_chunks : int = -1     # -1 = train hết toàn bộ dataset

    log_every  : int = 100
    eval_every : int = 500
    save_every : int = 1000

    save_dir     : str            = "./checkpoints_llama"
    resume_from  : Optional[str]  = None   # thư mục HF save_pretrained() (chunk_N/ hoặc best/)

    device            : str  = "cuda"
    mixed_precision   : bool = True
    bf16              : bool = True   # ưu tiên bf16 nếu GPU hỗ trợ (Ampere+), fallback fp16

    hf_repo_id : Optional[str] = None
    hf_token   : Optional[str] = None


@dataclass
class TokenizerConfig:
    """
    base_tokenizer_dir : tokenizer BPE gốc, train bằng app/memlm/scripts/train_tokenizer.py
    output_dir         : nơi lưu bản đã add_tokens() price vocab THẬT
                         (khác app/memlm/tokenizer.py — không cộng ID ảo bên ngoài nữa)
    n_price_bins       : số bin mỗi kênh O/H/L/C (mặc định 1024, khớp ChartCodec)
    """
    base_tokenizer_dir : str = "custom_tokenizer"
    output_dir          : str = "custom_tokenizer_llama"
    n_price_bins         : int = 1024


@dataclass
class Config:
    llama     : Optional[LlamaConfig] = None
    data      : DataConfig            = field(default_factory=DataConfig)
    train     : TrainConfig           = field(default_factory=TrainConfig)
    tokenizer : TokenizerConfig       = field(default_factory=TokenizerConfig)
    seed      : int = 42

    def __post_init__(self):
        if self.llama is None:
            self.llama = get_110m_llama_config()


def get_110m_llama_config() -> LlamaConfig:
    """
    ~110-125M params (cỡ SmolLM2-135M), phù hợp Colab T4 / Kaggle P100.
    vocab_size=24000 là placeholder — sẽ bị build_model() resize đúng theo
    tokenizer thật (base BPE 16k + price vocab ~4098 + special tokens).
    """
    return LlamaConfig(
        vocab_size             = 24_000,
        hidden_size             = 576,
        intermediate_size        = 1536,
        num_hidden_layers         = 22,
        num_attention_heads        = 9,
        num_key_value_heads         = 3,     # GQA: 9 Q head / 3 KV group
        max_position_embeddings      = 2048,
        rms_norm_eps                  = 1e-5,
        rope_theta                     = 10000.0,
        tie_word_embeddings              = True,
        use_cache                         = True,
        attention_bias                     = False,
        mlp_bias                            = False,
    )


def get_small_llama_config() -> LlamaConfig:
    """Config nhỏ để test nhanh pipeline (không dùng để train thật)."""
    return LlamaConfig(
        vocab_size             = 24_000,
        hidden_size              = 64,
        intermediate_size         = 128,
        num_hidden_layers          = 2,
        num_attention_heads         = 2,
        num_key_value_heads           = 1,
        max_position_embeddings        = 128,
        tie_word_embeddings              = True,
        use_cache                         = True,
    )


def get_default_config() -> Config:
    return Config()