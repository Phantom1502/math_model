"""
tokenizer.py — Wrapper cho BPE tokenizer + Price token riêng biệt
=============================================================================
Hỗ trợ 2 chế độ base tokenizer:

    1. Custom BPE (khuyến nghị) — train bằng scripts/train_tokenizer.py
       ByteLevel BPE 16k vocab, train trên Wikipedia + VTSNLP tiếng Việt.
       cfg.tokenizer.pretrained_name = "custom_tokenizer"  (path local)
       cfg.tokenizer.use_fast        = True

    2. PhoBERT (legacy) — vinai/phobert-base, 64k vocab, chỉ Slow tokenizer.
       cfg.tokenizer.pretrained_name = "vinai/phobert-base"
       cfg.tokenizer.use_fast        = False  (PhoBERT không có Fast)

Tổng vocab với custom BPE:
    BPE base   : 16,000
    Price vocab:  4,098  (O/H/L/C × 1024 + <chart> + </chart>)
    ─────────────────────
    Total      : ~20,098  → Embedding = 20k × 512 = ~10M params

────────────────────────────────────────────────────────────────────────────
PRICE TOKEN — KHÔNG dùng add_tokens(), dùng vocab RIÊNG BIỆT:

Token dạng "O_512", "H_3", "L_999", "C_0" và marker "<chart>"/"</chart>"
được nhận diện bằng regex, KHÔNG đi qua BPE — mỗi price token là MỘT ID
duy nhất nằm ngoài dải BPE vocab.

⚠️ BẢO VỆ 2 LỚP CHỐNG MATCH NHẦM:

Lớp 1 — strict_chart_mode=True (mặc định):
    CHỈ nhận diện price token khi nằm trong cặp <chart>...</chart>.

Lớp 2 — bin range validation trong _split_segments_loose:
    Token có bin nằm ngoài [0, n_price_bins-1] bị bỏ qua.

────────────────────────────────────────────────────────────────────────────
THAY ĐỔI SO VỚI PHIÊN BẢN PHOBERT:

- use_fast=True hoạt động bình thường với custom BPE (PreTrainedTokenizerFast)
- PhoBERT legacy path vẫn hoạt động — use_fast=False, bỏ qua cảnh báo
- Không thay đổi gì ở price vocab logic, _split_segments, encode/decode
────────────────────────────────────────────────────────────────────────────
"""

import re
import warnings
from transformers import AutoTokenizer


# Regex nhận diện token giá: O_512, H_3, L_999, C_0 ...
PRICE_TOKEN_RE = re.compile(r"<chart>|</chart>|\b[OHLC]_\d{1,4}\b")

# Regex chỉ nhận diện price token KHI nằm trong cặp <chart>...</chart>
_CHART_BLOCK_RE = re.compile(r"<chart>.*?</chart>", re.DOTALL)

# Tên tokenizer PhoBERT legacy — dùng để detect và cảnh báo use_fast
_PHOBERT_NAMES = {"vinai/phobert-base", "vinai/phobert-large"}


def _build_price_vocab(base_len: int, n_bins: int = 1024) -> dict:
    tokens  = [f"{p}_{i}" for p in "OHLC" for i in range(n_bins)]
    tokens += ["<chart>", "</chart>"]      # +2 token marker
    return {tok: base_len + i for i, tok in enumerate(tokens)}


class VietnameseTokenizer:
    def __init__(
        self,
        pretrained_name  : str  = "custom_tokenizer",
        use_fast         : bool = True,
        n_price_bins     : int  = 1024,
        strict_chart_mode: bool = True,
    ):
        """
        pretrained_name  : path local tới custom tokenizer (sau train_tokenizer.py),
                           hoặc "vinai/phobert-base" cho legacy PhoBERT.
        use_fast         : True cho custom BPE (PreTrainedTokenizerFast),
                           False cho PhoBERT (chỉ có Slow tokenizer).
                           Nếu dùng PhoBERT mà truyền use_fast=True, sẽ tự
                           động fallback về False và in cảnh báo.
        n_price_bins     : số bin giá mỗi loại O/H/L/C (default 1024).
        strict_chart_mode: True = chỉ parse price token trong <chart>...</chart>.
        """
        # ── Tự động fallback use_fast cho PhoBERT legacy ─────────────────
        _effective_use_fast = use_fast
        if pretrained_name in _PHOBERT_NAMES and use_fast:
            warnings.warn(
                f"PhoBERT ('{pretrained_name}') không có Fast tokenizer. "
                f"Tự động dùng use_fast=False. "
                f"Để dùng Fast tokenizer, hãy train custom BPE qua "
                f"scripts/train_tokenizer.py.",
                UserWarning,
                stacklevel=2,
            )
            _effective_use_fast = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name,
            use_fast=_effective_use_fast,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.strict_chart_mode = strict_chart_mode
        self.n_price_bins      = n_price_bins
        self.is_phobert        = pretrained_name in _PHOBERT_NAMES

        base_len = len(self.tokenizer)
        self.price_vocab     = _build_price_vocab(base_len, n_price_bins)
        self.price_vocab_inv = {v: k for k, v in self.price_vocab.items()}

        # vocab_size = base BPE + price vocab
        # QUAN TRỌNG: KHÔNG add_tokens() vào self.tokenizer, nên tự cộng
        # tại đây — nếu quên, model build Embedding thiếu ID → IndexError.
        self.vocab_size = base_len + len(self.price_vocab)

        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.unk_id = self.tokenizer.unk_token_id

    # ──────────────────────────────────────────────────────────────────────
    def _split_segments(self, text: str) -> list[tuple[str, str]]:
        """
        Tách text thành list (loại, nội dung) theo thứ tự xuất hiện.
        loại ∈ {"text", "price"}.

        strict_chart_mode=True (mặc định):
            CHỈ nhận diện price token khi nằm trong cặp <chart>...</chart>.

        strict_chart_mode=False:
            Nhận diện price token ở BẤT KỲ ĐÂU.
        """
        if not self.strict_chart_mode:
            return self._split_segments_loose(text)
        return self._split_segments_strict(text)

    def _split_segments_loose(self, text: str) -> list[tuple[str, str]]:
        """
        Parse price token trong đoạn text (không kiểm tra chart boundary).

        Lớp bảo vệ 2: bỏ qua token OHLC có bin nằm ngoài [0, n_price_bins-1].
        """
        segments, pos = [], 0
        for m in PRICE_TOKEN_RE.finditer(text):
            token = m.group(0)

            if token not in ("<chart>", "</chart>"):
                try:
                    if int(token.split("_")[1]) >= self.n_price_bins:
                        continue
                except (IndexError, ValueError):
                    continue

            if m.start() > pos:
                chunk = text[pos:m.start()]
                if chunk.strip():
                    segments.append(("text", chunk))
            segments.append(("price", token))
            pos = m.end()

        if pos < len(text):
            chunk = text[pos:]
            if chunk.strip():
                segments.append(("text", chunk))
        return segments

    def _split_segments_strict(self, text: str) -> list[tuple[str, str]]:
        """Chỉ parse price token bên trong cặp <chart>...</chart>."""
        segments, pos = [], 0
        for block_m in _CHART_BLOCK_RE.finditer(text):
            if block_m.start() > pos:
                chunk = text[pos:block_m.start()]
                if chunk.strip():
                    segments.append(("text", chunk))

            inner_segs = self._split_segments_loose(block_m.group(0))
            segments.extend(inner_segs)

            pos = block_m.end()

        if pos < len(text):
            chunk = text[pos:]
            if chunk.strip():
                segments.append(("text", chunk))
        return segments

    # ──────────────────────────────────────────────────────────────────────
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        segs   = self._split_segments(text)
        chunks = [c for k, c in segs if k == "text"]
        encoded = self.tokenizer(chunks, add_special_tokens=False)["input_ids"] if chunks else []
        it = iter(encoded)

        ids = []
        for kind, content in segs:
            if kind == "text":
                ids.extend(next(it))
            else:
                ids.append(self.price_vocab[content])

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        """
        Tokenize một lần cho nhiều câu — tránh overhead gọi tokenizer lặp lại.
        Fast tokenizer (custom BPE) nhanh hơn đáng kể so với PhoBERT Slow.
        """
        all_segs    = [self._split_segments(t) for t in texts]
        flat_chunks = [c for segs in all_segs for k, c in segs if k == "text"]
        encoded     = self.tokenizer(flat_chunks, add_special_tokens=False)["input_ids"] if flat_chunks else []
        it          = iter(encoded)

        results = []
        for segs in all_segs:
            ids = []
            for kind, content in segs:
                if kind == "text":
                    ids.extend(next(it))
                else:
                    ids.append(self.price_vocab[content])
            if add_special_tokens:
                ids = [self.bos_id] + ids + [self.eos_id]
            results.append(ids)
        return results

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        base_len = len(self.tokenizer)
        pieces, run = [], []
        for i in ids:
            if i >= base_len:
                if run:
                    pieces.append(self.tokenizer.decode(run, skip_special_tokens=skip_special_tokens))
                    run = []
                pieces.append(self.price_vocab_inv.get(i, "<unk_price>"))
            else:
                run.append(i)
        if run:
            pieces.append(self.tokenizer.decode(run, skip_special_tokens=skip_special_tokens))
        return " ".join(p for p in pieces if p)

    def __len__(self):
        return self.vocab_size


def load_tokenizer(cfg) -> VietnameseTokenizer:
    """Entry point để load tokenizer từ TokenizerConfig."""
    return VietnameseTokenizer(
        pretrained_name  = cfg.tokenizer.pretrained_name,
        use_fast         = cfg.tokenizer.use_fast,
        n_price_bins     = cfg.tokenizer.n_price_bins,
        strict_chart_mode= cfg.tokenizer.strict_chart_mode,
    )


if __name__ == "__main__":
    import sys

    tok_path = sys.argv[1] if len(sys.argv) > 1 else "custom_tokenizer"
    print(f"Testing tokenizer: {tok_path}\n")

    tok = VietnameseTokenizer(pretrained_name=tok_path)
    print(f"Vocab size (BPE base + price): {tok.vocab_size:,}")
    print(f"  BPE base  : {len(tok.tokenizer):,}")
    print(f"  Price vocab: {len(tok.price_vocab):,}")
    print()

    # Test text thường
    texts = [
        "Trí tuệ nhân tạo đang thay đổi thế giới.",
        "Albert Einstein là nhà vật lý nổi tiếng.",
        "RSI và MACD là chỉ báo kỹ thuật phổ biến.",
    ]
    for text in texts:
        ids = tok.encode(text)
        dec = tok.decode(ids)
        print(f"  [{len(ids):>3} tok] {text}")
        print(f"           → {dec}")
        print()

    # Test price token trong chart block
    chart_text = "Phân tích: <chart> O_512 H_800 L_300 C_600 </chart> kết thúc."
    ids2 = tok.encode(chart_text)
    dec2 = tok.decode(ids2)
    print(f"  Chart test:")
    print(f"  Input  : {chart_text}")
    print(f"  Tokens : {len(ids2)} ids")
    print(f"  Decoded: {dec2}")

    # Test strict_chart_mode — C_2022 ngoài chart không được parse
    ids3 = tok.encode("Năm C_2022 và O_157 tăng trưởng.")
    dec3 = tok.decode(ids3)
    print(f"\n  Strict mode test (C_2022/O_157 ngoài chart):")
    print(f"  Decoded: {dec3}")
    assert "<unk_price>" not in dec3, "C_2022/O_157 không được là price token!"
    print(f"  ✓ strict_chart_mode OK")