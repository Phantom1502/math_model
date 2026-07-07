"""
app/llama/benchmark.py — Benchmark cho HF LlamaForCausalLM
================================================================
Bench DATA (SEMANTIC_BENCH, ENTITY_BENCH, FACT_BENCH, OOD_BENCH,
LANGUAGE_PROMPTS, WEIGHTS) tái sử dụng NGUYÊN VẸN từ app/memlm/benchmark.py —
nội dung câu hỏi tiếng Việt không phụ thuộc kiến trúc model, không cần viết
lại. Chỉ tầng GỌI MODEL/TOKENIZER được viết lại theo convention HF chuẩn.

Khác bản cũ (app/memlm/benchmark.py):
    - encode  : tokenizer(text, add_special_tokens=False)["input_ids"]
                (HF tokenizer thật — không cần _split_segments thủ công)
    - forward : model(input_ids=ids, attention_mask=mask).logits
                (KHÔNG cần tự build causal_mask — LlamaModel tự sinh causal
                mask nội bộ dựa trên attention_mask)
    - generate: model.generate() dùng KV-cache — nhanh hơn nhiều so với vòng
                lặp forward lại toàn bộ sequence mỗi step của bản cũ
                (app/memlm/generate.py hiện KHÔNG có cache).

Nếu về sau xoá hẳn nhánh app/memlm/, copy trực tiếp các list hằng số đó từ
app/memlm/benchmark.py sang đây để bỏ dependency chéo.

LƯU Ý — benchmark_ict.py CHƯA được rewrite ở đây:
    File benchmark_ict.py (đánh giá Swept/FVG/Shift pattern) không nằm trong
    bộ tài liệu được cung cấp nên KHÔNG thể viết lại chính xác ở đây. Khi
    rewrite, cần đổi tối thiểu 3 điểm để tương thích nhánh Llama:
        1. encode/forward theo convention HF như file này (không causal_mask
           thủ công, không _split_segments).
        2. Ground truth so khớp: nếu benchmark_ict so sánh completion text
           dạng "O_512 ..." với build_facts(), cần convert_legacy_price_tokens()
           (app/llama/tokenizer.py) TRƯỚC khi encode prompt/completion, vì
           price token giờ là "<px_O_512>" chứ không còn "O_512" trần.
        3. max_seq nên lấy từ cfg.llama.max_position_embeddings (2048) thay
           vì cfg.model.max_seq (512) — ảnh hưởng đến việc cắt prompt dài.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class BenchItem:
    prompt   : str
    positive : List[str]
    negative : List[str]
    note     : str = ""

# ══════════════════════════════════════════════════════════════════════════
# Benchmark data
# ══════════════════════════════════════════════════════════════════════════

SEMANTIC_BENCH: List[BenchItem] = [
    BenchItem("Con mèo là", ["động vật có vú", "thú nuôi", "sinh vật"], ["loài chim", "loài cá", "thực vật", "côn trùng"]),
    BenchItem("Albert Einstein là", ["nhà vật lý", "nhà khoa học", "học giả"], ["nhà hóa học", "nhà toán học", "nhà triết học", "nhà văn"]),
    BenchItem("Hà Nội là", ["thành phố", "thủ đô", "đô thị lớn"], ["thị trấn", "làng quê", "vùng nông thôn", "tỉnh lẻ"]),
    BenchItem("Python là", ["ngôn ngữ lập trình", "công cụ lập trình"], ["ngôn ngữ tự nhiên", "ngôn ngữ đánh dấu", "ngôn ngữ truy vấn"]),
    BenchItem("Sông Hồng là", ["con sông", "dòng sông"], ["hồ nước", "vịnh biển", "suối nhỏ", "kênh đào"]),
    BenchItem("Bóng đá là", ["môn thể thao tập thể", "trò chơi vận động"], ["môn thể thao cá nhân", "trò chơi điện tử", "bộ môn nghệ thuật"]),
    BenchItem("Mặt Trời là", ["ngôi sao", "thiên thể phát sáng"], ["hành tinh", "vệ tinh", "sao lùn trắng", "lỗ đen"]),
    BenchItem("Bác sĩ là", ["chuyên gia y tế", "người hành nghề y"], ["y tá", "dược sĩ", "kỹ thuật viên xét nghiệm", "hộ lý"]),
    BenchItem("Piano là", ["nhạc cụ", "nhạc cụ có phím"], ["nhạc cụ có dây", "nhạc cụ hơi", "nhạc cụ gõ màng", "nhạc cụ điện tử"]),
    BenchItem("Tiểu thuyết là", ["tác phẩm văn học", "thể loại văn xuôi dài"], ["truyện ngắn", "bài thơ", "kịch bản", "tản văn"]),
    BenchItem("Luật sư là", ["người hành nghề pháp lý", "chuyên gia pháp luật"], ["thẩm phán", "công tố viên", "thư ký tòa án", "cảnh sát"]),
    BenchItem("Muỗi là", ["côn trùng", "động vật chân đốt"], ["động vật có vú", "loài chim", "bò sát", "động vật thân mềm"]),
    BenchItem("Đái tháo đường là", ["bệnh rối loạn chuyển hóa", "bệnh mãn tính"], ["bệnh truyền nhiễm", "bệnh ung thư", "bệnh tim mạch", "bệnh hô hấp"]),
    BenchItem("Kiến trúc sư là", ["người thiết kế công trình", "chuyên gia xây dựng"], ["kỹ sư kết cấu", "thợ nề", "nhà điêu khắc", "kỹ sư điện"]),
    BenchItem("Lúa là", ["cây lương thực", "thực vật", "cây trồng"], ["cây cảnh", "cây dược liệu", "cây công nghiệp", "cây ăn quả"]),
]

ENTITY_BENCH: List[BenchItem] = [
    BenchItem("Albert Einstein là", ["nhà vật lý", "nhà khoa học"], ["nhà hóa học", "nhà văn", "ca sĩ", "vận động viên"]),
    BenchItem("Marie Curie là", ["nhà khoa học", "nhà vật lý", "nhà hóa học"], ["nhà văn", "ca sĩ", "diễn viên", "chính trị gia"]),
    BenchItem("Hồ Chí Minh là", ["chính trị gia", "lãnh tụ", "nhà cách mạng"], ["nhà khoa học", "nhạc sĩ", "vận động viên", "nhà văn"]),
    BenchItem("William Shakespeare là", ["nhà văn", "kịch tác gia", "nhà thơ"], ["nhà khoa học", "chính trị gia", "nhà thám hiểm", "nhạc sĩ"]),
    BenchItem("Mozart là", ["nhạc sĩ", "nhà soạn nhạc"], ["nhà vật lý", "cầu thủ", "bác sĩ", "nhà văn"]),
    BenchItem("Leonardo da Vinci là", ["họa sĩ", "nhà phát minh", "nghệ sĩ"], ["ca sĩ", "vận động viên", "phi hành gia", "chính trị gia"]),
    BenchItem("Charles Darwin là", ["nhà sinh học", "nhà khoa học", "nhà tự nhiên học"], ["ca sĩ", "nhà thơ", "cầu thủ bóng đá", "diễn viên"]),
    BenchItem("Nikola Tesla là", ["nhà phát minh", "kỹ sư điện"], ["ca sĩ", "nhà thơ", "vận động viên", "diễn viên"]),
    BenchItem("Hà Nội là thủ đô của", ["Việt Nam", "nước Việt Nam"], ["Trung Quốc", "Nhật Bản", "Thái Lan", "Campuchia"]),
    BenchItem("Tokyo là thủ đô của", ["Nhật Bản", "nước Nhật"], ["Hàn Quốc", "Trung Quốc", "Thái Lan", "Singapore"]),
    BenchItem("Berlin là thủ đô của", ["Đức", "nước Đức"], ["Pháp", "Áo", "Ba Lan", "Bỉ"]),
    BenchItem("Tháp Eiffel nằm ở", ["Pháp", "Paris"], ["Đức", "Ý", "Tây Ban Nha", "Anh"]),
    BenchItem("Sông Nile chảy qua", ["Ai Cập", "Bắc Phi"], ["Việt Nam", "Nhật Bản", "Brazil", "Ấn Độ"]),
    BenchItem("Sydney nằm ở", ["Úc", "Australia"], ["Canada", "Brazil", "Ấn Độ", "Nam Phi"]),
    BenchItem("Angkor Wat nằm ở", ["Campuchia", "Đông Nam Á"], ["Thái Lan", "Việt Nam", "Myanmar", "Lào"]),
    BenchItem("Sông Amazon chảy qua", ["Nam Mỹ", "Brazil"], ["châu Phi", "châu Á", "Bắc Mỹ", "châu Âu"]),
    BenchItem("Hổ là loài động vật", ["ăn thịt", "nguy hiểm", "thuộc họ mèo lớn"], ["ăn cỏ", "bay được", "sống dưới nước", "sống ở Bắc Cực"]),
    BenchItem("Cá heo là loài động vật", ["có vú", "thông minh", "sống dưới nước"], ["bò sát", "lưỡng cư", "côn trùng", "chim"]),
    BenchItem("Đại bàng là loài", ["chim", "chim săn mồi", "động vật có cánh"], ["thú", "bò sát", "cá", "côn trùng"]),
    BenchItem("Cá mập là loài", ["cá", "động vật săn mồi biển"], ["động vật có vú", "chim biển", "bò sát", "động vật giáp xác"]),
    BenchItem("Ngôn ngữ Python thường được dùng để", ["lập trình", "phân tích dữ liệu", "xây dựng ứng dụng"], ["nấu ăn", "leo núi", "chơi thể thao", "vẽ tranh sơn dầu"]),
    BenchItem("Trí tuệ nhân tạo là lĩnh vực thuộc", ["khoa học máy tính", "công nghệ thông tin"], ["y học", "nông nghiệp", "nghệ thuật truyền thống", "thể thao"]),
    BenchItem("Máy tính được dùng để", ["xử lý thông tin", "tính toán", "lưu trữ dữ liệu"], ["trồng cây", "chữa bệnh", "xây nhà", "nấu ăn"]),
    BenchItem("Điện thoại thông minh là", ["thiết bị điện tử", "công cụ liên lạc", "thiết bị di động"], ["dụng cụ nấu ăn", "nhạc cụ", "phương tiện giao thông", "vũ khí"]),
    BenchItem("Mặt Trăng là", ["vệ tinh của Trái Đất", "thiên thể tự nhiên"], ["ngôi sao", "hành tinh độc lập", "tiểu hành tinh", "sao chổi"]),
    BenchItem("Hành tinh Sao Hỏa có màu", ["đỏ", "đỏ cam"], ["xanh lam", "vàng kim", "trắng bạch", "đen"]),
]

FACT_BENCH: List[BenchItem] = [
    BenchItem("Thủ đô của Việt Nam là", [" Hà Nội"], [" Thành phố Hồ Chí Minh", " Đà Nẵng", " Huế", " Cần Thơ"]),
    BenchItem("Thủ đô của Pháp là", [" Paris"], [" London", " Berlin", " Rome", " Madrid"]),
    BenchItem("Nước láng giềng phía Bắc của Việt Nam là", [" Trung Quốc"], [" Lào", " Campuchia", " Thái Lan", " Myanmar"]),
    BenchItem("Thủ đô của Nhật Bản là", [" Tokyo"], [" Osaka", " Kyoto", " Hiroshima", " Nagoya"]),
    BenchItem("Ký hiệu hóa học của vàng là", [" Au"], [" Ag", " Fe", " Cu", " Pt"]),
    BenchItem("Ký hiệu hóa học của bạc là", [" Ag"], [" Au", " Fe", " Cu", " Zn"]),
    BenchItem("Đơn vị đo nhiệt độ trong hệ SI là", [" Kelvin"], [" Celsius", " Fahrenheit", " Rankine"]),
    BenchItem("Tốc độ ánh sáng trong chân không xấp xỉ", [" 300.000 km/s", " 3×10⁸ m/s"], [" 1.000 km/s", " 30.000 km/s", " 3.000.000 km/s"]),
    BenchItem("Nước sôi ở nhiệt độ", [" 100 độ Celsius", " 100°C"], [" 0 độ Celsius", " 37 độ Celsius", " 200 độ Celsius"]),
    BenchItem("Đơn vị cơ bản đo khối lượng trong hệ SI là", [" kilogram"], [" gram", " pound", " ounce", " tấn"]),
    BenchItem("Hành tinh thứ ba tính từ Mặt Trời là", [" Trái Đất"], [" Sao Hỏa", " Sao Kim", " Sao Mộc", " Sao Thủy"]),
    BenchItem("Hành tinh lớn nhất trong hệ Mặt Trời là", [" Sao Mộc"], [" Sao Thổ", " Sao Hải Vương", " Trái Đất", " Sao Hỏa"]),
    BenchItem("Hành tinh gần Mặt Trời nhất là", [" Sao Thủy"], [" Sao Kim", " Trái Đất", " Sao Hỏa", " Sao Mộc"]),
    BenchItem("Trái Đất quay quanh", [" Mặt Trời"], [" Mặt Trăng", " Sao Hỏa", " Sao Mộc", " trục của nó"]),
    BenchItem("DNA là viết tắt của", [" Deoxyribonucleic Acid"], [" Digital Network Access", " Dynamic Numeric Array", " Direct Neural Architecture"]),
    BenchItem("Quang hợp là quá trình", [" thực vật tổng hợp chất hữu cơ từ ánh sáng", " chuyển hóa năng lượng ánh sáng"], [" động vật tiêu hóa thức ăn", " vi khuẩn phân hủy chất hữu cơ", " con người hít thở oxygen"]),
    BenchItem("Chiến tranh thế giới thứ hai kết thúc vào năm", [" 1945"], [" 1918", " 1939", " 1950", " 1975"]),
    BenchItem("Số nguyên tố nhỏ nhất là", [" 2"], [" 1", " 3", " 5", " 0"]),
]

LANGUAGE_PROMPTS = [
    "Con mèo là",
    "Albert Einstein là",
    "Trí tuệ nhân tạo là",
    "Máy tính là",
    "Hà Nội là thành phố",
    "Sông Hồng chảy qua",
    "Bóng đá là môn thể thao",
    "Khoa học máy tính là",
    "Văn học Việt Nam có",
    "Lịch sử Việt Nam bắt đầu",
]

OOD_BENCH: List[BenchItem] = [
    BenchItem("Robot là", ["máy móc tự động", "thiết bị cơ điện tử"], ["sinh vật sống", "loài động vật", "thực vật", "khoáng vật"]),
    BenchItem("Blockchain là", ["công nghệ lưu trữ dữ liệu phân tán", "chuỗi khối"], ["loài động vật", "địa danh", "môn thể thao", "nhạc cụ"]),
    BenchItem("Vaccine là", ["chế phẩm sinh học phòng bệnh", "thuốc phòng ngừa"], ["loại thực phẩm", "loại máy móc", "vũ khí", "dụng cụ thể thao"]),
    BenchItem("Năng lượng mặt trời là", ["nguồn năng lượng tái tạo", "năng lượng sạch"], ["nhiên liệu hóa thạch", "khoáng sản", "thực phẩm", "vũ khí"]),
    BenchItem("Mạng xã hội là", ["nền tảng kết nối trực tuyến", "dịch vụ internet"], ["mạng lưới điện", "mạng giao thông đường bộ", "loài sinh vật", "loại thực phẩm"]),
    BenchItem("Trí tuệ nhân tạo tổng quát là", ["hệ thống AI có khả năng tổng quát", "công nghệ AI tiên tiến"], ["loài động vật", "địa danh", "môn thể thao", "phương tiện giao thông"]),
    BenchItem("Zorb là một loài động vật. Mọi động vật đều là sinh vật. Zorb là", ["sinh vật"], ["quốc gia", "thành phố", "hành tinh", "phần mềm"]),
    BenchItem("Mọi flar đều là zent. Mọi zent đều là sinh vật. Blen là một flar. Blen là", ["sinh vật"], ["quốc gia", "hành tinh", "phần mềm", "thiên hà"]),
    BenchItem("Kira là bác sĩ. Mọi bác sĩ đều làm việc trong ngành y tế. Kira thuộc", ["ngành y tế"], ["ngành nông nghiệp", "ngành thể thao", "ngành hàng không", "ngành giáo dục"]),
    BenchItem("Mọi nori đều là phương tiện. Mọi phương tiện đều được dùng để di chuyển. Teka là một nori. Teka được dùng để", ["di chuyển"], ["quang hợp", "săn mồi", "lập trình", "nấu ăn"]),
    BenchItem("Mọi drako đều là máy móc. Mọi máy móc đều phục vụ một mục đích. Lena là một drako. Lena là", ["máy móc", "công cụ"], ["động vật", "thực vật", "hành tinh", "cảm xúc"]),
    BenchItem("An là kỹ sư. Mọi kỹ sư đều là người lao động. Mọi người lao động đều có thu nhập. An có", ["thu nhập"], ["vây cá", "cánh chim", "vỏ cây", "nhiệt độ sôi"]),
]

WEIGHTS = {"semantic": 0.20, "entity": 0.20, "fact": 0.30, "language": 0.20, "ood": 0.10}
# ══════════════════════════════════════════════════════════════════════════
# Core scoring — HF convention
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def avg_logprob_per_token(
    model, tokenizer, prompt: str, completion: str, device, max_seq: int = 2048,
) -> float:
    prompt_ids     = tokenizer(prompt,     add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

    if not completion_ids:
        return float("-inf")

    full_ids = prompt_ids + completion_ids
    if len(full_ids) > max_seq:
        keep = max_seq - len(completion_ids)
        if keep <= 0:
            return float("-inf")
        full_ids = prompt_ids[-keep:] + completion_ids

    ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    mask  = torch.ones_like(ids_t)

    logits    = model(input_ids=ids_t, attention_mask=mask, use_cache=False).logits[0]
    log_probs = F.log_softmax(logits.float(), dim=-1)

    n_prompt = ids_t.size(1) - len(completion_ids)
    total_lp = sum(
        log_probs[n_prompt + i - 1, tok_id].item()
        for i, tok_id in enumerate(completion_ids)
        if n_prompt + i - 1 >= 0
    )
    return total_lp / len(completion_ids)


@torch.no_grad()
def score_item(model, tokenizer, item: BenchItem, device, max_seq: int = 2048) -> dict:
    pos_scores = [avg_logprob_per_token(model, tokenizer, item.prompt, p, device, max_seq)
                  for p in item.positive]
    neg_scores = [avg_logprob_per_token(model, tokenizer, item.prompt, n, device, max_seq)
                  for n in item.negative]
    pos_mean = sum(pos_scores) / len(pos_scores)
    neg_mean = sum(neg_scores) / len(neg_scores)
    return {
        "pos_mean": pos_mean, "neg_mean": neg_mean, "score": pos_mean - neg_mean,
        "pos_scores": pos_scores, "neg_scores": neg_scores, "prompt": item.prompt,
    }


def run_logprob_benchmark(
    model, tokenizer, bench, level_name, device, max_seq: int = 2048, verbose: bool = True,
) -> float:
    model.eval()
    scores = []

    if verbose:
        print(f"\n{'─'*60}\n  [{level_name.upper()}]\n{'─'*60}")

    for item in bench:
        r = score_item(model, tokenizer, item, device, max_seq)
        scores.append(r["score"])

        if verbose:
            pos_fmt = "  ".join(f"{s:+.2f}" for s in r["pos_scores"])
            neg_fmt = "  ".join(f"{s:+.2f}" for s in r["neg_scores"])
            status  = "✓" if r["score"] > 0 else "✗"
            print(f"\n  prompt : {item.prompt!r}")
            print(f"  pos_lp : [{pos_fmt}]  mean={r['pos_mean']:+.3f}")
            print(f"  neg_lp : [{neg_fmt}]  mean={r['neg_mean']:+.3f}")
            print(f"  score  : {r['score']:+.3f}  {status}")

    avg    = sum(scores) / len(scores)
    n_pass = sum(1 for s in scores if s > 0)

    if verbose:
        print(f"\n  ► {level_name} avg_score = {avg:+.3f}  |  pass = {n_pass}/{len(scores)}")

    return avg


# ══════════════════════════════════════════════════════════════════════════
# Language quality — model.generate() với KV-cache
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_sample(
    model, tokenizer, prompt: str, max_new: int = 50,
    temperature: float = 0.8, top_k: int = 50, device=None,
) -> list:
    device     = device or next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ids        = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=pad_id,
    )
    return out[0, ids.size(1):].tolist()


def _compute_language_metrics(token_seqs: list) -> dict:
    all_tokens, all_bigrams = [], []
    repeat_count = total_tokens = 0

    for seq in token_seqs:
        all_tokens.extend(seq)
        total_tokens += len(seq)
        for i in range(len(seq) - 1):
            all_bigrams.append((seq[i], seq[i + 1]))
            if seq[i] == seq[i + 1]:
                repeat_count += 1

    distinct1 = len(set(all_tokens))  / max(len(all_tokens),  1)
    distinct2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    repeat    = repeat_count / max(total_tokens - 1, 1)

    return {
        "distinct1": distinct1, "distinct2": distinct2, "repeat_ratio": repeat,
        "language_score": (distinct1 + distinct2) / 2 - repeat,
    }


def run_language_benchmark(
    model, tokenizer, prompts=None, n_samples: int = 5,
    max_new: int = 50, device=None, verbose: bool = True,
) -> dict:
    model.eval()
    prompts = prompts or LANGUAGE_PROMPTS
    device  = device or next(model.parameters()).device

    if verbose:
        print(f"\n{'─'*60}\n  [LANGUAGE QUALITY]\n{'─'*60}")

    all_seqs = []
    for prompt in prompts:
        for _ in range(n_samples):
            all_seqs.append(_generate_sample(model, tokenizer, prompt, max_new, device=device))

    metrics = _compute_language_metrics(all_seqs)

    if verbose:
        print(f"\n  distinct-1   : {metrics['distinct1']:.4f}")
        print(f"  distinct-2   : {metrics['distinct2']:.4f}")
        print(f"  repeat_ratio : {metrics['repeat_ratio']:.4f}")
        print(f"\n  ► language_score = {metrics['language_score']:+.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════
# run_all
# ══════════════════════════════════════════════════════════════════════════

def run_all(model, tokenizer, cfg, verbose: bool = True, step=None, n_language_samples: int = 5) -> dict:
    device  = next(model.parameters()).device
    max_seq = cfg.llama.max_position_embeddings

    if verbose:
        step_str = f"step {step}" if step is not None else "checkpoint"
        print(f"\n{'═'*60}\n  BENCHMARK  —  {step_str}\n{'═'*60}")

    sem = run_logprob_benchmark(model, tokenizer, SEMANTIC_BENCH, "semantic", device, max_seq, verbose)
    ent = run_logprob_benchmark(model, tokenizer, ENTITY_BENCH,   "entity",   device, max_seq, verbose)
    fct = run_logprob_benchmark(model, tokenizer, FACT_BENCH,     "fact",     device, max_seq, verbose)
    ood = run_logprob_benchmark(model, tokenizer, OOD_BENCH,      "ood",      device, max_seq, verbose)

    lang_metrics = run_language_benchmark(
        model, tokenizer, n_samples=n_language_samples, device=device, verbose=verbose,
    )
    lang = lang_metrics["language_score"]

    total = (
        sem * WEIGHTS["semantic"] + ent * WEIGHTS["entity"] + fct * WEIGHTS["fact"]
        + lang * WEIGHTS["language"] + ood * WEIGHTS["ood"]
    )

    if verbose:
        print(f"\n{'═'*60}\n  SUMMARY\n{'─'*60}")
        print(f"  semantic  (×{WEIGHTS['semantic']:.2f}) : {sem:+.3f}")
        print(f"  entity    (×{WEIGHTS['entity']:.2f}) : {ent:+.3f}")
        print(f"  fact      (×{WEIGHTS['fact']:.2f}) : {fct:+.3f}")
        print(f"  language  (×{WEIGHTS['language']:.2f}) : {lang:+.4f}  "
              f"[d1={lang_metrics['distinct1']:.3f} d2={lang_metrics['distinct2']:.3f} "
              f"rep={lang_metrics['repeat_ratio']:.3f}]")
        print(f"  ood       (×{WEIGHTS['ood']:.2f}) : {ood:+.3f}")
        print(f"{'─'*60}\n  TOTAL               : {total:+.3f}\n{'═'*60}\n")

    return {
        "semantic": sem, "entity": ent, "fact": fct, "language": lang, "ood": ood, "total": total,
        "distinct1": lang_metrics["distinct1"], "distinct2": lang_metrics["distinct2"],
        "repeat_ratio": lang_metrics["repeat_ratio"],
    }


# ══════════════════════════════════════════════════════════════════════════
# Compare checkpoints
# ══════════════════════════════════════════════════════════════════════════

def compare_checkpoints(checkpoint_dirs: list, cfg, verbose: bool = False) -> None:
    """
    checkpoint_dirs: list thư mục HF save_pretrained() (vd "checkpoints_llama/step_5000").
    """
    from transformers import LlamaForCausalLM, AutoTokenizer

    rows = []
    for path in checkpoint_dirs:
        print(f"\n── Loading {path} ──")
        model     = LlamaForCausalLM.from_pretrained(path).to(cfg.train.device)
        tokenizer = AutoTokenizer.from_pretrained(path)
        r = run_all(model, tokenizer, cfg, verbose=verbose)
        r["checkpoint"] = path
        rows.append(r)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'═'*88}")
    print(f"  {'CHECKPOINT':<30} {'sem':>6} {'ent':>6} {'fact':>6} {'lang':>6} {'ood':>6} {'total':>8}")
    print(f"{'─'*88}")
    for r in rows:
        name = r["checkpoint"].rstrip("/").split("/")[-1]
        print(f"  {name:<30} {r['semantic']:>+6.2f} {r['entity']:>+6.2f} "
              f"{r['fact']:>+6.2f} {r['language']:>+6.3f} {r['ood']:>+6.2f} {r['total']:>+8.3f}")
    print(f"{'═'*88}")


if __name__ == "__main__":
    import sys
    from transformers import LlamaForCausalLM, AutoTokenizer
    from config import Config

    paths = sys.argv[1:]
    if not paths:
        print("Usage: python benchmark.py <checkpoint_dir> [checkpoint_dir2 ...]")
        sys.exit(1)

    cfg = Config()
    if len(paths) == 1:
        model     = LlamaForCausalLM.from_pretrained(paths[0]).to(cfg.train.device)
        tokenizer = AutoTokenizer.from_pretrained(paths[0])
        run_all(model, tokenizer, cfg, verbose=True)
    else:
        compare_checkpoints(paths, cfg, verbose=False)