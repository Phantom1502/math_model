"""
app/llama/train_sft_example.py — Ví dụ SFT dùng TRL, chạy thẳng trên checkpoint
pretrain của nhánh này
====================================================================================
Đây là ví dụ MINH HOẠ (khung sườn), không phải script production — mục đích
là chứng minh: vì model/tokenizer ở nhánh này là HF PreTrainedModel/Fast
tokenizer THẬT, TRL SFTTrainer dùng được NGAY, không cần tự viết loss
masking/multi-turn packing tay (khác nếu vẫn dùng MemoryLM tự viết).

Yêu cầu:
    pip install trl>=0.9

Format dữ liệu SFT (theo kế hoạch trong README dự án):
    60% chart+text trading analysis, 25% trading knowledge, 15% general
    Vietnamese conversation — multi-turn, khai thác memory carry-over giữa
    các lượt hội thoại. TRL SFTTrainer hỗ trợ multi-turn qua "messages"
    (list role/content) + chat template — bên dưới dùng chat template ChatML
    đơn giản làm ví dụ, đổi lại cho khớp field thật của dataset SFT khi có.

Ví dụ 1 sample (multi-turn, chart + phân tích):
    {
        "messages": [
            {"role": "user", "content": "Phân tích chart sau: <chart> <px_O_512> ... </chart>"},
            {"role": "assistant", "content": "Chart này cho thấy một Bullish FVG..."},
            {"role": "user", "content": "Vậy có nên vào lệnh BUY không?"},
            {"role": "assistant", "content": "Dựa trên FVG vừa xác định..."}
        ]
    }
"""

from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer

# TRL — cần cài riêng: pip install trl
from trl import SFTTrainer, SFTConfig


# ── Chat template ChatML tối giản (đổi tuỳ theo convention project) ────────
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_pretrained(checkpoint_dir: str):
    """Load model + tokenizer đã pretrain từ nhánh này (thư mục save_pretrained)."""
    model     = LlamaForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Thêm special token hội thoại nếu chưa có trong tokenizer pretrain
    special = ["<|im_start|>", "<|im_end|>"]
    added   = tokenizer.add_tokens([t for t in special if t not in tokenizer.get_vocab()])
    if added:
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Thêm {added} special token hội thoại, resize embeddings → {len(tokenizer):,}")

    tokenizer.chat_template = CHATML_TEMPLATE
    return model, tokenizer


def main(
    pretrain_checkpoint: str = "checkpoints_llama/best",
    sft_dataset_path    : str = "data/sft_dataset.parquet",   # messages: list[dict]
    output_dir          : str = "checkpoints_sft",
    max_seq_length      : int = 2048,
):
    model, tokenizer = load_pretrained(pretrain_checkpoint)

    dataset = load_dataset("parquet", data_files={"train": sft_dataset_path}, split="train")

    sft_config = SFTConfig(
        output_dir                   = output_dir,
        max_length                    = max_seq_length,
        packing                        = False,   # multi-turn nên tắt packing để không lẫn hội thoại
        per_device_train_batch_size      = 4,
        gradient_accumulation_steps       = 16,
        learning_rate                      = 1e-5,   # SFT LR thường thấp hơn pretrain 10-30x
        num_train_epochs                    = 2,
        logging_steps                        = 20,
        save_steps                            = 500,
        bf16                                   = True,
        assistant_only_loss                     = True,   # chỉ tính loss trên phần assistant trả lời
    )

    trainer = SFTTrainer(
        model         = model,
        args          = sft_config,
        train_dataset = dataset,
        processing_class = tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ SFT hoàn tất → {output_dir}")


if __name__ == "__main__":
    main()