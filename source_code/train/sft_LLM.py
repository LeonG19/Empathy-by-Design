

import os, json, argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------
# Domain prompt (style control)
# ------------------------------
SYSTEM_PROMPT = (
    "You are a careful, evidence-based clinical assistant specialized in healthcare, "
    "geriatrics, dementia, and Alzheimer's disease. Start your answer with a brief, "
    "compassionate one-line sentence acknowledging the user's concern, then provide a "
    "clear, simple, and friendly explanation. Do not invent facts. Be polite and kind."
)
USER_TEMPLATE = "Question: {q}\nAnswer:"

def build_prompt(q: str) -> str:
    q = (q or "").strip()
    return f"{SYSTEM_PROMPT}\n\n" + USER_TEMPLATE.format(q=q)

# ------------------------------
# Argument parser
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="SFT with QLoRA — save adapters only")
    ap.add_argument("--model", required=True, help="Base HF model, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data", required=True, help="JSONL with fields: prompt, chosen[, rejected]")
    ap.add_argument("--out",  required=True, help="Output directory for LoRA adapters")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--hf_token", type=str, default=None, help="Optional HF token")
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    # bumped defaults for capacity
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    return ap.parse_args()

# ------------------------------
# Main training entry
# ------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------ Load dataset ------------------
    ds = load_dataset("json", data_files={"train": args.data})["train"]

    # ------------------ Tokenizer ------------------
    tok = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
        token=args.hf_token,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    # ------------------ Preprocess ------------------
    def to_sft(example):
        prompt_text = build_prompt(example["prompt"])
        answer_text = (example.get("chosen") or "").strip()

        # Encode parts separately
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(answer_text, add_special_tokens=False)["input_ids"]

        # Ensure EOS at the end of each answer
        if eos_id is not None and (len(answer_ids) == 0 or answer_ids[-1] != eos_id):
            answer_ids += [eos_id]

        # Concatenate prompt + answer; mask prompt tokens with -100
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        # Truncate from left if needed
        if len(input_ids) > args.max_len:
            overflow = len(input_ids) - args.max_len
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    ds = ds.map(
        to_sft,
        remove_columns=[c for c in ds.column_names if c not in {"input_ids", "labels", "attention_mask"}]
    )

    # ------------------ Quantization Config (QLoRA) ------------------
    quant_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # ------------------ Model + LoRA ------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        token=args.hf_token,
    )
    model.config.pad_token_id = pad_id

    model = prepare_model_for_kbit_training(model)

    # Keep targets on attention only (q/k/v/o); bumped r/alpha
    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lconf)

    # ------------------ Collator ------------------
    def collate(batch):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
        attn = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    # ------------------ TrainingArgs ------------------
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        #warmup_ratio=args.warmup_ratio,
        #weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
    )

    # ------------------ Trainer ------------------
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        eval_dataset=None,
        data_collator=collate,
        tokenizer=tok,
    )

    trainer.train()

    # ------------------ SAVE: Adapters only (no merge) ------------------
    adapters_dir = args.out
    os.makedirs(adapters_dir, exist_ok=True)
    trainer.model.save_pretrained(adapters_dir)  # adapter_model.bin & adapter_config.json
    tok.save_pretrained(adapters_dir)

    # Helpful metadata for deployment
    meta = {
        "base_model_name_or_path": args.model,
        "quantization_note": "Trained with 4-bit QLoRA. To MERGE at deploy, load base in (b)float16/no-quant.",
        "prompt": {"system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE},
    }
    with open(os.path.join(adapters_dir, "ADAPTER_METADATA.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SFT] ✅ Done. Adapters saved to: {adapters_dir}")
    print("[SFT] Load base + adapters at inference; optionally merge at startup.")

if __name__ == "__main__":
    main()
