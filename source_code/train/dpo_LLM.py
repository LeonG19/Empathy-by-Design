#!/usr/bin/env python3
import os, json, argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments, BitsAndBytesConfig,
)

# --- TRL imports (with DPOConfig) ---
from trl import DPOTrainer
try:
    from trl import DPOConfig
except Exception:
    DPOConfig = None  # fallback handled below

# ------------------------------
# Domain prompt: clear & concise
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
# Args
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Simple DPO fine-tuner (QLoRA 4-bit, merged output)")
    ap.add_argument("--model", required=True, help="Base HF model, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data", required=True, help="JSONL with fields: prompt, chosen, rejected")
    ap.add_argument("--out",  required=True, help="Output dir for merged model")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr",     type=float, default=2e-4)
    ap.add_argument("--beta",   type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len",    type=int, default=2048)
    ap.add_argument("--hf_token",   type=str, default=None, help="Optional HF token")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ------------------ Load dataset ------------------
    ds = load_dataset("json", data_files={"train": args.data})["train"]
    def _map(ex):
        return {
            "prompt":  build_prompt(ex["prompt"]),
            "chosen":  str(ex["chosen"]),
            "rejected":str(ex["rejected"]),
        }
    ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c not in {"prompt","chosen","rejected"}])

    # ------------------ Tokenizer ------------------
    tok = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, trust_remote_code=True,
        token = "input hugging face token"

    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.eos_token_id
        

    # ------------------ QLoRA (bitsandbytes 4-bit) ------------------
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=False,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    

    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        token = "input hugging face token",
    )
    policy.config.pad_token_id = tok.pad_token_id


    # --- LoRA (good defaults for LLaMA/Mistral) ---
    from peft import LoraConfig, get_peft_model
    lconf = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    policy = get_peft_model(policy, lconf)

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True,
    quantization_config=quant_config, token="input hugging face token"
    )
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.eval()

    # Use DPOConfig to avoid deprecation + future errors
    dpo_cfg = DPOConfig(
            beta=args.beta,
            learning_rate=args.lr,
            max_length=args.max_len,
            
            padding_value = tok.pad_token_id,
            output_dir=os.path.join(args.out, "checkpoints"),
            model_init_kwargs=None,   # <-- explicitly None so TRL won't complain
        )

    from transformers import PreTrainedTokenizerBase, ProcessorMixin
    class TokenizerProcessor(ProcessorMixin):
        def __init__(self, tokenizer: PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        def __call__(self, text=None, **kwargs):
            return self.tokenizer(text, **kwargs)

    processor = TokenizerProcessor(tok)

    trainer = DPOTrainer(
            model=policy,        # already-instantiated model is OK now
            ref_model=ref_model,
            args=dpo_cfg,
            train_dataset=ds,
            eval_dataset=None,
            #processing_class = processor
            
            

            
            
        )

    trainer.train()

    adapters_dir = os.path.join(args.out, "lora_adapters")
    os.makedirs(adapters_dir, exist_ok=True)

    # With a PEFT-wrapped model, save_pretrained() writes ONLY the adapter weights/config.
    trainer.model.save_pretrained(adapters_dir)
    tok.save_pretrained(adapters_dir)

    with open(os.path.join(adapters_dir, "DPO_PROMPT.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"system_prompt": SYSTEM_PROMPT, "user_template": USER_TEMPLATE},
            f, indent=2
        )

    print(f"[DPO] Done. LoRA adapters saved to: {adapters_dir}")
    print("[DPO] To load for inference:")
    print("    from transformers import AutoTokenizer, AutoModelForCausalLM")
    print("    from peft import PeftModel")
    print(f"    tok = AutoTokenizer.from_pretrained('{args.model}')")
    print(f"    base = AutoModelForCausalLM.from_pretrained('{args.model}', device_map='auto')")
    print(f"    model = PeftModel.from_pretrained(base, '{adapters_dir}')")
    print("    model.eval()")

if __name__ == "__main__":
    main()