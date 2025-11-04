from typing import List, Optional, Dict, Any, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch import cuda
from peft import PeftModel
class HuggingFaceLLM:

    def __init__(
        self,
        model_name_or_path: str,
        device_map: Union[str, Dict[str, int]] = "auto",
        max_new_tokens: int = 150,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,  # e.g., "flash_attention_2"
        # bitsandbytes config
        load_in_4bit: bool = True,       # default to 4-bit quantization
        load_in_8bit: bool = False,      # if True, overrides 4-bit
        bnb_compute_dtype: str = "bfloat16",  # "float16"|"bfloat16"|"float32"
        bnb_quant_type: str = "nf4",     # "nf4"|"fp4"
        bnb_use_double_quant: bool = True,
        finetuned: bool = False,
        **generate_kwargs,
        
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.extra_gen = generate_kwargs
        device_map  = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

        adapters = "/mnt/lagarza3/llm-healthcare/source_code/train/sft_ft_llama3_weights" #Path to model fine tuned adapters

        # ----- Tokenizer -----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            token = "input your hugging face token here"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ----- Quantization Config -----
        compute_dtype = getattr(torch, bnb_compute_dtype)
        quant_config = BitsAndBytesConfig(
            
            #load_in_8bit=load_in_8bit,
            load_in_4bit=(not load_in_8bit) and load_in_4bit,
            #bnb_4bit_compute_dtype=compute_dtype,
            #bnb_4bit_use_double_quant=bnb_use_double_quant,
            bnb_4bit_quant_type=bnb_quant_type,
        )

        print("model", model_name_or_path)
        # ----- Load model -----
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            quantization_config=quant_config,
            token = "input your hugging face token here"
        )
        if finetuned:
            print("using fine tune model", adapters)
            self.model = PeftModel.from_pretrained(self.model, adapters)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.eos_token_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id

    # -----------------------------
    # Inference
    # -----------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        full = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        gen_out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k if self.top_k > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_id,
            **self.extra_gen,
        )
        gen_ids = gen_out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if stop:
            text = self._apply_stop(text, stop)
        return text.strip()

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        batch_size: int = 4,
    ) -> List[str]:
        outs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            if system_prompt:
                batch = [f"{system_prompt}\n\n{p}" for p in batch]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            with torch.no_grad():
                gen_out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k if self.top_k > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    **self.extra_gen,
                )
            B = inputs["input_ids"].shape[0]
            start = inputs["input_ids"].shape[1]
            for b in range(B):
                gen_ids = gen_out[b][start:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if stop:
                    text = self._apply_stop(text, stop)
                outs.append(text.strip())
        return outs

    @staticmethod
    def _apply_stop(text: str, stop: List[str]) -> str:
        cut = len(text)
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]
