"""
merge_lora_only.py — Merges the plain-LoRA adapter into the base
F2LLM-v2-1.7B model, producing a standalone model SentenceTransformer
can load directly — matching how f2lm_finetuned_merged and
f2lm_finetuned_v2_merged were prepared for QLoRA.

Run:
    python3 merge_lora_only.py
"""

import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

BASE_MODEL = "codefuse-ai/F2LLM-v2-1.7B"
ADAPTER_PATH = "/home/nsingh/f2lm_finetuned_lora_only/final"
OUTPUT_PATH = "/home/nsingh/f2lm_finetuned_lora_only_merged"

print("Loading base model...")
base_model = AutoModel.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging adapter into base model...")
merged = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
merged.save_pretrained(OUTPUT_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Done.")
