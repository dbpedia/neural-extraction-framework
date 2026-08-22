"""
peek_predictions.py — Quick standalone check: model prediction vs ground truth
for a couple of validation examples, using the already-saved smoke test checkpoint.

Run:
    python3 peek_predictions.py
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
ADAPTER_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"
VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
N_EXAMPLES = 2
MAX_NEW_TOKENS = 256


def merge_system_into_user(messages):
    merged = []
    pending_system = None
    for m in messages:
        if m["role"] == "system":
            pending_system = m["content"]
        elif m["role"] == "user":
            content = m["content"]
            if pending_system:
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            merged.append({"role": "user", "content": content})
        else:
            merged.append(m)
    return merged


def extract_final_answer(text):
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]")[-1]
    return text.strip()


def main():
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print(f"Loading {N_EXAMPLES} examples from validation set...")
    examples = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= N_EXAMPLES:
                break

    eos_id = tokenizer.eos_token_id

    for idx, example in enumerate(examples):
        messages = example["messages"]
        user_turns = merge_system_into_user(messages)
        reference = extract_final_answer(messages[-1]["content"])
        sentence = messages[1]["content"]  # original user turn (pre-merge) for display

        prompt = tokenizer.apply_chat_template(
            [m for m in user_turns if m["role"] == "user"],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = input_ids.shape[1]
        generated = input_ids

        with torch.no_grad():
            past_key_values = None
            current_input = generated
            for _ in range(MAX_NEW_TOKENS):
                outputs = model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                past_key_values = outputs.past_key_values
                current_input = next_token
                if next_token.item() == eos_id:
                    break

        generated_text = tokenizer.decode(
            generated[0][prompt_len:], skip_special_tokens=True
        )
        predicted = extract_final_answer(generated_text)

        print(f"\n{'='*60}")
        print(f"Example {idx + 1}")
        print(f"{'='*60}")
        print(f"Sentence:\n{sentence}\n")
        print(f"Model predicted:\n{predicted}\n")
        print(f"Correct answer:\n{reference}")

    print(f"\n{'='*60}\nDone.")


if __name__ == "__main__":
    main()
