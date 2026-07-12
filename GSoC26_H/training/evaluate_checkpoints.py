"""
evaluate_checkpoints.py — Offline generation-based evaluation across all
saved checkpoints from a training run.

For each checkpoint, loads the LoRA adapter on top of the base model and
runs pass@1 / valid_format_rate against a sample of the validation set.
Produces a clean results table so progress across training can actually
be compared — this is what the zeroth-step baseline alone can't show.

Run (inside tmux, since this can take a while across many checkpoints):
    python3 evaluate_checkpoints.py
"""

import os
import re
import json
import glob
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_DIR = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002"
VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
N_SAMPLES = 100
MAX_NEW_TOKENS = 256
SEED = 42


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


def is_valid_slug_format(text):
    text = extract_final_answer(text)
    if text == "NONE":
        return True
    lines = [l for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    for line in lines:
        parts = line.split("|")
        if len(parts) != 3 or not all(p.strip() for p in parts):
            return False
    return True


def get_prompt_and_reference(example, tokenizer):
    messages = example["messages"]
    user_turns = merge_system_into_user(messages)
    reference = extract_final_answer(messages[-1]["content"])
    prompt = tokenizer.apply_chat_template(
        [m for m in user_turns if m["role"] == "user"],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, reference


def find_all_checkpoints(checkpoint_dir):
    """Returns a sorted list of (label, path) tuples: numeric checkpoints
    in ascending step order, followed by 'final' last."""
    paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))

    def step_num(p):
        match = re.search(r"checkpoint-(\d+)$", p)
        return int(match.group(1)) if match else -1

    paths.sort(key=step_num)
    result = [(f"checkpoint-{step_num(p)}", p) for p in paths]

    final_path = os.path.join(checkpoint_dir, "final")
    if os.path.isdir(final_path):
        result.append(("final", final_path))

    return result


def run_generation_eval(model, tokenizer, samples, max_new_tokens=256):
    model.eval()
    correct = 0
    valid_format = 0
    eos_id = tokenizer.eos_token_id

    for idx, example in enumerate(samples):
        print(f"    sample {idx + 1}/{len(samples)}...", end="\r", flush=True)

        prompt, reference = get_prompt_and_reference(example, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if input_ids.dim() > 2:
            input_ids = input_ids.view(input_ids.shape[0], -1)
        if attention_mask.dim() > 2:
            attention_mask = attention_mask.view(attention_mask.shape[0], -1)

        prompt_len = input_ids.shape[1]
        generated = input_ids

        with torch.no_grad():
            past_key_values = None
            current_input = generated
            for _ in range(max_new_tokens):
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

        if is_valid_slug_format(generated_text):
            valid_format += 1
        if predicted.strip() == reference.strip():
            correct += 1

    print()
    n = len(samples)
    return correct / n if n else 0.0, valid_format / n if n else 0.0


def main():
    random.seed(SEED)

    print("Loading tokenizer and base model (once, reused across checkpoints)...")
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

    print(f"Loading {N_SAMPLES} fixed validation samples (same set used for every checkpoint)...")
    all_val_examples = []
    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_val_examples.append(json.loads(line))
    samples = random.sample(all_val_examples, min(N_SAMPLES, len(all_val_examples)))

    checkpoints = find_all_checkpoints(CHECKPOINT_DIR)
    print(f"Found {len(checkpoints)} checkpoints to evaluate: {[c[0] for c in checkpoints]}\n")

    results = []

    for label, path in checkpoints:
        print(f"Evaluating {label} ({path})...")
        model = PeftModel.from_pretrained(base_model, path)
        pass_at_1, valid_format_rate = run_generation_eval(
            model, tokenizer, samples, max_new_tokens=MAX_NEW_TOKENS
        )
        print(f"  -> pass@1={pass_at_1:.3f}  valid_format_rate={valid_format_rate:.3f}\n")
        results.append({
            "checkpoint": label,
            "pass_at_1": pass_at_1,
            "valid_format_rate": valid_format_rate,
        })
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"{'Checkpoint':<20}{'pass@1':<12}{'valid_format_rate':<20}")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['checkpoint']:<20}{r['pass_at_1']:<12.3f}{r['valid_format_rate']:<20.3f}")

    output_path = "/home/nsingh/checkpoint_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
