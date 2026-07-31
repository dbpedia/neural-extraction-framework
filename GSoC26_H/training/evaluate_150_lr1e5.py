"""
evaluate_150_lr1e5.py — Same 150-sample set (50 Wikipedia, 50 Train, 50
BenchIE), same random seed, as the lr=2e-4 evaluation — but run against
the lr=1e-5 checkpoint instead. Runs on CPU deliberately, so it doesn't
compete with the GPU-based gold-set jobs currently running.

Uses the corrected <end_of_turn> stopping logic (same fix applied after
the original lr=2e-4 run revealed the repetition-garbage bug).

Run (inside tmux, this will take hours on CPU):
    python3 evaluate_150_lr1e5.py
"""

import os
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr1e-05/final"
WIKI_VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
BENCHIE_FILE = "/home/nsingh/benchie_converted.jsonl"
N_PER_SOURCE = 50
MAX_NEW_TOKENS = 256
SEED = 42  # SAME seed as lr=2e-4 run — ensures identical 150 samples for a fair comparison

OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""


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


def build_prompt(tokenizer, sentence):
    messages = [{"role": "user", "content": f"{OPTIMAL_INSTRUCTION}\n\n{sentence}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Manual greedy decode with KV-cache reuse. Stops on either the base
    eos_token_id OR Gemma's <end_of_turn> token — the confirmed fix from
    the lr=2e-4 evaluation."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    generated = input_ids

    eos_id = tokenizer.eos_token_id
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = {eos_id}
    if end_of_turn_id is not None and end_of_turn_id >= 0:
        stop_ids.add(end_of_turn_id)

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
            if next_token.item() in stop_ids:
                break

    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)


def load_wikipedia_samples(n, seed):
    seen_sentences = set()
    examples = []
    with open(WIKI_VAL_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            reference = extract_final_answer(entry["messages"][2]["content"])
            examples.append({"sentence": sentence, "reference": reference})

    random.Random(seed).shuffle(examples)
    return examples[:n]


def load_train_samples(n, seed):
    examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            reference = extract_final_answer(entry["messages"][2]["content"])
            examples.append({"sentence": sentence, "reference": reference})

    random.Random(seed).shuffle(examples)
    return examples[:n]


def load_benchie_samples(n, seed):
    if not os.path.exists(BENCHIE_FILE):
        print(f"  WARNING: BenchIE file not found at {BENCHIE_FILE}")
        return []

    examples = []
    with open(BENCHIE_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            examples.append({
                "sentence": entry["sentence"],
                "reference": entry["reference"],
            })

    random.Random(seed).shuffle(examples)
    return examples[:n]


def main():
    print("Loading tokenizer and base model on CPU (avoids GPU conflict "
          "with active gold-set jobs)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    print("\nLoading the SAME 150 samples used for lr=2e-4 (identical seed)...")
    wiki_samples = load_wikipedia_samples(N_PER_SOURCE, SEED)
    train_samples = load_train_samples(N_PER_SOURCE, SEED)
    benchie_samples = load_benchie_samples(N_PER_SOURCE, SEED)

    print(f"  Wikipedia: {len(wiki_samples)}  Train: {len(train_samples)}  BenchIE: {len(benchie_samples)}")

    all_results = {"wikipedia": [], "train": [], "benchie": []}

    for source_name, samples in [
        ("wikipedia", wiki_samples),
        ("train", train_samples),
        ("benchie", benchie_samples),
    ]:
        print(f"\n{'='*70}\nEvaluating source: {source_name} ({len(samples)} samples)\n{'='*70}")

        for idx, ex in enumerate(samples):
            print(f"  [{source_name}] sample {idx+1}/{len(samples)}...", flush=True)
            prompt = build_prompt(tokenizer, ex["sentence"])
            generated_text = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
            predicted = extract_final_answer(generated_text)
            valid_format = is_valid_slug_format(generated_text)
            exact_match = (predicted.strip() == ex["reference"].strip())

            all_results[source_name].append({
                "sentence": ex["sentence"],
                "reference": ex["reference"],
                "predicted": predicted,
                "valid_format": valid_format,
                "exact_match": exact_match,
            })

    print(f"\n{'='*70}\nSUMMARY — lr=1e-5\n{'='*70}")

    for source_name in ("wikipedia", "train", "benchie"):
        results = all_results[source_name]
        if not results:
            continue
        n = len(results)
        valid_rate = sum(r["valid_format"] for r in results) / n
        pass_at_1 = sum(r["exact_match"] for r in results) / n

        print(f"\n{source_name.upper()} (n={n})")
        print(f"  pass@1 (exact match): {pass_at_1:.3f}")
        print(f"  valid_format_rate:    {valid_rate:.3f}")

    output_path = "/home/nsingh/eval_150_lr1e5_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
