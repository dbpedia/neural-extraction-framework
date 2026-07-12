"""
evaluate_150_mixed.py — Offline CPU evaluation across 3 sources: 50 from
cleanest Wikipedia validation set, 50 from Hindi BenchIE gold data
(pre-converted to slug format), 50 random from the training set.
Evaluates the lr=2e-4 'final' checkpoint.

Runs on CPU deliberately, so it can run safely alongside an active GPU
training job (e.g. the lr=1e-5 run) with zero memory conflict.

NOTE on BenchIE scoring: BenchIE's gold data (already converted from its
original multi-cluster format to a single-cluster slug format via
convert_benchie.py) is real gold-standard data, but was authored under
somewhat different annotation conventions than our own training data.
We report both pass@1 (exact match) and valid_format_rate, but treat
BenchIE's pass@1 as a rougher signal than Wikipedia/Train's — worth
reviewing the side-by-side output, not just the number.

Run (inside tmux, this will take a while on CPU):
    python3 evaluate_150_mixed.py
"""

import os
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"
WIKI_VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
BENCHIE_FILE = "/home/nsingh/benchie_converted.jsonl"
N_PER_SOURCE = 50
MAX_NEW_TOKENS = 256
SEED = 42

OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""


# ── Shared helpers (same logic as train.py) ──────────────────

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
    """Manual greedy decode with KV-cache reuse — same crash-proof
    approach as train.py, since model.generate() has a confirmed
    open bug for Gemma3 (huggingface/transformers#36815)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    generated = input_ids
    eos_id = tokenizer.eos_token_id

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

    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)


# ── Source loaders ─────────────────────────────────────────────

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


# ── Main ─────────────────────────────────────────────────────

def main():
    print("Loading tokenizer and base model on CPU (this avoids any GPU "
          "conflict with active training)...")
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

    print("\nLoading samples from all 3 sources...")
    wiki_samples = load_wikipedia_samples(N_PER_SOURCE, SEED)
    train_samples = load_train_samples(N_PER_SOURCE, SEED)
    benchie_samples = load_benchie_samples(N_PER_SOURCE, SEED)

    print(f"  Wikipedia: {len(wiki_samples)} samples")
    print(f"  Train:     {len(train_samples)} samples")
    print(f"  BenchIE:   {len(benchie_samples)} samples")

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

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")

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

    print(f"\nNote: BenchIE pass@1 is a rougher signal than Wikipedia/Train's, "
          f"since BenchIE was authored under somewhat different annotation "
          f"conventions. Review the side-by-side output below, not just the number.")

    # ── Save full results ────────────────────────────────────
    output_path = "/home/nsingh/eval_150_mixed_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results (including all sentences/gold/predictions) saved to:")
    print(f"  {output_path}")

    # ── Print BenchIE side-by-side for immediate manual review ───
    print(f"\n{'='*70}\nBenchIE — sentence / gold / prediction (for manual review)\n{'='*70}")
    for i, r in enumerate(all_results["benchie"][:10]):
        print(f"\n--- BenchIE sample {i+1} ---")
        print(f"Sentence: {r['sentence']}")
        print(f"Gold:\n{r['reference']}")
        print(f"Predicted:\n{r['predicted']}")
        print(f"Exact match: {r['exact_match']}  |  Valid format: {r['valid_format']}")


if __name__ == "__main__":
    main()
