"""
evaluate_few_shot.py — Same extraction model, generation logic, and
FULL scale as evaluate_full_scale.py (ALL 1,817 Wikipedia validation
sentences + ALL BenchIE + 50 Train), but with real few-shot examples
pulled directly from exp1_train_optimal_only.jsonl prepended to the
prompt, per Debarghya's request to check whether few-shot examples
improve extraction quality.

Few-shot examples use the REAL multi-triple format (multiple relations
per sentence, including "property"-labeled lines), matching exactly
what the model was fine-tuned to produce — not a simplified version.

The exact sentences used as few-shot examples are excluded from the
Train eval pool below, to avoid the model being tested on something
it just saw in-context.

Run (inside tmux):
    python3 evaluate_few_shot.py
"""

import os
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"
WIKI_VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
BENCHIE_FILE = "/home/nsingh/benchie_converted.jsonl"

N_TRAIN = 50
MAX_NEW_TOKENS = 256
SEED = 42
OUTPUT_FILE = "/home/nsingh/eval_few_shot_results.json"
SAVE_EVERY = 50

OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""

# Real examples pulled directly from exp1_train_optimal_only.jsonl (seed=7).
# Kept in the model's actual multi-triple, property-labeled output format.
FEW_SHOT_EXAMPLES = [
    {
        "sentence": "नदी का उदगम स्थान है हिमालय की गलीश में, जहाँ प्रत्येक साल जलवायु परिवर्तन के प्रभावों की जांच हो रही है।",
        "answer": """नदी | उदगम स्थान है | हिमालय की गलीश में
जांच | हो रही | है
नदी | property | उदगम स्थान
हिमालय | property | गलीश
जलवायु परिवर्तन | property | प्रभावों
प्रभावों | property | जांच
प्रत्येक साल | property | जांच""",
    },
    {
        "sentence": "विकिरणीय सुरक्षा मानकों के अनुसार, चंद्रमा की सतह पर स्थापित लेजर प्रयोगशाला को पूर्णतया कैलिब्रेटेड होना चाहिए, और इसकी माप प्रणाली सिद्ध होती है प्रयोगात्मक परिणाम में।",
        "answer": """लेजर प्रयोगशाला | होना चाहिए | पूर्णतया कैलिब्रेटेड
इसकी माप प्रणाली | सिद्ध होती है | प्रयोगात्मक परिणाम में
विकिरणीय | property | सुरक्षा मानकों
सुरक्षा | property | मानकों
के अनुसार | property | विकिरणीय सुरक्षा मानकों
चंद्रमा | property | सतह
सतह | property | स्थापित लेजर प्रयोगशाला
स्थापित | property | लेजर प्रयोगशाला
लेजर | property | प्रयोगशाला
माप | property | प्रणाली
प्रयोगात्मक | property | परिणाम""",
    },
    {
        "sentence": "(Yahoo!)\nखेल: लौरियस वर्ल्ड स्पोर्ट्स अवार्ड (Laureus World Sports Awards)\nप्रौद्योगिकी हार्डवेयर और यंत्र: सिस्को सिस्टम्स (Cisco Systems), कौर्निंग इंक.",
        "answer": "NONE",
    },
]

# Exact sentences used above — excluded from the Train eval pool so the
# model is never tested on something it just saw in the prompt.
FEW_SHOT_SENTENCES = {ex["sentence"] for ex in FEW_SHOT_EXAMPLES}


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
    examples_text = "\n\n".join(
        f"Sentence: {ex['sentence']}\n{ex['answer']}" for ex in FEW_SHOT_EXAMPLES
    )
    content = (
        f"{OPTIMAL_INSTRUCTION}\n\n"
        f"Examples:\n\n{examples_text}\n\n"
        f"Now extract for this sentence:\n"
        f"Sentence: {sentence}"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Manual greedy decode with KV-cache reuse. Stops on either the base
    eos_token_id OR Gemma's <end_of_turn> token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def load_all_wikipedia():
    """Load ALL Wikipedia validation sentences, deduplicated. No cap, no shuffle."""
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
    return examples


def load_train_samples(n, seed):
    """Same as evaluate_full_scale.py, but skips any sentence used as a
    few-shot example above, to prevent leakage."""
    examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            if sentence in FEW_SHOT_SENTENCES:
                continue
            reference = extract_final_answer(entry["messages"][2]["content"])
            examples.append({"sentence": sentence, "reference": reference})
    random.Random(seed).shuffle(examples)
    return examples[:n]


def load_all_benchie():
    """Load ALL BenchIE sentences. No cap, no shuffle."""
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
    return examples


def load_model():
    print("Loading tokenizer and base model on GPU (4-bit)...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()
    return model, tokenizer


def run_source(source_name, samples, model, tokenizer, all_results):
    print(f"\n{'='*70}\nEvaluating source: {source_name} ({len(samples)} samples)\n{'='*70}")

    for idx, ex in enumerate(samples):
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

        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            n_done = idx + 1
            running_pass1 = sum(r["exact_match"] for r in all_results[source_name]) / n_done
            print(f"  [{source_name}] {n_done}/{len(samples)} "
                  f"running pass@1={running_pass1:.3f}", flush=True)

        if (idx + 1) % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    model, tokenizer = load_model()

    print("\nLoading samples from all 3 sources...")
    wiki_samples = load_all_wikipedia()
    train_samples = load_train_samples(N_TRAIN, SEED)
    benchie_samples = load_all_benchie()

    print(f"  Wikipedia: {len(wiki_samples)} samples (full validation set)")
    print(f"  Train:     {len(train_samples)} samples (existing subset, few-shot examples excluded)")
    print(f"  BenchIE:   {len(benchie_samples)} samples (full set)")

    all_results = {"wikipedia": [], "train": [], "benchie": []}

    run_source("wikipedia", wiki_samples, model, tokenizer, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("train", train_samples, model, tokenizer, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("benchie", benchie_samples, model, tokenizer, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}\nSUMMARY (few-shot, full scale)\n{'='*70}")
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

    print(f"\nFull results saved to: {OUTPUT_FILE}")
    print("Compare against zero-shot: /home/nsingh/eval_full_scale_results.json")


if __name__ == "__main__":
    main()
