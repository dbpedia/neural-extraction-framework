"""
evaluate_few_shot_similarity.py — Full-scale extraction with SIMILARITY-
RETRIEVED few-shot examples (per-sentence, via F2LLM embeddings), instead
of a fixed random set.

Rationale: AgentRE (CIKM'24) showed randomly-selected few-shot examples
barely beat zero-shot (29.0 -> 30.1 F1 in their ablation), while
embedding-similarity-retrieved examples gave a large jump (-> 43.0 F1).
Today's earlier few-shot attempt used 3 fixed, randomly-chosen examples
and underperformed zero-shot on every source. This script tests whether
per-sentence similarity retrieval (top-5 nearest training examples by
F2LLM embedding) performs better, as the research would predict.

Model loading, generation loop, and stop-token handling are copied
unchanged from evaluate_full_scale.py (already verified this session).
Embedding pattern (SentenceTransformer, normalize_embeddings=True,
batch_size=64) matches generate_hitl_data.py, using the same confirmed
FINETUNED_MODEL path.

Run (inside tmux):
    python3 evaluate_few_shot_similarity.py
"""

import os
import json
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"
FINETUNED_MODEL = "/home/nsingh/f2lm_finetuned_v2_merged"
WIKI_VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
BENCHIE_FILE = "/home/nsingh/benchie_converted.jsonl"

N_TRAIN = 50
K_SHOTS = 5           # matches AgentRE's winning k=5
MAX_NEW_TOKENS = 256
SEED = 42
OUTPUT_FILE = "/home/nsingh/eval_few_shot_similarity_results.json"
SAVE_EVERY = 50

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


def load_all_wikipedia():
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


def load_all_train_sentences():
    """Loads every sentence in the training corpus, for both the eval
    subset AND the retrieval pool (same file, as in evaluate_full_scale.py)."""
    examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            reference = extract_final_answer(entry["messages"][2]["content"])
            examples.append({"sentence": sentence, "reference": reference})
    return examples


def load_all_benchie():
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


def build_retrieval_index(embed_model, train_examples):
    """Embeds every training sentence once. Returns (embeddings, sentences_list)."""
    sentences = [ex["sentence"] for ex in train_examples]
    print(f"Embedding {len(sentences)} training sentences for retrieval index...")
    embeddings = embed_model.encode(
        sentences, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True, batch_size=64,
    )
    return embeddings, train_examples


def retrieve_top_k(embed_model, query_sentence, train_embeddings, train_examples, k):
    """Finds the k most similar training examples by cosine similarity,
    excluding any training sentence that is byte-identical to the query
    (prevents an eval sentence retrieving itself when it's drawn from
    the same underlying training file)."""
    query_vec = embed_model.encode(
        [query_sentence], convert_to_numpy=True, normalize_embeddings=True,
    )[0]
    sims = train_embeddings @ query_vec
    ranked_idx = np.argsort(-sims)

    results = []
    for idx in ranked_idx:
        candidate = train_examples[idx]
        if candidate["sentence"].strip() == query_sentence.strip():
            continue
        results.append(candidate)
        if len(results) >= k:
            break
    return results


def build_prompt(tokenizer, sentence, few_shot_examples):
    examples_text = "\n\n".join(
        f"Sentence: {ex['sentence']}\n{ex['reference']}" for ex in few_shot_examples
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
    eos_token_id OR Gemma's <end_of_turn> token. (Unchanged from
    evaluate_full_scale.py.)"""
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


def run_source(source_name, samples, model, tokenizer, embed_model,
               train_embeddings, train_examples, all_results):
    print(f"\n{'='*70}\nEvaluating source: {source_name} ({len(samples)} samples)\n{'='*70}")

    for idx, ex in enumerate(samples):
        few_shot = retrieve_top_k(
            embed_model, ex["sentence"], train_embeddings, train_examples, K_SHOTS
        )
        prompt = build_prompt(tokenizer, ex["sentence"], few_shot)
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
            "retrieved_shot_sentences": [f["sentence"] for f in few_shot],
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
    print("Loading embedding model for similarity retrieval...")
    embed_model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)

    print("\nLoading all sources...")
    all_train_examples = load_all_train_sentences()
    wiki_samples = load_all_wikipedia()
    benchie_samples = load_all_benchie()

    random.Random(SEED).shuffle(all_train_examples)
    train_eval_samples = all_train_examples[:N_TRAIN]

    print(f"  Wikipedia: {len(wiki_samples)} samples (full validation set)")
    print(f"  Train:     {len(train_eval_samples)} samples (existing subset)")
    print(f"  BenchIE:   {len(benchie_samples)} samples (full set)")
    print(f"  Retrieval pool: {len(all_train_examples)} training sentences, k={K_SHOTS}")

    train_embeddings, retrieval_examples = build_retrieval_index(embed_model, all_train_examples)

    model, tokenizer = load_model()

    all_results = {"wikipedia": [], "train": [], "benchie": []}

    run_source("wikipedia", wiki_samples, model, tokenizer, embed_model,
               train_embeddings, retrieval_examples, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("train", train_eval_samples, model, tokenizer, embed_model,
               train_embeddings, retrieval_examples, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("benchie", benchie_samples, model, tokenizer, embed_model,
               train_embeddings, retrieval_examples, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}\nSUMMARY (similarity-retrieved few-shot, full scale)\n{'='*70}")
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
    print("Compare against random-few-shot: /home/nsingh/eval_few_shot_results.json")


if __name__ == "__main__":
    main()
