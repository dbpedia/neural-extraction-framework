"""
evaluate_finetuned_fair.py — Fair comparison of original F2LLM-1.7B vs
fine-tuned v2 F2LLM-1.7B on the same 8B gold set, using identical
SentenceTransformer encoding for both models.

Steps:
1. Merge v2 LoRA adapter into base model weights
2. Save merged model
3. Load both original and merged model via SentenceTransformer
4. Evaluate both on 8B gold set (5855 entries)
5. Report side-by-side comparison

Run:
    python3 evaluate_finetuned_fair.py
"""

import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sentence_transformers import SentenceTransformer

ORIGINAL_MODEL = "codefuse-ai/F2LLM-v2-1.7B"
ADAPTER_PATH = "/home/nsingh/f2lm_finetuned_v2/final"
MERGED_MODEL_PATH = "/home/nsingh/f2lm_finetuned_v2_merged"
GOLD_SET_FILE = "/home/nsingh/gold_set_8b_final.jsonl"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"
RESULTS_FILE = "/home/nsingh/eval_finetuned_fair_results_v2.json"


def load_gold_set():
    gold = []
    with open(GOLD_SET_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["gold_dbo"] != "NONE":
                gold.append(entry)
    return gold


def load_catalog():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    texts = []
    uris = []
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        texts.append(text)
        uris.append(short_uri)
    return texts, uris


def evaluate_model(model, gold_entries, catalog_texts, catalog_uris,
                   top_k_check=(1, 5, 10)):
    predicates = [e["predicate"] for e in gold_entries]
    gold_uris = [e["gold_dbo"] for e in gold_entries]

    print(f"  Encoding {len(predicates)} predicates...")
    query_vecs = model.encode(
        predicates, convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True, batch_size=64,
    ).astype(np.float32)

    print(f"  Encoding {len(catalog_texts)} catalog properties...")
    catalog_vecs = model.encode(
        catalog_texts, convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True, batch_size=64,
    ).astype(np.float32)

    hits = {k: 0 for k in top_k_check}
    max_k = max(top_k_check)

    for i, q_vec in enumerate(query_vecs):
        sims = catalog_vecs @ q_vec
        top_idx = np.argsort(-sims)[:max_k]
        top_uris = [catalog_uris[idx] for idx in top_idx]
        for k in top_k_check:
            if gold_uris[i] in top_uris[:k]:
                hits[k] += 1

    n = len(gold_entries)
    return {k: hits[k] / n for k in top_k_check}


def merge_and_save():
    if os.path.exists(MERGED_MODEL_PATH) and os.path.exists(
            os.path.join(MERGED_MODEL_PATH, "config.json")):
        print(f"Merged model already exists at {MERGED_MODEL_PATH} — skipping merge.")
        return

    print(f"Loading base model {ORIGINAL_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, trust_remote_code=True)
    base = AutoModel.from_pretrained(
        ORIGINAL_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading and merging LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_MODEL_PATH}...")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print("Merge complete.")


def main():
    print("=" * 60)
    print("Fair Evaluation: Original vs Fine-tuned v2 F2LLM-1.7B")
    print("Same SentenceTransformer encoding, same 8B gold set")
    print("=" * 60)

    print("\nLoading gold set...")
    gold_entries = load_gold_set()
    print(f"Gold entries: {len(gold_entries)}")

    print("\nLoading catalog...")
    catalog_texts, catalog_uris = load_catalog()
    print(f"Catalog size: {len(catalog_uris)}")

    print("\nStep 1: Merging v2 LoRA adapter into base model...")
    merge_and_save()

    print("\nStep 2: Evaluating ORIGINAL F2LLM-1.7B...")
    original_model = SentenceTransformer(ORIGINAL_MODEL, trust_remote_code=True)
    original_results = evaluate_model(
        original_model, gold_entries, catalog_texts, catalog_uris
    )
    del original_model
    torch.cuda.empty_cache()

    print("\nStep 3: Evaluating FINE-TUNED v2 F2LLM-1.7B...")
    finetuned_model = SentenceTransformer(MERGED_MODEL_PATH, trust_remote_code=True)
    finetuned_results = evaluate_model(
        finetuned_model, gold_entries, catalog_texts, catalog_uris
    )
    del finetuned_model
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("FAIR COMPARISON — 8B gold set (5,855 predicates)")
    print("Both models: SentenceTransformer wrapper, identical encoding")
    print(f"{'='*60}")
    print(f"{'Metric':<15}{'Original':<15}{'Fine-tuned v2':<15}{'Change':<10}")
    print("-" * 55)
    for k in (1, 5, 10):
        orig = original_results[k]
        fine = finetuned_results[k]
        change = fine - orig
        direction = "+" if change >= 0 else ""
        print(f"precision@{k:<5}{orig:<15.3f}{fine:<15.3f}{direction}{change:.3f}")

    results = {
        "original": {str(k): v for k, v in original_results.items()},
        "finetuned_v2": {str(k): v for k, v in finetuned_results.items()},
        "gold_set_size": len(gold_entries),
        "gold_set": "8B-built (unbiased)",
        "encoding": "SentenceTransformer wrapper (identical for both)",
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
