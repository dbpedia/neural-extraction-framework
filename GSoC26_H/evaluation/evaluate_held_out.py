"""
evaluate_held_out.py — Evaluate ALL F2LLM checkpoints (QLoRA rounds AND
plain LoRA) on the TRUE held-out test set (585 entries, never touched
during any training run), using identical SentenceTransformer encoding
for every single one.

This is the ONLY trustworthy comparison across checkpoints — any script
that trains AND evaluates in one file uses AutoModel + manual pooling
internally, which gives incompatible numbers. Always re-evaluate every
new checkpoint through THIS script before comparing it to anything else.

Plain LoRA checkpoint must be MERGED first (via merge_lora_only.py) —
SentenceTransformer cannot properly load an adapter-only checkpoint.

Run:
    python3 evaluate_held_out.py
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HELD_OUT_FILE = "/home/nsingh/f2lm_finetune_test.jsonl"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"
RESULTS_FILE = "/home/nsingh/eval_held_out_results.json"

TOP_K_CHECK = (1, 5, 10, 15, 20, 25, 30, 35, 40)

CHECKPOINTS = [
    ("Original (0 epochs)",    "codefuse-ai/F2LLM-v2-1.7B"),
    ("Round 1 QLoRA (3 ep)",   "/home/nsingh/f2lm_finetuned_merged"),
    ("Round 2 QLoRA (9 ep)",   "/home/nsingh/f2lm_finetuned_v2_merged"),
    ("Plain LoRA (9 ep)",      "/home/nsingh/f2lm_finetuned_lora_only_merged"),
]


def load_held_out():
    entries = []
    with open(HELD_OUT_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["gold_dbo"] != "NONE":
                entries.append(entry)
    return entries


def load_catalog():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    texts, uris = [], []
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        texts.append(text)
        uris.append(short_uri)
    return texts, uris


def evaluate(model, gold_entries, catalog_texts, catalog_uris,
             top_k_check=TOP_K_CHECK):
    predicates = [e["predicate"] for e in gold_entries]
    gold_uris = [e["gold_dbo"] for e in gold_entries]

    print(f"    Encoding {len(predicates)} held-out predicates...")
    query_vecs = model.encode(
        predicates, convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True, batch_size=64,
    ).astype(np.float32)

    print(f"    Encoding {len(catalog_texts)} catalog properties...")
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


def find_first_k_above_90(results):
    for k in TOP_K_CHECK:
        if results[k] >= 0.90:
            return k
    return None


def main():
    print("=" * 65)
    print("F2LLM Checkpoint Comparison — TRUE Held-Out Evaluation")
    print("Encoding: SentenceTransformer (identical for ALL checkpoints)")
    print("Data: f2lm_finetune_test.jsonl (585 held-out, never trained on)")
    print(f"Checking precision@k for k = {TOP_K_CHECK}")
    print("=" * 65)

    print("\nLoading held-out set...")
    gold_entries = load_held_out()
    print(f"Held-out entries (excl. NONE): {len(gold_entries)}")

    print("\nLoading catalog...")
    catalog_texts, catalog_uris = load_catalog()
    print(f"Catalog size: {len(catalog_uris)}")

    all_results = {}

    for label, model_path in CHECKPOINTS:
        print(f"\n{'─'*65}")
        print(f"Evaluating: {label}")
        print(f"Path: {model_path}")
        print(f"{'─'*65}")

        model = SentenceTransformer(model_path, trust_remote_code=True)
        results = evaluate(model, gold_entries, catalog_texts, catalog_uris)
        all_results[label] = results

        for k in TOP_K_CHECK:
            print(f"  precision@{k}: {results[k]:.3f}")

        first_90 = find_first_k_above_90(results)
        if first_90:
            print(f"  → Crosses 90% precision at k={first_90}")
        else:
            print(f"  → Does not reach 90% precision within k≤{max(TOP_K_CHECK)}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*65}")
    print("COMPARISON TABLE — TRUE held-out set, identical encoding")
    print(f"{'='*65}")
    header = "Checkpoint".ljust(25) + "".join(f"p@{k}".ljust(9) for k in TOP_K_CHECK)
    print(header)
    print("-" * len(header))
    for label, _ in CHECKPOINTS:
        r = all_results[label]
        row = label.ljust(25) + "".join(f"{r[k]:.3f}".ljust(9) for k in TOP_K_CHECK)
        print(row)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
