"""
precompute_top40_for_hitl.py — Computes and saves the top-40 candidate
DBpedia properties for every unique predicate appearing in the HITL
review data, so hitl_app.py can show the REAL per-triple top-k
candidates instead of a generic fixed property list.

Reuses the exact, verified get_top_k() and load_catalog_and_embeddings()
logic from normalize_full_scale.py -- same model, same catalog, same
similarity computation -- so results are guaranteed consistent with
what the real pipeline actually considered, not a re-derived
approximation.

This is a one-time (or periodic) offline precomputation -- the hosted
HITL app never loads the model or catalog itself, it only reads the
saved JSON this script produces. Avoids loading F2LLM-1.7B inside the
public Streamlit app, which risks hitting resource limits.

Run:
    python3 precompute_top40_for_hitl.py
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

FINETUNED_MODEL = "/home/nsingh/f2lm_finetuned_v2_merged"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"
ALIGNMENT_FILE = "/home/nsingh/alignment_results_full_20k.jsonl"
OUTPUT_FILE = "/home/nsingh/hitl_top40_candidates.json"
TOP_K = 40


def load_catalog_and_embeddings(model):
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_texts = []
    catalog_uris = []
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        catalog_texts.append(text)
        catalog_uris.append(short_uri)
    print(f"  Embedding {len(catalog_texts)} catalog properties...")
    catalog_vecs = model.encode(
        catalog_texts, convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True, batch_size=64,
    ).astype(np.float32)
    return catalog_vecs, catalog_uris


def get_top_k(predicate, model, catalog_vecs, catalog_uris, k=TOP_K):
    q_vec = model.encode(
        [predicate], convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]
    sims = catalog_vecs @ q_vec
    top_idx = np.argsort(-sims)[:k]
    return [(catalog_uris[i], round(float(sims[i]), 4)) for i in top_idx]


def main():
    print("Loading F2LLM-1.7B model...")
    model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)

    catalog_vecs, catalog_uris = load_catalog_and_embeddings(model)

    print(f"\nLoading HITL alignment data from {ALIGNMENT_FILE}...")
    unique_predicates = set()
    with open(ALIGNMENT_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rel = row.get("relation", "").strip()
            if rel and rel.lower() != "property":
                unique_predicates.add(rel)

    print(f"Unique predicates needing top-{TOP_K} candidates: {len(unique_predicates)}")

    results = {}
    for i, predicate in enumerate(sorted(unique_predicates)):
        top_k_pairs = get_top_k(predicate, model, catalog_vecs, catalog_uris, k=TOP_K)
        results[predicate] = [{"uri": uri, "score": score} for uri, score in top_k_pairs]

        if (i + 1) % 100 == 0 or (i + 1) == len(unique_predicates):
            print(f"  [{i+1}/{len(unique_predicates)}] done", flush=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved top-{TOP_K} candidates for {len(results)} predicates to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
