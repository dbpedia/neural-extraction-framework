"""
generate_hitl_data.py — Builds the per-triple JSONL file the HITL
Streamlit app expects (alignment_results_full_20k.jsonl), using data
already computed: extracted triples from eval_full_scale_results.json
+ predicate→DBO mappings from normalization_cache_k40.json.

No new API calls — cache already has every predicate's DBO decision.
Only computes a confidence score (cosine similarity) fresh, since that
wasn't saved anywhere yet.

Output fields per line: sentence, subject, relation, object, dbo_uri, score
  - relation = original Hindi predicate (not translated)
  - dbo_uri  = matched DBpedia property, or None if NONE/property/unparseable
  - score    = cosine similarity between predicate and matched DBO label

Run:
    python3 generate_hitl_data.py
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

EXTRACTION_FILE = "/home/nsingh/eval_full_scale_results.json"
CACHE_FILE = "/home/nsingh/normalization_cache_k40.json"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"
FINETUNED_MODEL = "/home/nsingh/f2lm_finetuned_v2_merged"

OUTPUT_FILE = "/home/nsingh/alignment_results_full_20k.jsonl"


def parse_triples(text):
    triples = []
    if not text or text.strip() == "NONE":
        return triples
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line == "NONE":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(tuple(parts))
    return triples


def load_catalog_texts():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    uri_to_text = {}
    for entry in catalog:
        short_uri = "dbo:" + entry["property_uri"].split("/")[-1]
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        text = " | ".join(labels[:3]) if labels else short_uri
        uri_to_text[short_uri] = text
    return uri_to_text


def main():
    print("Loading extraction results...")
    with open(EXTRACTION_FILE, encoding="utf-8") as f:
        extraction_data = json.load(f)

    print("Loading normalization cache (predicate -> DBO, no API calls needed)...")
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"  Cache has {len(cache)} predicate mappings")

    print("Loading F2LLM model (for confidence scoring only)...")
    model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)
    uri_to_text = load_catalog_texts()

    rows = []
    unique_predicates_needing_score = set()

    for source in ("wikipedia", "train", "benchie"):
        for r in extraction_data.get(source, []):
            sentence = r["sentence"]
            pred_triples = parse_triples(r["predicted"])
            for subject, relation, obj in pred_triples:
                if relation.strip().lower() == "property":
                    continue
                dbo_uri = cache.get(relation)
                if dbo_uri in (None, "NONE", "UNPARSEABLE"):
                    continue
                rows.append({
                    "sentence": sentence,
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                    "dbo_uri": dbo_uri,
                    "source": source,
                })
                unique_predicates_needing_score.add((relation, dbo_uri))

    print(f"\nTotal triples with a real DBO mapping: {len(rows)}")
    print(f"Unique (predicate, dbo) pairs needing a score: {len(unique_predicates_needing_score)}")

    print("\nComputing confidence scores (cosine similarity)...")
    pairs = list(unique_predicates_needing_score)
    predicates = [p[0] for p in pairs]
    dbo_texts = [uri_to_text.get(p[1], p[1]) for p in pairs]

    pred_vecs = model.encode(predicates, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    dbo_vecs = model.encode(dbo_texts, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=True, batch_size=64)

    score_lookup = {}
    for i, (relation, dbo_uri) in enumerate(pairs):
        score = float(np.dot(pred_vecs[i], dbo_vecs[i]))
        score_lookup[(relation, dbo_uri)] = round(score, 4)

    for row in rows:
        row["score"] = score_lookup[(row["relation"], row["dbo_uri"])]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} rows to: {OUTPUT_FILE}")
    print("Ready to drop into GSoC26_H/results/ for the HITL app.")


if __name__ == "__main__":
    main()
