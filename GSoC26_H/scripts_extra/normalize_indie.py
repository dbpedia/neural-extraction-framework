"""
normalize_full_scale.py — Full-scale ontology normalization.
lr=2e-4 ONLY (no lr=1e-5 comparison — no full-scale extraction exists
for it). Uses TOP_K=40 per Debarghya's request (p@40 = 89.9% on true
held-out set). Fresh cache — the old 432-entry cache was built with
TOP_K=50 and would be stale/incorrect at k=40.

Runs on: eval_full_scale_results.json (1,817 Wikipedia + 50 Train +
         112 BenchIE, lr=2e-4, Round 2 F2LLM checkpoint)

Run:
    python3 normalize_full_scale.py
"""

import json
import os
import re
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
API_KEY = "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"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "openai/gpt-oss-120b"

FINETUNED_MODEL = "/home/nsingh/f2lm_finetuned_v2_merged"
CATALOG_FILE = "/home/nsingh/f2lm_property_catalog.json"

INPUT_FILE = "/home/nsingh/indie_converted_for_normalization.json"

OUTPUT_FILE = "/home/nsingh/normalization_results_indie.json"
CACHE_FILE = "/home/nsingh/normalization_cache_k40.json"   # fresh, k=40-specific

TOP_K = 40

FEW_SHOT = """
Example 1:
Predicate: का जन्म
Top candidates:
1. dbo:birthPlace
2. dbo:birthDate
3. dbo:almaMater
Correct answer: dbo:birthPlace

Example 2:
Predicate: का निर्माण किया
Top candidates:
1. dbo:builder
2. dbo:architect
3. dbo:foundedBy
Correct answer: dbo:builder

Example 3:
Predicate: बहुत सुंदर है
Top candidates:
1. dbo:genre
2. dbo:knownFor
3. dbo:related
Correct answer: NONE

Example 4:
Predicate: के मुख्यमंत्री हैं
Top candidates:
1. dbo:leaderName
2. dbo:leader
3. dbo:president
Correct answer: dbo:leaderName

Example 5:
Predicate: पुरस्कार जीता
Top candidates:
1. dbo:winner
2. dbo:award
3. dbo:participant
Correct answer: dbo:winner
"""

PROMPT_TEMPLATE = """You are an expert in Hindi linguistics and the DBpedia ontology. Given a Hindi predicate and a ranked list of candidate DBpedia ontology properties, select the SINGLE correct property.

Rules:
- Only pick a property if it is a genuinely accurate, precise match.
- If none fit, respond with NONE.
- End your response with EXACTLY: "ANSWER: dbo:propertyName" or "ANSWER: NONE"

{few_shot}

Predicate: {predicate}
Top candidates:
{candidates}
"""


# ── Catalog + embedding ───────────────────────────────────────

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
    return [catalog_uris[i] for i in top_idx]


# ── LLM disambiguation ───────────────────────────────────────

def parse_llm_answer(raw, valid_uris):
    match = re.search(r"ANSWER:\s*(dbo:\w+|NONE)", raw, re.IGNORECASE)
    if match:
        answer = match.group(1)
        if answer.upper() == "NONE":
            return "NONE"
        for uri in valid_uris:
            if uri.lower() == answer.lower():
                return uri
        return answer
    mentions = re.findall(r"dbo:\w+", raw)
    if mentions:
        last = mentions[-1]
        for uri in valid_uris:
            if uri.lower() == last.lower():
                return uri
        return last
    if re.search(r"\bNONE\b", raw):
        return "NONE"
    return "UNPARSEABLE"


def ask_llm(predicate, top_uris, retries=3):
    candidates_text = "\n".join(f"{i+1}. {uri}" for i, uri in enumerate(top_uris))
    valid_uris = set(top_uris)
    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT,
        predicate=predicate,
        candidates=candidates_text,
    )
    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 400,
            }, timeout=90)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"    Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            choice = resp.json()["choices"][0]["message"]
            raw = (choice.get("content") or
                   choice.get("reasoning_content") or "").strip()
            if not raw:
                raise ValueError("Empty response")
            return parse_llm_answer(raw, valid_uris)
        except Exception:
            if attempt < retries:
                time.sleep(3)
    return "NONE"


# ── Triple parsing ────────────────────────────────────────────

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


def is_property_relation(predicate):
    return predicate.strip().lower() == "property"


# ── Normalization cache ───────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def normalize_predicate(predicate, model, catalog_vecs, catalog_uris, cache):
    if predicate in cache:
        return cache[predicate]
    top_uris = get_top_k(predicate, model, catalog_vecs, catalog_uris)
    result = ask_llm(predicate, top_uris)
    cache[predicate] = result
    save_cache(cache)
    return result


# ── Scoring ───────────────────────────────────────────────────

def score_normalized(pred_triples_norm, gold_triples_norm):
    pred_set = {t for t in pred_triples_norm
                if t[1] not in ("NONE", "UNPARSEABLE", "property")}
    gold_set = {t for t in gold_triples_norm
                if t[1] not in ("NONE", "UNPARSEABLE", "property")}

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set:
        return 0.0, 0.0, 0.0
    if not gold_set:
        return 0.0, 0.0, 0.0

    correct = pred_set & gold_set
    precision = len(correct) / len(pred_set)
    recall = len(correct) / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ── Main ─────────────────────────────────────────────────────

def process_result_file(filepath, model, catalog_vecs, catalog_uris, cache):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    all_scores = {}
    total_predicates_normalized = 0

    for source in ("wikipedia", "train", "benchie"):
        results = data[source]
        precisions, recalls, f1s = [], [], []

        print(f"\n  Processing {source} ({len(results)} samples)...")
        for idx, r in enumerate(results):
            pred_triples = parse_triples(r["predicted"])
            gold_triples = parse_triples(r["reference"])

            pred_norm = []
            for s, p, o in pred_triples:
                if is_property_relation(p):
                    pred_norm.append((s, "property", o))
                else:
                    dbo = normalize_predicate(p, model, catalog_vecs, catalog_uris, cache)
                    pred_norm.append((s, dbo, o))
                    total_predicates_normalized += 1

            gold_norm = []
            for s, p, o in gold_triples:
                if is_property_relation(p):
                    gold_norm.append((s, "property", o))
                else:
                    dbo = normalize_predicate(p, model, catalog_vecs, catalog_uris, cache)
                    gold_norm.append((s, dbo, o))
                    total_predicates_normalized += 1

            p_score, r_score, f1 = score_normalized(pred_norm, gold_norm)
            precisions.append(p_score)
            recalls.append(r_score)
            f1s.append(f1)

            if (idx + 1) % 25 == 0:
                print(f"    [{idx+1}/{len(results)}] "
                      f"avg_f1={sum(f1s)/len(f1s):.3f}", flush=True)

        all_scores[source] = {
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": sum(f1s) / len(f1s),
            "n": len(results),
        }

    return all_scores, total_predicates_normalized


def main():
    print("=" * 60)
    print("Ontology Normalization — FULL SCALE (lr=2e-4 only)")
    print(f"TOP_K = {TOP_K} (per Debarghya's request, p@40=89.9% held-out)")
    print("Model: Round 2 F2LLM (9 epochs) + GPT-OSS-120B disambiguation")
    print("=" * 60)

    print("\nLoading fine-tuned F2LLM-v2 model...")
    model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)
    print("Loaded.")

    print("\nLoading catalog and computing embeddings...")
    catalog_vecs, catalog_uris = load_catalog_and_embeddings(model)

    print("\nLoading normalization cache (fresh, k=40-specific)...")
    cache = load_cache()
    print(f"  Cache has {len(cache)} existing predicate mappings.")

    print(f"\n{'='*60}")
    print(f"Processing: {INPUT_FILE}")
    print(f"{'='*60}")

    scores, n_normalized = process_result_file(
        INPUT_FILE, model, catalog_vecs, catalog_uris, cache
    )

    print(f"\n{'='*60}")
    print("FINAL RESULTS — Full Scale (lr=2e-4)")
    print(f"{'='*60}")
    print(f"{'Source':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}{'N':<8}")
    print("-" * 54)
    for source in ("wikipedia", "train", "benchie"):
        s = scores[source]
        print(f"{source:<12}{s['precision']:<12.3f}"
              f"{s['recall']:<12.3f}{s['f1']:<10.3f}{s['n']:<8}")
    print(f"\nTotal predicates normalized: {n_normalized}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {OUTPUT_FILE}")
    print(f"Cache saved to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
