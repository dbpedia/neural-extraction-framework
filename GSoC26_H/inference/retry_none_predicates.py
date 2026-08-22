"""
retry_none_predicates.py — For the 2,174 predicates that returned NONE
in the original gold set (top-50 candidates checked, no match found),
retrieve the NEXT top-50 (ranks 51-100) using F2LLM-8B, then ask
GPT-OSS-120B if a correct DBO exists among those.

Per Debarghya: no fine-tuning here — just deeper retrieval + a test,
using the same F2LLM-8B model that built the original gold set.

Run:
    python3 retry_none_predicates.py
"""

import json
import re
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

GOLD_SET_FILE = "/home/nsingh/gold_set_8b_final.jsonl"
EMBEDDINGS_FILE = "/home/nsingh/f2lm8b_property_embeddings.npy"
CATALOG_FILE = "/home/nsingh/f2lm8b_property_catalog.json"
OUTPUT_FILE = "/home/nsingh/none_retry_results.jsonl"

TOP_K_FULL = 100   # retrieve up to rank 100
SKIP_FIRST = 50    # skip ranks 1-50 (already checked, confirmed no match)

API_KEY = "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"
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "openai/gpt-oss-120b"

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


def load_none_predicates():
    entries = []
    with open(GOLD_SET_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["gold_dbo"] == "NONE":
                entries.append(entry)
    return entries


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


def ask_llm(predicate, candidates_51_100, retries=3):
    candidates_text = "\n".join(
        f"{i+1}. {uri}" for i, uri in enumerate(candidates_51_100)
    )
    valid_uris = set(candidates_51_100)
    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT, predicate=predicate, candidates=candidates_text,
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
            raw = (choice.get("content") or choice.get("reasoning_content") or "").strip()
            if not raw:
                raise ValueError("Empty response")
            return parse_llm_answer(raw, valid_uris)
        except Exception:
            if attempt < retries:
                time.sleep(3)
    return "NONE"


def main():
    print("=" * 60)
    print("NONE-predicate retry — ranks 51-100, F2LLM-8B + GPT-OSS-120B")
    print("=" * 60)

    print("\nLoading NONE predicates from original gold set...")
    none_entries = load_none_predicates()
    print(f"NONE predicates to retry: {len(none_entries)}")

    print("\nLoading F2LLM-v2-8B in 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = SentenceTransformer(
        "codefuse-ai/F2LLM-v2-8B",
        trust_remote_code=True,
        model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
    )
    print("Loaded.")

    print("\nLoading catalog + precomputed embeddings...")
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    catalog_embeddings = np.load(EMBEDDINGS_FILE)
    catalog_uris = [entry["property_uri"] for entry in catalog]
    print(f"Catalog size: {len(catalog_uris)}")

    predicates = [e["predicate"] for e in none_entries]
    print(f"\nEncoding {len(predicates)} NONE predicates...")
    query_vecs = model.encode(
        predicates, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True, batch_size=8,
    ).astype(np.float32)

    recovered = 0
    still_none = 0
    results = []

    print(f"\nRetrieving ranks {SKIP_FIRST+1}-{TOP_K_FULL} and asking LLM...")
    for i, predicate in enumerate(predicates):
        sims = catalog_embeddings @ query_vecs[i]
        top_idx = np.argsort(-sims)[:TOP_K_FULL]
        candidates_51_100 = [catalog_uris[idx] for idx in top_idx[SKIP_FIRST:TOP_K_FULL]]

        decision = ask_llm(predicate, candidates_51_100)

        if decision not in ("NONE", "UNPARSEABLE"):
            recovered += 1
        else:
            still_none += 1

        results.append({
            "predicate": predicate,
            "candidates_51_100": candidates_51_100,
            "decision": decision,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(predicates)}] recovered so far: {recovered}", flush=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for r in results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total NONE predicates retried: {len(predicates)}")
    print(f"Recovered a real DBO in ranks 51-100: {recovered} "
          f"({recovered/len(predicates)*100:.1f}%)")
    print(f"Still NONE (no match even in 51-100): {still_none} "
          f"({still_none/len(predicates)*100:.1f}%)")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
