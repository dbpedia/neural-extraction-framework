"""
build_gold_set_chunk0.py — Chunk 0 of 2. Predicates 0-4017.
"""

import json, os, re, time, requests
import numpy as np
from sentence_transformers import SentenceTransformer

API_KEY = os.environ["NVIDIA_API_KEY"]
OUTPUT_FILE = os.path.expanduser("~/gold_set_chunk0.jsonl")
START_IDX = 0
END_IDX = 4017

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "openai/gpt-oss-120b"

PREDICATES_FILE = os.path.expanduser("~/unique_predicates.json")
EMBEDDINGS_FILE = os.path.expanduser("~/f2lm_property_embeddings.npy")
CATALOG_FILE = os.path.expanduser("~/f2lm_property_catalog.json")

TOP_K = 50

FEW_SHOT_EXAMPLES = """
Example 1:
Predicate: का जन्म
Top candidates:
1. dbo:birthPlace | born in birth place birthplace
2. dbo:birthDate | birth date born year
3. dbo:almaMater | alma mater studied university degree
4. dbo:origin | origin source came from
5. dbo:nationality | nationality citizen country
(45 more candidates omitted for brevity)
Correct answer: dbo:birthPlace

Example 2:
Predicate: का निर्माण किया
Top candidates:
1. dbo:builder | built by constructed by builder architect
2. dbo:architect | architect designed building
3. dbo:foundedBy | founded by established creator
4. dbo:producer | producer produced made
5. dbo:author | author writer wrote composed
(45 more candidates omitted for brevity)
Correct answer: dbo:builder

Example 3:
Predicate: बहुत सुंदर है
Top candidates:
1. dbo:genre | genre type style
2. dbo:knownFor | known for famous for recognised
3. dbo:related | related connected associated linked
4. dbo:origin | origin source came from
5. dbo:movement | movement ideology
(45 more candidates omitted for brevity)
Correct answer: NONE

Example 4:
Predicate: के मुख्यमंत्री हैं
Top candidates:
1. dbo:leaderName | leader head chief minister
2. dbo:leader | leader head president chief
3. dbo:president | president
4. dbo:position | position role plays as
5. dbo:successor | successor followed replaced
(45 more candidates omitted for brevity)
Correct answer: dbo:leaderName

Example 5:
Predicate: पुरस्कार जीता
Top candidates:
1. dbo:winner | winner won champion first
2. dbo:award | award prize honour
3. dbo:participant | participant involved competed
4. dbo:position | position role plays as
5. dbo:knownFor | known for famous for recognised
(45 more candidates omitted for brevity)
Correct answer: dbo:winner
"""

PROMPT_TEMPLATE = """You are an expert in Hindi linguistics and the DBpedia ontology. Given a Hindi predicate (relation phrase) and a ranked list of 50 candidate DBpedia ontology properties, select the SINGLE correct property that this predicate most precisely maps to.

Rules:
- Only pick a property if it is a genuinely accurate, precise match for the predicate's meaning.
- If none of the 50 candidates are a good match, respond with NONE.
- End your response with a final line in EXACTLY this format: "ANSWER: dbo:propertyName" or "ANSWER: NONE"

{few_shot}

Now evaluate this:

Predicate: {predicate}
Top candidates:
{candidates}
"""


def load_predicates():
    with open(PREDICATES_FILE, encoding="utf-8") as f:
        preds = json.load(f)
    if END_IDX is not None:
        return preds[START_IDX:END_IDX]
    return preds[START_IDX:]


def load_catalog_and_embeddings():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    embeddings = np.load(EMBEDDINGS_FILE)
    return catalog, embeddings


def get_top_k_candidates(predicate, model, catalog, embeddings, k=TOP_K):
    q_vec = model.encode([predicate], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    sims = embeddings @ q_vec
    top_idx = np.argsort(-sims)[:k]

    lines = []
    valid_uris = set()
    for rank, idx in enumerate(top_idx, 1):
        entry = catalog[idx]
        uri = entry["property_uri"]
        short_uri = "dbo:" + uri.split("/")[-1]
        valid_uris.add(short_uri)
        labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
        label_text = " ".join(labels[:3]) if labels else short_uri
        lines.append(f"{rank}. {short_uri} | {label_text}")
    return "\n".join(lines), [catalog[idx]["property_uri"] for idx in top_idx], valid_uris


def parse_answer(raw_text, valid_uris):
    marker_match = re.search(r"ANSWER:\s*(dbo:\w+|NONE)", raw_text, re.IGNORECASE)
    if marker_match:
        answer = marker_match.group(1)
        if answer.upper() == "NONE":
            return "NONE"
        for uri in valid_uris:
            if uri.lower() == answer.lower():
                return uri
        return answer

    all_dbo_mentions = re.findall(r"dbo:\w+", raw_text)
    if all_dbo_mentions:
        last = all_dbo_mentions[-1]
        for uri in valid_uris:
            if uri.lower() == last.lower():
                return uri
        return last

    if re.search(r"\bNONE\b", raw_text):
        return "NONE"

    return "UNPARSEABLE"


def ask_llm(predicate, candidates_text, valid_uris, retries=3):
    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT_EXAMPLES,
        predicate=predicate,
        candidates=candidates_text,
    )

    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 800,
            }, timeout=120)

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            choice = resp.json()["choices"][0]["message"]
            raw = (choice.get("content") or choice.get("reasoning_content") or "").strip()
            if not raw:
                raise ValueError("Empty response")

            return parse_answer(raw, valid_uris), raw

        except Exception:
            if attempt < retries:
                time.sleep(3)

    return "ERROR", ""


def main():
    print("Loading F2LM model...")
    model = SentenceTransformer("codefuse-ai/F2LLM-v2-1.7B", trust_remote_code=True)
    print("Loaded.")

    print("Loading property catalog and embeddings...")
    catalog, embeddings = load_catalog_and_embeddings()
    print(f"Catalog size: {len(catalog)}")

    predicates = load_predicates()
    print(f"Predicates to process (this chunk): {len(predicates)}")

    done_predicates = set()
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("gold_dbo") not in (None, "ERROR", "UNPARSEABLE"):
                        done_predicates.add(item["predicate"])
        print(f"Resuming — {len(done_predicates)} already done.")

    remaining = [p for p in predicates if p not in done_predicates]
    print(f"Remaining: {len(remaining)}")

    success = failed = completed = 0
    start_time = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for predicate in remaining:
            candidates_text, candidate_uris, valid_uris = get_top_k_candidates(predicate, model, catalog, embeddings)
            answer, raw_response = ask_llm(predicate, candidates_text, valid_uris)

            result = {
                "predicate": predicate,
                "top_50_candidates": candidate_uris,
                "gold_dbo": answer,
                "raw_llm_response": raw_response,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if answer not in ("ERROR", "UNPARSEABLE"):
                success += 1
            else:
                failed += 1

            completed += 1
            if completed % 10 == 0:
                out_f.flush()
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed * 3600
                eta = (len(remaining) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(remaining)}] ok:{success} err:{failed} | {rate:.0f}/hr | ETA {eta:.1f}h")

    print(f"\nDone. Success: {success} | Errors: {failed}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
