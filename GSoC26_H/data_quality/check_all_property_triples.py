"""
check_all_property_triples.py — Checks EVERY "property"-type triple
(not a sample) using GPT-OSS-120B, classifying each as either:
  GENUINE  — correctly non-relational (adjective+noun, determiner+noun,
             number+unit fragment, etc.) — no DBpedia property applies
  MISLABELED — actually represents a real, mappable relationship that
             should have gone through DBO matching instead

Corrupted-sentence triples (coreference-reasoning leakage) are filtered
out first — no point judging garbage input.

FIXED (previous run's results were invalid — all verdicts were
reasoning-preamble truncated before the model ever stated GENUINE or
MISLABELED, causing mislabeled_count to stay at 0 through 4,800 items):
  - max_tokens raised from 100 to 600, giving the reasoning model room
    to actually finish before answering
  - Verdict matching now checks the END of the response for the final
    GENUINE/MISLABELED line, instead of checking if the response
    STARTS WITH it (it never did, since every response opens with
    reasoning text like "We need to decide if...")
  - API key now loaded from environment, not hardcoded

Run (inside tmux — this will take a while, ~6000+ API calls):
    python3 check_all_property_triples.py
"""

import json
import os
import re
import time
import requests

EXTRACTION_FILE = "/home/nsingh/eval_full_scale_results.json"
OUTPUT_FILE = "/home/nsingh/property_triple_audit.jsonl"
API_KEY = os.environ["NVIDIA_API_KEY"]
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL = "openai/gpt-oss-120b"

TELLTALE_PHRASES = [
    "we need to see", "need to see if pronoun", "refers to an entity",
    "named elsewhere in", "the rule:", "the instruction:",
]

FEW_SHOT = """
Example 1:
Sentence: वेंटवुड समुद्र तल से 310 मीटर ऊपर पहुंचता है।
Subject: 310
Object: मीटर
Verdict: GENUINE (number + unit fragment, not a relationship)

Example 2:
Sentence: बसरेहर भारत के उत्तर प्रदेश राज्य के इटावा ज़िला में स्थित एक गाँव है।
Subject: उत्तर प्रदेश राज्य
Object: इटावा ज़िला
Verdict: MISLABELED (this is a real containment relationship — state contains district — should map to a DBpedia property like isPartOf or state)

Example 3:
Sentence: इस दवा के प्रयोग पर रोक लगाने को काफी विवादास्पद माना गया है।
Subject: इस
Object: दवा
Verdict: GENUINE (determiner + noun, not a relationship)

Example 4:
Sentence: अज़रबैजान का दूतावास नई दिल्ली में है।
Subject: नई
Object: दिल्ली
Verdict: MISLABELED (this is a broken entity split — "नई दिल्ली" is one place name, New Delhi — not a real subject-object pair, but flagging since it indicates an extraction error worth knowing about)
"""

PROMPT_TEMPLATE = """You are auditing Hindi triple extraction. Given a sentence and a (subject, object) pair that was labeled as non-relational "property" type, decide if that label is correct.

GENUINE = correctly non-relational (adjective+noun, determiner+noun, number+unit fragment, vague phrase — nothing DBpedia would represent as a property)
MISLABELED = this actually represents a real, mappable relationship (geographic containment, birth/death facts, roles, affiliations, etc.) that should have been extracted as a real relation instead of "property"

{few_shot}

Sentence: {sentence}
Subject: {subject}
Object: {object}

Think briefly if needed, but you MUST end your response with exactly one final line, with nothing after it, in exactly this format:
FINAL VERDICT: GENUINE
or
FINAL VERDICT: MISLABELED: <short reason>
"""


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


def is_corrupted(sentence):
    lower = sentence.lower()
    return any(phrase in lower for phrase in TELLTALE_PHRASES)


def ask_llm(sentence, subject, obj, retries=3):
    prompt = PROMPT_TEMPLATE.format(
        few_shot=FEW_SHOT, sentence=sentence, subject=subject, object=obj,
    )
    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 600,
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
            return raw
        except Exception:
            if attempt < retries:
                time.sleep(3)
    return "ERROR"


def classify_verdict(raw):
    """Look for the final 'FINAL VERDICT: ...' line, searching from the
    END of the response backwards, since the real answer is always the
    last thing the model says after its reasoning."""
    match = re.search(r"FINAL VERDICT:\s*(GENUINE|MISLABELED)", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: response got cut off even at 600 tokens (rare) — treat
    # as an error rather than silently defaulting to GENUINE.
    return "ERROR"


def main():
    print("=" * 60)
    print("Full Audit — ALL 'property'-type triples")
    print("=" * 60)

    with open(EXTRACTION_FILE, encoding="utf-8") as f:
        data = json.load(f)

    all_candidates = []
    skipped_corrupted = 0

    for source in ("wikipedia", "train", "benchie"):
        for r in data.get(source, []):
            sentence = r["sentence"]
            if is_corrupted(sentence):
                triples = parse_triples(r["predicted"])
                skipped_corrupted += sum(
                    1 for _, rel, _ in triples if rel.strip().lower() == "property"
                )
                continue
            triples = parse_triples(r["predicted"])
            for subject, relation, obj in triples:
                if relation.strip().lower() == "property":
                    all_candidates.append({
                        "source": source, "sentence": sentence,
                        "subject": subject, "object": obj,
                    })

    print(f"Total property triples: {len(all_candidates) + skipped_corrupted}")
    print(f"Skipped (corrupted sentence): {skipped_corrupted}")
    print(f"To audit: {len(all_candidates)}")

    results = []
    genuine_count = 0
    mislabeled_count = 0
    error_count = 0

    for i, c in enumerate(all_candidates):
        raw = ask_llm(c["sentence"], c["subject"], c["object"])
        verdict_label = classify_verdict(raw) if raw != "ERROR" else "ERROR"

        if verdict_label == "MISLABELED":
            mislabeled_count += 1
        elif verdict_label == "ERROR":
            error_count += 1
        else:
            genuine_count += 1

        results.append({**c, "verdict_label": verdict_label, "raw_response": raw})

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(all_candidates)}] "
                  f"genuine={genuine_count} mislabeled={mislabeled_count} errors={error_count}",
                  flush=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                for r in results:
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for r in results:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total audited: {len(all_candidates)}")
    print(f"Genuine (correctly non-relational): {genuine_count} ({genuine_count/len(all_candidates)*100:.1f}%)")
    print(f"Mislabeled (real relation, lost): {mislabeled_count} ({mislabeled_count/len(all_candidates)*100:.1f}%)")
    print(f"Errors: {error_count}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
