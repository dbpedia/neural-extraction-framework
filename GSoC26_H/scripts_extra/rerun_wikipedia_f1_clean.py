"""
rerun_wikipedia_f1_clean.py — Recomputes Wikipedia F1 excluding the 79
corrupted validation sentences (leftover coreference-reasoning text),
using the exact same scoring logic as normalize_full_scale.py and the
exact same corruption detector as check_all_property_triples.py.

Uses the normalization cache directly (already fully populated by the
original full-scale run, now including the 1,055 merged NONE-retry
matches) -- no live API calls needed, so this reproduces exactly what
a full live rerun would produce.

Run:
    python3 rerun_wikipedia_f1_clean.py
"""

import json

EXTRACTION_FILE = "/home/nsingh/eval_full_scale_results.json"
CACHE_FILE = "/home/nsingh/normalization_cache_k40.json"

TELLTALE_PHRASES = [
    "we need to see", "need to see if pronoun", "refers to an entity",
    "named elsewhere in", "the rule:", "the instruction:",
]


def is_corrupted(sentence):
    lower = sentence.lower()
    return any(phrase in lower for phrase in TELLTALE_PHRASES)


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


def normalize_triples(triples, cache):
    """Matches normalize_full_scale.py's logic: property-tagged as
    'property', real relations looked up in the cache."""
    out = []
    for s, p, o in triples:
        if p.strip().lower() == "property":
            out.append((s, "property", o))
        else:
            dbo = cache.get(p, "NONE")
            out.append((s, dbo, o))
    return out


def score_normalized(pred_triples_norm, gold_triples_norm):
    """Exact same logic as normalize_full_scale.py's score_normalized."""
    pred_set = {t for t in pred_triples_norm
                if t[1] not in ("NONE", "UNPARSEABLE", "property")}
    gold_set = {t for t in gold_triples_norm
                if t[1] not in ("NONE", "UNPARSEABLE", "property")}

    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0

    correct = pred_set & gold_set
    precision = len(correct) / len(pred_set)
    recall = len(correct) / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def main():
    with open(EXTRACTION_FILE, encoding="utf-8") as f:
        data = json.load(f)
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    wiki_results = data["wikipedia"]
    print(f"Total Wikipedia sentences: {len(wiki_results)}")

    clean = [r for r in wiki_results if not is_corrupted(r["sentence"])]
    corrupted = [r for r in wiki_results if is_corrupted(r["sentence"])]
    print(f"Corrupted (excluded): {len(corrupted)}")
    print(f"Clean (rescored): {len(clean)}")
    print()

    def average_f1(results):
        precisions, recalls, f1s = [], [], []
        for r in results:
            pred = parse_triples(r["predicted"])
            gold = parse_triples(r["reference"])
            pred_norm = normalize_triples(pred, cache)
            gold_norm = normalize_triples(gold, cache)
            p, rec, f1 = score_normalized(pred_norm, gold_norm)
            precisions.append(p)
            recalls.append(rec)
            f1s.append(f1)
        n = len(results)
        return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n

    orig_p, orig_r, orig_f1 = average_f1(wiki_results)
    clean_p, clean_r, clean_f1 = average_f1(clean)

    print("=" * 60)
    print("ORIGINAL (all 1,817 sentences, including 79 corrupted)")
    print("=" * 60)
    print(f"  Precision: {orig_p:.3f}")
    print(f"  Recall:    {orig_r:.3f}")
    print(f"  F1:        {orig_f1:.3f}")
    print()
    print("=" * 60)
    print(f"CLEAN ({len(clean)} sentences, 79 corrupted excluded)")
    print("=" * 60)
    print(f"  Precision: {clean_p:.3f}")
    print(f"  Recall:    {clean_r:.3f}")
    print(f"  F1:        {clean_f1:.3f}")
    print()
    print(f"F1 change: {orig_f1:.3f} -> {clean_f1:.3f} ({(clean_f1 - orig_f1):+.3f})")


if __name__ == "__main__":
    main()
