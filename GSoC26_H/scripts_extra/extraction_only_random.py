"""
extraction_only_score.py — Computes extraction-only precision/recall/F1,
comparing raw predicted triples against raw gold triples DIRECTLY (Hindi
text on both subject/relation/object), with NO predicate linking / DBO
normalization involved. Isolates extraction quality from predicate
linking quality, per Debarghya's request to report the two separately.

Reuses the exact parse_triples() logic already verified in
normalize_full_scale.py. Property-type triples ARE included here
(unlike the full-pipeline score), since correctly identifying a
property-type fragment is still a correct extraction -- the
relational/property split only matters at the predicate-linking stage,
which this script does not touch.

Run:
    python3 extraction_only_score.py
"""
import json

INPUT_FILE = "/home/nsingh/eval_few_shot_results.json"


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


def score(pred_triples, gold_triples):
    pred_set = set(pred_triples)
    gold_set = set(gold_triples)
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    correct = pred_set & gold_set
    precision = len(correct) / len(pred_set)
    recall = len(correct) / len(gold_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("EXTRACTION-ONLY SCORE (raw triples, no predicate linking)")
    print("=" * 60)
    for source in ("wikipedia", "train", "benchie"):
        results = data.get(source, [])
        if not results:
            continue
        precisions, recalls, f1s = [], [], []
        for r in results:
            pred = parse_triples(r["predicted"])
            gold = parse_triples(r["reference"])
            p, rec, f1 = score(pred, gold)
            precisions.append(p)
            recalls.append(rec)
            f1s.append(f1)
        n = len(results)
        print(f"{source:12s} P={sum(precisions)/n:.3f}  R={sum(recalls)/n:.3f}  "
              f"F1={sum(f1s)/n:.3f}  N={n}")


if __name__ == "__main__":
    main()
