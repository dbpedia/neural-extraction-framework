"""
score_indie_baseline.py — Scores IndIE's raw extracted triples against
gold triples, using the same set-based precision/recall/F1 methodology
as extraction_only_score.py. IndIE has no predicate-linking step, so
this is inherently an extraction-only comparison.

Honest note: IndIE's triple format (raw dependency-parse spans) follows
different span conventions than our own extraction schema's gold data.
Exact-match scoring may understate IndIE's real quality for this reason
-- flagged here explicitly rather than presented as a clean apples-to-
apples number.

Train section covers the FULL 39,621-sentence training set (not the
50-sample subset used elsewhere), by deliberate choice -- not directly
comparable to other systems' Train F1 without noting this difference.

Run:
    python3 score_indie_baseline.py
"""
import json

INDIE_RESULTS_FILE = "/home/nsingh/indie_baseline_results.json"
EXTRACTION_RESULTS_FILE = "/home/nsingh/eval_full_scale_results.json"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"


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
    with open(INDIE_RESULTS_FILE, encoding="utf-8") as f:
        indie_data = json.load(f)

    # Wikipedia and BenchIE: gold lives in eval_full_scale_results.json,
    # matched by sentence text (same file/order used throughout the project)
    with open(EXTRACTION_RESULTS_FILE, encoding="utf-8") as f:
        extraction_data = json.load(f)

    gold_by_sentence = {"wikipedia": {}, "benchie": {}}
    for source in ("wikipedia", "benchie"):
        for r in extraction_data.get(source, []):
            gold_by_sentence[source][r["sentence"].strip()] = parse_triples(r["reference"])

    # Train: gold lives directly in the training file itself
    train_gold_by_sentence = {}
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            gold_text = entry["messages"][2]["content"]
            if "[ANSWER]" in gold_text:
                gold_text = gold_text.split("[ANSWER]")[-1]
            train_gold_by_sentence[sentence] = parse_triples(gold_text)

    print("=" * 60)
    print("INDIE BASELINE SCORE (extraction-only, no predicate linking)")
    print("=" * 60)

    for source in ("wikipedia", "train", "benchie"):
        results = indie_data.get(source, [])
        if not results:
            continue

        precisions, recalls, f1s = [], [], []
        unmatched = 0
        for r in results:
            sentence = r["sentence"].strip()
            pred_triples = [tuple(t) for t in r.get("triples", [])]

            if source == "train":
                gold_triples = train_gold_by_sentence.get(sentence)
            else:
                gold_triples = gold_by_sentence[source].get(sentence)

            if gold_triples is None:
                unmatched += 1
                continue

            p, rec, f1 = score(pred_triples, gold_triples)
            precisions.append(p)
            recalls.append(rec)
            f1s.append(f1)

        n = len(precisions)
        if n == 0:
            print(f"{source:12s} NO MATCHED GOLD DATA -- 0 scored, {unmatched} unmatched")
            continue

        note = " (FULL 39,621-sentence set, not the 50-sample subset)" if source == "train" else ""
        print(f"{source:12s} P={sum(precisions)/n:.3f}  R={sum(recalls)/n:.3f}  "
              f"F1={sum(f1s)/n:.3f}  N={n}  (unmatched: {unmatched}){note}")


if __name__ == "__main__":
    main()
