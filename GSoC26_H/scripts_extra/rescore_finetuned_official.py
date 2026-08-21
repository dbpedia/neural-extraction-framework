"""
rescore_finetuned_official.py — Re-scores the fine-tuned model's real
BenchIE predictions using the OFFICIAL BenchIEDetailedComparator, the
same evaluator just used to correct IndIE's number, so the baseline
table's BenchIE column is genuinely apples-to-apples.

Run:
    python3 rescore_finetuned_official.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.expanduser("~/neural-extraction-framework/GSoC26_H/llm_IE")
                 if os.path.exists(os.path.expanduser("~/neural-extraction-framework/GSoC26_H/llm_IE"))
                 else os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE"))
from detailed_comparison_using_benchIE import BenchIEDetailedComparator

GOLD_FILE = os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE/hindi_benchie_gold.txt")
FINETUNED_RESULTS_FILE = os.path.expanduser("~/eval_full_scale_results.json")
RESULTS_DIR = os.path.expanduser("~/official_benchie_rescore")
MODEL_NAME = "FineTuned"
STRATEGY = "reofficial"


def parse_pipe_triples(text):
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading official gold standard...")
    comparator = BenchIEDetailedComparator(GOLD_FILE, RESULTS_DIR)
    print(f"  Gold file has {len(comparator.sentences)} sentences.")

    print("\nLoading real fine-tuned model BenchIE results...")
    with open(FINETUNED_RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    benchie_results = data.get("benchie", [])
    print(f"  Loaded {len(benchie_results)} entries.")

    gold_text_to_id = {text.strip(): sid for sid, text in comparator.sentences.items()}

    matched = 0
    unmatched = 0
    extraction_lines = []
    for r in benchie_results:
        sentence = r["sentence"].strip()
        sent_id = gold_text_to_id.get(sentence)
        if sent_id is None:
            unmatched += 1
            continue
        matched += 1
        # Exclude "property" relations, same convention as everywhere else --
        # only real relational triples go to DBpedia/BenchIE matching.
        for subj, rel, obj in parse_pipe_triples(r["predicted"]):
            if rel.strip().lower() == "property":
                continue
            extraction_lines.append(f"{sent_id}\t{subj}\t{rel}\t{obj}")

    print(f"\n  Matched to gold sentence IDs: {matched}")
    print(f"  Unmatched: {unmatched}")

    if matched == 0:
        print("\nERROR: zero sentences matched -- stopping before producing a fake number.")
        return

    out_path = os.path.join(RESULTS_DIR, f"extractions_{MODEL_NAME}_{STRATEGY}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(extraction_lines))
    print(f"  Wrote {len(extraction_lines)} extraction lines to: {out_path}")

    print("\nRunning the OFFICIAL BenchIE evaluation...")
    report = comparator.generate_report(model_name=MODEL_NAME, strategy=STRATEGY, save_to_json=True)

    if report is None:
        print("ERROR: generate_report returned None.")
        return

    total_tp = sum(s["summary"]["TP"] for s in report["sentences"])
    total_fp = sum(s["summary"]["FP"] for s in report["sentences"])
    total_fn = sum(s["summary"]["FN"] for s in report["sentences"])

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"REAL, OFFICIAL-EVALUATOR FINE-TUNED MODEL BENCHIE SCORE (extraction-only)")
    print(f"{'='*60}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"\n(compare: naive strict-match extraction-only score was F1=0.056)")


if __name__ == "__main__":
    main()
