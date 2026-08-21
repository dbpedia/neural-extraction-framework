"""
rescore_indie_official.py — Re-scores our real IndIE BenchIE output using
the OFFICIAL BenchIEDetailedComparator (the same evaluator that produced
the F1=0.46 Phase 1 result), instead of the naive strict-match script
that produced F1=0.076/0.151.

Reuses the real class directly -- does not reimplement its matching logic.

Run:
    python3 rescore_indie_official.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE"))
from detailed_comparison_using_benchIE import BenchIEDetailedComparator

GOLD_FILE = os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE/hindi_benchie_gold.txt")
INDIE_RESULTS_FILE = os.path.expanduser("~/indie_baseline_results.json")
RESULTS_DIR = os.path.expanduser("~/official_benchie_rescore")
MODEL_NAME = "IndIE"
STRATEGY = "reofficial"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading official gold standard (real class, real parsing)...")
    comparator = BenchIEDetailedComparator(GOLD_FILE, RESULTS_DIR)
    print(f"  Gold file has {len(comparator.sentences)} sentences.")

    print("\nLoading our real IndIE BenchIE results...")
    with open(INDIE_RESULTS_FILE, encoding="utf-8") as f:
        indie_data = json.load(f)
    benchie_results = indie_data.get("benchie", [])
    print(f"  Loaded {len(benchie_results)} IndIE BenchIE entries.")

    # Match each of our results to the gold file's sentence IDs by exact text match
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
        for triple in r.get("triples", []):
            if len(triple) != 3:
                continue
            subj, rel, obj = triple
            extraction_lines.append(f"{sent_id}\t{subj}\t{rel}\t{obj}")

    print(f"\n  Matched to gold sentence IDs: {matched}")
    print(f"  Unmatched (no gold sentence found): {unmatched}")

    if matched == 0:
        print("\nERROR: zero sentences matched -- sentence text likely differs between "
              "our JSON and the gold file. Stopping before producing a fake number.")
        return

    # Write in the exact filename format generate_report() expects
    out_path = os.path.join(RESULTS_DIR, f"extractions_{MODEL_NAME}_{STRATEGY}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(extraction_lines))
    print(f"  Wrote {len(extraction_lines)} extraction lines to: {out_path}")

    print("\nRunning the OFFICIAL BenchIE evaluation...")
    report = comparator.generate_report(model_name=MODEL_NAME, strategy=STRATEGY, save_to_json=True)

    if report is None:
        print("ERROR: generate_report returned None -- check the file path matched correctly.")
        return

    total_tp = sum(s["summary"]["TP"] for s in report["sentences"])
    total_fp = sum(s["summary"]["FP"] for s in report["sentences"])
    total_fn = sum(s["summary"]["FN"] for s in report["sentences"])

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"REAL, OFFICIAL-EVALUATOR INDIE BENCHIE SCORE")
    print(f"{'='*60}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    print(f"\n(compare: naive strict-match script gave F1=0.076 earlier)")


if __name__ == "__main__":
    main()
