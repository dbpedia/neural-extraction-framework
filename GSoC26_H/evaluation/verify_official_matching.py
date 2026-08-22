"""
verify_official_matching.py — Runs the official evaluator for BOTH IndIE
and the fine-tuned model, but redirects the huge per-sentence dump to a
file instead of the screen, so we can actually SEE the matched/unmatched
counts and final stats clearly, and verify nothing silently failed.

Run:
    python3 verify_official_matching.py
"""
import sys
import os
import json
import io
import contextlib

sys.path.insert(0, os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE"))
from detailed_comparison_using_benchIE import BenchIEDetailedComparator

GOLD_FILE = os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE/hindi_benchie_gold.txt")
RESULTS_DIR = os.path.expanduser("~/official_benchie_rescore")


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


def check_system(name, results_file, source_key, is_indie):
    print(f"\n{'='*70}")
    print(f"CHECKING: {name}")
    print(f"{'='*70}")

    comparator = BenchIEDetailedComparator(GOLD_FILE, RESULTS_DIR)
    n_gold_sentences = len(comparator.sentences)
    print(f"Gold file sentences: {n_gold_sentences}")

    # Check for duplicate sentence texts in gold (would make matching ambiguous)
    gold_texts = list(comparator.sentences.values())
    n_unique_gold_texts = len(set(t.strip() for t in gold_texts))
    if n_unique_gold_texts != n_gold_sentences:
        print(f"WARNING: {n_gold_sentences - n_unique_gold_texts} duplicate sentence "
              f"text(s) in gold file -- matching by text could be ambiguous!")
    else:
        print(f"Confirmed: all {n_gold_sentences} gold sentence texts are unique.")

    gold_text_to_id = {text.strip(): sid for sid, text in comparator.sentences.items()}

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get(source_key, [])
    print(f"Our results file has: {len(results)} entries")

    matched = 0
    unmatched = 0
    unmatched_examples = []
    extraction_lines = []

    for r in results:
        sentence = r["sentence"].strip()
        sent_id = gold_text_to_id.get(sentence)
        if sent_id is None:
            unmatched += 1
            if len(unmatched_examples) < 3:
                unmatched_examples.append(sentence[:60])
            continue
        matched += 1

        if is_indie:
            for triple in r.get("triples", []):
                if len(triple) == 3:
                    subj, rel, obj = triple
                    extraction_lines.append(f"{sent_id}\t{subj}\t{rel}\t{obj}")
        else:
            for subj, rel, obj in parse_pipe_triples(r["predicted"]):
                if rel.strip().lower() != "property":
                    extraction_lines.append(f"{sent_id}\t{subj}\t{rel}\t{obj}")

    print(f"\n  MATCHED:   {matched} / {len(results)}")
    print(f"  UNMATCHED: {unmatched} / {len(results)}")
    if unmatched_examples:
        print(f"  Sample unmatched sentences (first 3):")
        for ex in unmatched_examples:
            print(f"    - {ex}")

    if matched == 0:
        print("  STOPPING: zero matches, cannot score.")
        return None

    out_path = os.path.join(RESULTS_DIR, f"extractions_{name}_verify.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(extraction_lines))

    # Suppress the huge per-sentence printout -- capture it instead
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = comparator.generate_report(model_name=name, strategy="verify", save_to_json=False)

    if report is None:
        print("  ERROR: generate_report returned None.")
        return None

    total_tp = sum(s["summary"]["TP"] for s in report["sentences"])
    total_fp = sum(s["summary"]["FP"] for s in report["sentences"])
    total_fn = sum(s["summary"]["FN"] for s in report["sentences"])
    n_scored_sentences = len(report["sentences"])

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n  Sentences actually scored by comparator: {n_scored_sentences} (should equal gold count: {n_gold_sentences})")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    return {"matched": matched, "unmatched": unmatched, "f1": f1,
            "precision": precision, "recall": recall,
            "n_scored": n_scored_sentences, "n_gold": n_gold_sentences}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    indie_result = check_system(
        "IndIE",
        os.path.expanduser("~/indie_baseline_results.json"),
        "benchie",
        is_indie=True,
    )

    ft_result = check_system(
        "FineTuned",
        os.path.expanduser("~/eval_full_scale_results.json"),
        "benchie",
        is_indie=False,
    )

    print(f"\n{'='*70}")
    print("FINAL VERIFIED COMPARISON")
    print(f"{'='*70}")
    if indie_result:
        print(f"IndIE:      matched={indie_result['matched']}  scored={indie_result['n_scored']}/{indie_result['n_gold']}  "
              f"P={indie_result['precision']:.4f}  R={indie_result['recall']:.4f}  F1={indie_result['f1']:.4f}")
    if ft_result:
        print(f"FineTuned:  matched={ft_result['matched']}  scored={ft_result['n_scored']}/{ft_result['n_gold']}  "
              f"P={ft_result['precision']:.4f}  R={ft_result['recall']:.4f}  F1={ft_result['f1']:.4f}")


if __name__ == "__main__":
    main()
