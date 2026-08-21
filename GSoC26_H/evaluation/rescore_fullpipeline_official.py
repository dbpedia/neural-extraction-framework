"""
rescore_fullpipeline_official.py — Regenerates BenchIE full-pipeline
normalization for both IndIE and the fine-tuned model, this time SAVING
the actual per-sentence normalized (subject, dbo_relation, object)
triples (not just aggregate stats), then scores them with the OFFICIAL
BenchIEDetailedComparator instead of strict set-matching.

Read-only on the shared predicate cache -- safe to run without
disturbing any other job.

Run:
    python3 rescore_fullpipeline_official.py
"""
import sys
import os
import json
import io
import contextlib

sys.path.insert(0, os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE"))
from detailed_comparison_using_benchIE import BenchIEDetailedComparator

sys.path.insert(0, os.path.expanduser("~"))
import normalize_indie_fixed as norm_mod  # reuse real, verified normalize/model-loading logic

GOLD_FILE = os.path.expanduser("~/neural-extraction-framework/GSoC25_H/llm_IE/hindi_benchie_gold.txt")
RESULTS_DIR = os.path.expanduser("~/official_benchie_rescore")
INDIE_CONVERTED_FILE = os.path.expanduser("~/indie_converted_for_normalization_fixed.json")
FINETUNED_RESULTS_FILE = os.path.expanduser("~/eval_full_scale_results.json")


def normalize_readonly(predicate, model, catalog_vecs, catalog_uris, local_cache):
    if predicate in local_cache:
        return local_cache[predicate]
    top_uris = norm_mod.get_top_k(predicate, model, catalog_vecs, catalog_uris)
    result = norm_mod.ask_llm(predicate, top_uris)
    local_cache[predicate] = result
    return result


def score_with_official(name, sentence_triples, comparator):
    """sentence_triples: list of (sentence_text, [(s,rel,o), ...]) for predicted triples."""
    gold_text_to_id = {text.strip(): sid for sid, text in comparator.sentences.items()}
    matched, unmatched = 0, 0
    lines = []
    for sentence, triples in sentence_triples:
        sent_id = gold_text_to_id.get(sentence.strip())
        if sent_id is None:
            unmatched += 1
            continue
        matched += 1
        for s, rel, o in triples:
            lines.append(f"{sent_id}\t{s}\t{rel}\t{o}")

    print(f"  [{name}] matched={matched}  unmatched={unmatched}")
    out_path = os.path.join(RESULTS_DIR, f"extractions_{name}_fp.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = comparator.generate_report(model_name=name, strategy="fp", save_to_json=False)
    analyzed = [s for s in report["sentences"] if s.get("status") == "analyzed"]
    tp = sum(s["summary"]["TP"] for s in analyzed)
    fp = sum(s["summary"]["FP"] for s in analyzed)
    fn = sum(s["summary"]["FN"] for s in analyzed)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    print(f"  [{name}] analyzed={len(analyzed)}/112  TP={tp} FP={fp} FN={fn}  P={p:.4f} R={r:.4f} F1={f1:.4f}")
    return f1


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model + catalog + cache (read-only)...")
    model = norm_mod.SentenceTransformer(norm_mod.FINETUNED_MODEL, trust_remote_code=True)
    catalog_vecs, catalog_uris = norm_mod.load_catalog_and_embeddings(model)
    shared_cache = norm_mod.load_cache()
    local_cache = dict(shared_cache)
    print(f"  Cache: {len(shared_cache)} entries loaded as read-only snapshot.\n")

    print("=" * 60)
    print("Normalizing IndIE BenchIE (saving real triples this time)")
    print("=" * 60)
    with open(INDIE_CONVERTED_FILE, encoding="utf-8") as f:
        indie_data = json.load(f)
    indie_benchie = indie_data.get("benchie", [])

    indie_sentence_triples = []
    for r in indie_benchie:
        sentence = r["sentence"]
        pred_triples = norm_mod.parse_triples(r["predicted"])
        normed = []
        for s, p_rel, o in pred_triples:
            if norm_mod.is_property_relation(p_rel):
                continue
            dbo = normalize_readonly(p_rel, model, catalog_vecs, catalog_uris, local_cache)
            if dbo not in ("NONE", "UNPARSEABLE"):
                normed.append((s, dbo, o))
        indie_sentence_triples.append((sentence, normed))
    print(f"  Normalized {len(indie_sentence_triples)} IndIE BenchIE sentences.\n")

    print("=" * 60)
    print("Normalizing FineTuned BenchIE (saving real triples this time)")
    print("=" * 60)
    with open(FINETUNED_RESULTS_FILE, encoding="utf-8") as f:
        ft_data = json.load(f)
    ft_benchie = ft_data.get("benchie", [])

    ft_sentence_triples = []
    for r in ft_benchie:
        sentence = r["sentence"]
        pred_triples = norm_mod.parse_triples(r["predicted"])
        normed = []
        for s, p_rel, o in pred_triples:
            if norm_mod.is_property_relation(p_rel):
                continue
            dbo = normalize_readonly(p_rel, model, catalog_vecs, catalog_uris, local_cache)
            if dbo not in ("NONE", "UNPARSEABLE"):
                normed.append((s, dbo, o))
        ft_sentence_triples.append((sentence, normed))
    print(f"  Normalized {len(ft_sentence_triples)} FineTuned BenchIE sentences.\n")

    print("=" * 60)
    print("SCORING WITH OFFICIAL EVALUATOR")
    print("=" * 60)
    comparator1 = BenchIEDetailedComparator(GOLD_FILE, RESULTS_DIR)
    indie_f1 = score_with_official("IndIE_FP", indie_sentence_triples, comparator1)

    comparator2 = BenchIEDetailedComparator(GOLD_FILE, RESULTS_DIR)
    ft_f1 = score_with_official("FineTuned_FP", ft_sentence_triples, comparator2)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON -- full-pipeline BenchIE")
    print(f"{'='*60}")
    print(f"IndIE_FP:     F1={indie_f1:.4f}   (naive-script number was 0.151)")
    print(f"FineTuned_FP: F1={ft_f1:.4f}   (naive-script number was 0.173)")


if __name__ == "__main__":
    main()
