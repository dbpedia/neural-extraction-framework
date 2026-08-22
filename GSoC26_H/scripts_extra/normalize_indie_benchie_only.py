"""
normalize_indie_benchie_only.py — Computes IndIE's full-pipeline BenchIE
score RIGHT NOW, in parallel with the still-running normalize_indie_fixed.py
(which will eventually reach BenchIE on its own, but Train is taking a
long time first).

SAFETY: imports normalize_indie_fixed.py as a module (its own
if __name__=="__main__" guard means main() does NOT get triggered by
this import) to reuse its exact, already-verified functions --
load_catalog_and_embeddings, get_top_k, ask_llm, parse_triples,
is_property_relation, score_normalized, load_cache -- with zero risk of
transcription differences.

CRITICAL: this script loads the cache ONCE (read-only) and NEVER calls
save_cache(). Any new predicate encountered gets its own LOCAL,
in-memory-only cache entry -- never written back to the shared
CACHE_FILE. This avoids any write conflict with the other script still
running against the same file.

Run:
    python3 normalize_indie_benchie_only.py
"""
import sys
import json
import importlib.util

spec = importlib.util.spec_from_file_location(
    "indie_norm", "/home/nsingh/normalize_indie_fixed.py"
)
indie_norm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(indie_norm)  # runs the file WITHOUT triggering main()

INPUT_FILE = "/home/nsingh/indie_converted_for_normalization_fixed.json"
OUTPUT_FILE = "/home/nsingh/normalization_results_indie_benchie_only.json"


def normalize_predicate_readonly(predicate, model, catalog_vecs, catalog_uris, local_cache):
    """Same logic as the original normalize_predicate, but NEVER calls
    save_cache() -- new entries only ever live in this script's own
    local_cache dict, never written to the shared file on disk."""
    if predicate in local_cache:
        return local_cache[predicate]
    top_uris = indie_norm.get_top_k(predicate, model, catalog_vecs, catalog_uris)
    result = indie_norm.ask_llm(predicate, top_uris)
    local_cache[predicate] = result
    return result


def main():
    print("Loading model (shared logic from normalize_indie_fixed.py)...")
    model = indie_norm.SentenceTransformer(indie_norm.FINETUNED_MODEL, trust_remote_code=True)
    print("Loading catalog embeddings...")
    catalog_vecs, catalog_uris = indie_norm.load_catalog_and_embeddings(model)

    print("Loading cache READ-ONLY (will never write back to this file)...")
    shared_cache = indie_norm.load_cache()
    local_cache = dict(shared_cache)  # local copy -- safe to extend, never saved back
    print(f"  Loaded {len(shared_cache)} existing entries as a read-only snapshot.")

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    benchie_results = data["benchie"]
    print(f"\nProcessing BenchIE only: {len(benchie_results)} samples...")

    precisions, recalls, f1s = [], [], []
    for idx, r in enumerate(benchie_results):
        pred_triples = indie_norm.parse_triples(r["predicted"])
        gold_triples = indie_norm.parse_triples(r["reference"])

        pred_norm = []
        for s, p, o in pred_triples:
            if indie_norm.is_property_relation(p):
                pred_norm.append((s, "property", o))
            else:
                dbo = normalize_predicate_readonly(p, model, catalog_vecs, catalog_uris, local_cache)
                pred_norm.append((s, dbo, o))

        gold_norm = []
        for s, p, o in gold_triples:
            if indie_norm.is_property_relation(p):
                gold_norm.append((s, "property", o))
            else:
                dbo = normalize_predicate_readonly(p, model, catalog_vecs, catalog_uris, local_cache)
                gold_norm.append((s, dbo, o))

        p_score, r_score, f1 = indie_norm.score_normalized(pred_norm, gold_norm)
        precisions.append(p_score)
        recalls.append(r_score)
        f1s.append(f1)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(benchie_results):
            print(f"  [{idx+1}/{len(benchie_results)}] avg_f1={sum(f1s)/len(f1s):.3f}", flush=True)

    result = {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
        "n": len(benchie_results),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}\nBENCHIE FULL-PIPELINE RESULT (IndIE)\n{'='*60}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall:    {result['recall']:.3f}")
    print(f"F1:        {result['f1']:.3f}")
    print(f"N:         {result['n']}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("NOTE: shared cache file was NOT modified by this script.")


if __name__ == "__main__":
    main()
