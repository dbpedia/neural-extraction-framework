"""
================================================================================
Text2KGBench Runner — v13 multi-triple pipeline vs the official metric
================================================================================
Run AFTER autonomous_pipeline_v13.py in the same session (needs extract_all_triples).
Also needs text2kg_harness.py alongside it.

    export TEXT2KG_DIR=/path/to/Text2KGBench/data/dbpedia_webnlg
    python3 run_text2kg.py 3_airport          # one domain (79 sentences) — START HERE
    python3 run_text2kg.py 3_airport 10       # first 10 sentences — smoke test
    python3 run_text2kg.py all                # all 19 domains, 2014 sentences (hours)

--------------------------------------------------------------------------------
WHAT NUMBER TO EXPECT
--------------------------------------------------------------------------------
Published on this exact benchmark and metric:
    REBEL zero-shot        0.060
    T5-Large fine-tuned    0.389
    GPT-3.5 5-shot         0.510
    GPT-4o 6-shot          0.570
    NEF (paper)            0.628

Expect ~0.35-0.45 on the first honest run. NOT 0.93 — that was Entity-Pair F1
on 15 hand-picked sentences. This is EXACT triple match on 2,014. Different
metric, different set, not comparable. Reasons the gap is real and expected:
    - LLaMA-3.3-70B here vs GPT-4o in the paper
    - 1,105-property OWL snapshot vs their 2,889 + FAISS
    - fixed few-shots vs their per-sentence BM25+MMR retrieval
    - DBpedia Lookup API vs their Wikipedia anchor-count surface index
================================================================================
"""

import sys
import os
import json
import time

# ── IMPORT: works as a FILE or as a pasted NOTEBOOK CELL ─────────────────────
# If you pasted the harness into a cell, it is NOT an importable module — its
# functions are already in your namespace. So: try the import, and if it fails,
# check whether the names are simply already defined before giving up.
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_NEEDED = ["load_ontology", "load_test", "load_ground_truth", "all_domains",
           "score_sentence", "macro_average"]
try:
    from text2kg_harness import (            # noqa: F401
        load_ontology, load_test, load_ground_truth, all_domains,
        score_sentence, macro_average,
    )
except ModuleNotFoundError:
    _missing = [n for n in _NEEDED if n not in dir()]
    if _missing:
        raise ModuleNotFoundError(
            "Cannot find the harness.\n"
            f"  Looked for text2kg_harness.py in: {_HERE}\n"
            f"  Also not already in the namespace (missing: {_missing})\n\n"
            "  FIX — either:\n"
            "    (a) save it as a file:   %%writefile text2kg_harness.py\n"
            "        ...paste harness contents...   then re-run this cell, OR\n"
            "    (b) just run the harness cell FIRST — its functions land in the\n"
            "        namespace and this script will use them directly."
        )
    print("📎 Using harness functions already defined in the namespace.")


def run_domain(slug, limit=None, verbose=False):
    """Run every test sentence in one domain. Returns (P, R, F1, rows)."""
    onto = load_ontology(slug)
    tests = load_test(slug)
    gold = load_ground_truth(slug)
    if limit:
        tests = tests[:limit]

    allowed = onto["pids"]
    # Predicates whose declared range is a literal type. Measured across all 19
    # ontologies, the literal ranges are exactly: number, string, Date, Year, date.
    # Everything else (Person, Place, City, Country, ...) is an entity type. These
    # skip entity-linking and register fly-embedded pids as datatype in Node 3.
    _LITERAL_RANGES = {"number", "string", "date", "year"}
    literal_preds = {pid for pid, r in onto["relations"].items()
                     if str(r.get("range", "")).lower() in _LITERAL_RANGES}
    # pid -> declared range, so the literal normalizer can format EXACTLY like gold
    # (number: strip commas; Year/string codes: PRESERVE leading zeros like 01325;
    #  string with comma/space: quote it, e.g. "DL1, DL2, DL3").
    pred_ranges = {pid: str(r.get("range", "")) for pid, r in onto["relations"].items()}
    # Quoting convention learned from the TRAIN split only (no test leakage).
    # Tolerate the harness being pasted as a CELL rather than saved to disk:
    # use the namespace function if the file import fails.
    try:
        from text2kg_harness import learn_quote_conventions
    except ImportError:
        learn_quote_conventions = globals().get("learn_quote_conventions",
                                                lambda _slug: {})
    quote_hints = learn_quote_conventions(slug)
    print(f"\n{'#' * 78}")
    print(f"# DOMAIN {slug}   {len(tests)} sentences   {len(allowed)} allowed predicates")
    print(f"{'#' * 78}")

    ps, rs, fs, rows = [], [], [], []
    t0 = time.time()

    for i, item in enumerate(tests, 1):
        sid, sent = item["id"], item["sent"]
        gold_triples = gold.get(sid, [])

        try:
            out = extract_all_triples(sent, allowed_predicates=allowed,
                                      literal_predicates=literal_preds,
                                      predicate_ranges=pred_ranges,
                                      quote_hints=quote_hints, verbose=verbose)
            pred_triples = out["triples"]
        except Exception as e:
            print(f"   🔴 {sid} crashed: {str(e)[:100]}")
            pred_triples = []

        p, r, f = score_sentence(pred_triples, gold_triples)
        ps.append(p); rs.append(r); fs.append(f)
        rows.append({"id": sid, "sent": sent, "gold": gold_triples,
                     "pred": pred_triples, "p": p, "r": r, "f1": f})

        elapsed = time.time() - t0
        rate = elapsed / i
        eta = rate * (len(tests) - i)
        print(f"[{i:3d}/{len(tests)}] F1={f:.2f}  pred={len(pred_triples)} gold={len(gold_triples)}  "
              f"({rate:.1f}s/sent, ETA {eta/60:.1f}m)  {sent[:52]}")

    dp = sum(ps) / len(ps) if ps else 0.0
    dr = sum(rs) / len(rs) if rs else 0.0
    df = sum(fs) / len(fs) if fs else 0.0
    print(f"\n>>> {slug}:  P={dp:.4f}  R={dr:.4f}  F1={df:.4f}   ({time.time()-t0:.0f}s)")
    return dp, dr, df, rows


def error_analysis(rows):
    """Where the score is actually going. Read this before tuning anything."""
    n_pred = sum(len(r["pred"]) for r in rows)
    n_gold = sum(len(r["gold"]) for r in rows)
    perfect = sum(1 for r in rows if r["f1"] == 1.0)
    zero = sum(1 for r in rows if r["f1"] == 0.0)

    # Of the gold triples we MISSED, how many did we not even attempt?
    under = sum(max(0, len(r["gold"]) - len(r["pred"])) for r in rows)

    # How often is the ENTITY PAIR right but the PREDICATE wrong? That is a
    # predicate-linking problem, not an entity-linking problem — different fix.
    pair_ok_rel_bad = 0
    for r in rows:
        gpairs = {(t["sub"].lower(), t["obj"].lower()) for t in r["gold"]}
        for t in r["pred"]:
            if (t["sub"].lower(), t["obj"].lower()) in gpairs:
                keys = {(x["sub"].lower(), x["rel"].lower(), x["obj"].lower()) for x in r["gold"]}
                if (t["sub"].lower(), t["rel"].lower(), t["obj"].lower()) not in keys:
                    pair_ok_rel_bad += 1

    print("\n" + "=" * 78)
    print("ERROR ANALYSIS")
    print("=" * 78)
    print(f"  sentences            : {len(rows)}")
    print(f"  gold triples         : {n_gold}   ({n_gold/len(rows):.2f} per sentence)")
    print(f"  predicted triples    : {n_pred}   ({n_pred/len(rows):.2f} per sentence)")
    print(f"  UNDER-EXTRACTED      : {under} gold triples we never even attempted")
    print(f"  perfect sentences    : {perfect} ({100*perfect/len(rows):.1f}%)")
    print(f"  zero-score sentences : {zero} ({100*zero/len(rows):.1f}%)")
    print(f"  entity pair RIGHT but predicate WRONG: {pair_ok_rel_bad}")
    print("     ^ these are PREDICATE-LINKING misses, not entity misses.")
    if n_pred < n_gold * 0.8:
        print(f"\n  ⚠️  Emitting {n_pred/len(rows):.2f} triples/sentence against "
              f"{n_gold/len(rows):.2f} gold.")
        print("      Recall is capped by under-extraction — fix Node 1 before touching Node 3.")


def show_examples(rows, k=3):
    print("\n" + "=" * 78)
    print("SAMPLE ROWS")
    print("=" * 78)
    for r in rows[:k]:
        print(f"\n[{r['id']}] F1={r['f1']:.2f}")
        print(f"  {r['sent']}")
        print(f"  GOLD ({len(r['gold'])}):")
        for t in r["gold"]:
            print(f"     ({t['sub']}, {t['rel']}, {t['obj']})")
        print(f"  PRED ({len(r['pred'])}):")
        for t in r["pred"]:
            print(f"     ({t['sub']}, {t['rel']}, {t['obj']})")


def _in_notebook():
    """True inside Jupyter/IPython, where sys.argv belongs to the KERNEL."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def main(target=None, limit=None):
    """
    Run the benchmark.

    NOTEBOOK — call it directly, do NOT rely on sys.argv:
        main("3_airport", 10)      # smoke test
        main("3_airport")          # full domain, 79 sentences
        main("all")                # all 19, ~2014 sentences (hours)

    Or skip main() entirely and use the pieces:
        p, r, f, rows = run_domain("3_airport", limit=10)
        error_analysis(rows); show_examples(rows)

    TERMINAL:
        python3 run_text2kg.py 3_airport 10

    Why the guard: in Jupyter, sys.argv is the KERNEL's launch args —
        ['ipykernel_launcher', '-f', '/root/.../kernel-07b3....json']
    so sys.argv[2] is a JSON path and int() on it raises
        ValueError: invalid literal for int() with base 10: '/root/...json'
    """
    # Only read the command line when we are ACTUALLY on a command line.
    if target is None and not _in_notebook() and len(sys.argv) > 1:
        target = sys.argv[1]
        if len(sys.argv) > 2:
            try:
                limit = int(sys.argv[2])
            except ValueError:
                limit = None

    if not target:
        print(__doc__)
        print("\nUsage:")
        print('   main("3_airport", 10)    <- notebook: smoke test, 10 sentences')
        print('   main("3_airport")        <- notebook: full domain')
        print('   main("all")              <- notebook: all 19 domains')
        print("\nAvailable domains:")
        for d in all_domains():
            print(f"   {d}")
        return None

    slugs = all_domains() if target == "all" else [target]
    per_domain, all_rows = [], []

    for slug in slugs:
        p, r, f, rows = run_domain(slug, limit=limit)
        per_domain.append((p, r, f))
        all_rows.extend(rows)
        try:
            with open(f"results_{slug}.json", "w") as fh:
                json.dump(rows, fh, indent=2)
        except Exception as e:
            print(f"   (could not save results_{slug}.json: {e})")

    mp, mr, mf = macro_average(per_domain)
    print("\n" + "=" * 78)
    print(f"MACRO AVERAGE over {len(slugs)} domain(s)")
    print("=" * 78)
    print(f"  Precision : {mp:.4f}")
    print(f"  Recall    : {mr:.4f}")
    print(f"  F1        : {mf:.4f}")
    print("\n  Published on this benchmark (exact-triple macro F1, 19 domains):")
    print("     REBEL zero-shot     0.060")
    print("     T5-Large            0.389")
    print("     GPT-3.5 5-shot      0.510")
    print("     GPT-4o 6-shot       0.570")
    print("     NEF (paper)         0.628")
    if len(slugs) < 19:
        print(f"\n  ⚠️  {len(slugs)} domain(s), not 19 — NOT comparable to the table above.")
        print("      Per-domain F1 ranges 0.37-0.77 in the paper, so one domain can")
        print("      mislead by ±20 points. This is a debugging signal, not a result.")

    error_analysis(all_rows)
    show_examples(all_rows)
    return mp, mr, mf, all_rows


# Only auto-run from a real command line. In Jupyter this is a no-op, so pasting
# the whole file into a cell just defines the functions.
if __name__ == "__main__" and not _in_notebook() and len(sys.argv) > 1 \
        and not sys.argv[1].startswith("-"):
    main()
