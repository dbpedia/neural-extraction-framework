#!/usr/bin/env python3
"""Offline audit of qualifier-suffix mismatches in the frozen v14-final sweep.

Counts gold triples that the pipeline missed ONLY because a URI was emitted
without its parenthesised disambiguator (or with a spurious one), and splits
them by whether the disambiguator word actually appears in the source sentence
-- i.e. whether it is recoverable from context at all. It then replays
src/autonomous_pipeline_v13.py:_context_variants over the same sentences to
show which of those cases variant generation now reaches, and which variants
sit behind the spurious ones.

No network, no LLM, no Redis. Reads results/final_clean/ only. The frozen
files do not store the mention the LLM extracted, so the entity's base title
("Ardmore_Airport" -> "Ardmore Airport") stands in for it, and "reached" means
the variant is GENERATED -- whether the index returns it needs a real run.

Run from GSoC26/:  python3 scripts/analyze_qualifier_gap.py
                   python3 scripts/analyze_qualifier_gap.py --dump-variants out.txt
"""
import ast
import glob
import json
import os
import re
import sys
from collections import Counter

RESULTS = "results/final_clean"
PIPELINE = "src/autonomous_pipeline_v13.py"


def normalize_triple(sub, rel, obj):
    """Verbatim port of the official Text2KGBench run_eval.py normalisation,
    matching src/text2kg_harness.py:normalize_triple."""
    f = lambda x: re.sub(r"(_|\s+)", "", str(x)).lower()
    return f(sub) + f(rel) + f(obj)


def as_set(triples):
    return {normalize_triple(t["sub"], t["rel"], t["obj"]) for t in triples}


def f1(gold, pred):
    """Per-sentence F1, as src/text2kg_harness.py:calculate_precision_recall_f1."""
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load_context_variants():
    """_context_variants and its constants, exec'd straight from the pipeline
    source so this audits the real function without loading the model stack."""
    want = {"_QUAL_KIND_TRIGGERS", "_QUAL_STOPWORDS", "_context_variants"}
    ns = {"re": re}
    for node in ast.parse(open(PIPELINE, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            names = {getattr(node, "name", None)}
        if names & want:
            exec(compile(ast.Module([node], []), PIPELINE, "exec"), ns)
    return ns["_context_variants"], set(ns["_QUAL_KIND_TRIGGERS"])


QUAL_RE = re.compile(r"^(.*)_\(([^)]*)\)$")


def surface(uri_local_name):
    return str(uri_local_name).replace("_", " ")


def main():
    context_variants, kinds = load_context_variants()

    def generated(mention, sentence, target):
        """The variant equal to `target` (the index lowercases keys), or None."""
        for v in context_variants(mention, sentence):
            if v.lower() == target.lower():
                return v
        return None

    def is_bare_name(variant):
        """True for "<m> (<name>)", False for the kind forms "<m> (... <kind>)"."""
        return QUAL_RE.match(variant.replace(" ", "_")).group(2).split("_")[-1] not in kinds

    missing = []   # gold wants _(qualifier), pred emitted bare
    spurious = []  # pred emitted _(qualifier), gold wants bare
    domains = {}   # domain -> rows
    exposure = set()  # bare-name variants emitted for entities already correct

    files = sorted(f for f in glob.glob(os.path.join(RESULTS, "results_*.json"))
                   if "__" not in os.path.basename(f))
    if not files:
        raise SystemExit(f"no result files under {RESULTS}/ -- run from GSoC26/")

    for path in files:
        domain = os.path.basename(path)[len("results_"):-len(".json")]
        domains[domain] = json.load(open(path, encoding="utf-8"))
        for idx, row in enumerate(domains[domain]):
            gold_t, pred_t = row.get("gold", []), row.get("pred", [])
            gold, pred = as_set(gold_t), as_set(pred_t)
            sentence = row.get("sent", "")

            # gold carries a qualifier the prediction dropped
            for gt in gold_t:
                if normalize_triple(gt["sub"], gt["rel"], gt["obj"]) in pred:
                    continue
                for field in ("sub", "obj"):
                    m = QUAL_RE.match(str(gt[field]))
                    if not m:
                        continue
                    base, qual = m.group(1), m.group(2)
                    stripped = dict(gt, **{field: base})
                    if normalize_triple(stripped["sub"], stripped["rel"],
                                        stripped["obj"]) in pred:
                        in_sentence = re.search(
                            rf"\b{re.escape(qual.replace('_', ' '))}\b",
                            sentence, re.I) is not None
                        reached = generated(surface(base), sentence,
                                            surface(gt[field])) is not None
                        missing.append((domain, idx, field, base, gt[field],
                                        qual, gt["rel"], in_sentence, reached))

            for pt in pred_t:
                correct = normalize_triple(pt["sub"], pt["rel"], pt["obj"]) in gold
                for field in ("sub", "obj"):
                    m = QUAL_RE.match(str(pt[field]))
                    if correct:
                        # an entity the frozen run already resolves: every
                        # bare-name variant for it is a chance to regress
                        base = m.group(1) if m else pt[field]
                        exposure.update(
                            v for v in context_variants(surface(base), sentence)
                            if is_bare_name(v))
                    if correct or not m:
                        continue
                    # prediction invented a qualifier gold does not have
                    stripped = dict(pt, **{field: m.group(1)})
                    if normalize_triple(stripped["sub"], stripped["rel"],
                                        stripped["obj"]) in gold:
                        via = generated(surface(m.group(1)), sentence,
                                        surface(pt[field]))
                        spurious.append((domain, idx, field, pt[field],
                                         m.group(1), pt["rel"], via))

    print(f"Frozen sweep: {len(files)} domains under {RESULTS}/\n")

    print(f"A. GOLD-QUALIFIED, PRED-BARE  ->  {len(missing)} gold triples missed")
    recoverable = [c for c in missing if c[7]]
    reached = [c for c in missing if c[8]]
    print(f"   of which the qualifier word appears in the sentence: "
          f"{len(recoverable)}")
    print(f"   of which _context_variants generates the gold form:  "
          f"{len(reached)}\n")

    by_domain = Counter(c[0] for c in missing)
    print("   by domain:")
    for dom, n in by_domain.most_common():
        rec = sum(1 for c in missing if c[0] == dom and c[7])
        gen = sum(1 for c in missing if c[0] == dom and c[8])
        print(f"     {dom:26} {n:3}   recoverable from sentence: {rec:2}   "
              f"generated: {gen}")

    print("\n   by qualifier token:")
    by_qual = Counter(c[5] for c in missing)
    for qual, n in by_qual.most_common():
        rec = sum(1 for c in missing if c[5] == qual and c[7])
        gen = sum(1 for c in missing if c[5] == qual and c[8])
        print(f"     _({qual}){'':<{max(0, 24 - len(qual))}} {n:3}   "
              f"recoverable: {rec:2}   generated: {gen}")

    print(f"\nB. PRED-QUALIFIED, GOLD-BARE  ->  {len(spurious)} triples lost to a "
          f"spurious suffix")
    for dom, n in Counter(c[0] for c in spurious).most_common():
        print(f"     {dom:26} {n:3}")
    print("   variant behind each one:")
    for c in spurious:
        how = ("not generated by _context_variants" if c[6] is None else
               f"'{c[6]}' ({'bare-name' if is_bare_name(c[6]) else 'kind'} variant)")
        print(f"     {c[0]:20} #{c[1]:<4} {c[3]:30} <- {how}")

    # Oracle rescore: resolve the entity to gold's qualified form in every
    # triple of the sentence, for a chosen subset of the cases in A.
    def rescore(cases):
        fix = {}
        for c in cases:
            fix.setdefault((c[0], c[1]), {})[normalize_triple(c[3], "", "")] = c[4]
        out = {}
        for dom, rows in domains.items():
            scores = []
            for idx, row in enumerate(rows):
                swap = fix.get((dom, idx), {})
                pred = [{k: swap.get(normalize_triple(v, "", ""), v) if k != "rel" else v
                         for k, v in t.items()} for t in row.get("pred", [])]
                scores.append(f1(as_set(row.get("gold", [])), as_set(pred)))
            out[dom] = sum(scores) / len(scores)
        return out

    frozen, ceil, gen = rescore([]), rescore(recoverable), rescore(reached)
    macro = lambda d: sum(d.values()) / len(d)
    print("\nC. ORACLE RESCORE (assumes the index returns the variant and Node 3 "
          "picks it)")
    print(f"     {'domain':26} {'frozen':>7} {'ceiling':>8} {'generated':>10}")
    for dom in sorted(frozen, key=lambda d: frozen[d] - ceil[d]):
        if ceil[dom] != frozen[dom]:
            print(f"     {dom:26} {frozen[dom]:7.4f} {ceil[dom]:8.4f} {gen[dom]:10.4f}")
    print(f"     {'MACRO':26} {macro(frozen):7.4f} {macro(ceil):8.4f} "
          f"{macro(gen):10.4f}")

    print(f"\nD. REGRESSION EXPOSURE  ->  {len(exposure)} distinct bare-name "
          f"variants emitted for entities the frozen run already gets right")
    print("   each is harmless unless that exact surface is a key in the index")
    if "--dump-variants" in sys.argv:
        out_path = sys.argv[sys.argv.index("--dump-variants") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(exposure)) + "\n")
        print(f"   written to {out_path}")

    print("\n--- recoverable from the sentence but still not generated (A) ---")
    for c in recoverable:
        if not c[8]:
            print(f"  {c[0]:24} #{c[1]:<4} {c[2]}: {c[3]}  ->gold->  {c[4]}   (rel={c[6]})")


if __name__ == "__main__":
    main()
