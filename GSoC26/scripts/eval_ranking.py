#!/usr/bin/env python3
"""Text2KGBench ranking eval with swappable sort keys.

Default: evaluate the production sort (A). --compare evaluates all schemes on
the same candidate pools and prints one table plus A-vs-B disagreements.

Sort schemes (higher = better in every component):
  A  (tier, final)        strict tier dominance, popularity breaks ties (production)
  B  (final/tier,)        popularity only — tiers ignored
  C  (final,)             blended: final already equals tier*(1+log10(1+indegree))
  D  (merged, final)      labels+redirects share one tier, disambiguations below

Denominator for every column: all non-literal entity mentions.
"""
import argparse
import json
import os

from surface_index import candidates

BENCH = os.path.expanduser("~/Text2KGBench/data/dbpedia_webnlg/ground_truth")
DOMAINS = ["4_building", "10_comicscharacter", "13_food", "1_university"]
K = 15

SCHEMES = {
    "A": lambda uri, tier, final: (-tier, -final, uri),
    "B": lambda uri, tier, final: (-final / tier, uri),
    "C": lambda uri, tier, final: (-final, uri),
    "D": lambda uri, tier, final: (-(1.0 if tier >= 0.9 else tier), -final, uri),
}


def ranked(pool: list, scheme: str) -> list[str]:
    key = SCHEMES[scheme]
    return [uri for uri, _, _ in sorted(pool, key=lambda c: key(*c))[:K]]


def mentions(domain: str):
    with open(f"{BENCH}/ont_{domain}_ground_truth.jsonl") as f:
        for line in f:
            for t in json.loads(line)["triples"]:
                for e in (t["sub"], t["obj"]):
                    if not e.startswith('"'):
                        yield e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", action="store_true",
                    help="evaluate all sort schemes, not just the production one")
    ap.add_argument("--examples", type=int, default=3,
                    help="number of A-vs-B disagreement examples to show (with --compare)")
    args = ap.parse_args()
    schemes = list(SCHEMES) if args.compare else ["A"]

    # stats[scheme][domain] = [total, r15, r3, r1]
    stats = {s: {d: [0, 0, 0, 0] for d in DOMAINS} for s in schemes}
    disagreements = []

    for domain in DOMAINS:
        for e in mentions(domain):
            pool = candidates(e.replace("_", " "))
            tops = {}
            for s in schemes:
                st = stats[s][domain]
                st[0] += 1
                uris = ranked(pool, s)
                tops[s] = uris[:1]
                if e in uris:
                    st[1] += 1
                    rank = uris.index(e)
                    if rank < 3:
                        st[2] += 1
                    if rank == 0:
                        st[3] += 1
            if args.compare and tops["A"] != tops["B"]:
                disagreements.append((e, pool))

    print(f"{'sort':<5} {'domain':<22} {'recall@15':>10} {'rank@1':>8} {'rank@3':>8}")
    for s in schemes:
        T = R15 = R3 = R1 = 0
        for domain in DOMAINS:
            total, r15, r3, r1 = stats[s][domain]
            T += total; R15 += r15; R3 += r3; R1 += r1
            print(f"{s:<5} {domain:<22} {100*r15/total:>9.1f}% {100*r1/total:>7.1f}% {100*r3/total:>7.1f}%")
        print(f"{s:<5} {'OVERALL':<22} {100*R15/T:>9.1f}% {100*R1/T:>7.1f}% {100*R3/T:>7.1f}%")
        print()

    if args.compare and disagreements:
        seen = set()
        shown = 0
        print(f"=== A-vs-B disagreements (of {len(disagreements)} mention occurrences) ===")
        for e, pool in disagreements:
            if e in seen or shown >= args.examples:
                continue
            seen.add(e)
            shown += 1
            a, b = ranked(pool, "A")[:3], ranked(pool, "B")[:3]
            info = {uri: (tier, final) for uri, tier, final in pool}
            print(f"\nmention {e.replace('_', ' ')!r}   (gold: {e})")
            for label, uris in (("A", a), ("B", b)):
                row = ", ".join(f"{u} [tier {info[u][0]:g}, final {info[u][1]:.2f}]" for u in uris)
                print(f"  {label} top-3: {row}")


if __name__ == "__main__":
    main()
