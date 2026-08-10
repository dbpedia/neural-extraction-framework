#!/usr/bin/env python3
"""Repair sf:* tiers to max-tier-wins semantics.

Why: build_index.py uses HSETNX (never overwrite). A --limit trial run that
loads a *subset* of redirects/disambiguations before the full labels pass can
therefore permanently pin a candidate at tier 0.9/0.8 that the full labels
load should own at tier 1.0 (e.g. sf:australia -> Australia stuck at 0.8).

Fix: re-stream labels (tier 1.0) then redirects (tier 0.9); wherever the
stored tier for that (surface, candidate) pair is LOWER, upgrade the tier and
recompute  final = tier * (1 + log10(1 + indegree))  from pop:*.
Disambiguations (0.8) can never upgrade anything, so they are not streamed.

Idempotent: upgrades only ever raise a tier to the streamed value; a second
run finds nothing left to raise.
"""
import argparse
import math

import redis
from tqdm import tqdm

from build_index import find_dump, open_dump, parse_label, parse_redirect

REDIS_PORT = 6380
BATCH = 5_000


def parse_stored(value: str) -> float:
    return float(value.split("|", 1)[0]) if "|" in value else float(value)


def flush(r: redis.Redis, pairs: list[tuple[str, str]], tier: float) -> int:
    """Check one batch of (surface, candidate) pairs; upgrade lower tiers."""
    pipe = r.pipeline(transaction=False)
    for surface, cand in pairs:
        pipe.hget(f"sf:{surface}", cand)
    stored = pipe.execute()

    upgrades = [(s, c) for (s, c), v in zip(pairs, stored)
                if v is not None and parse_stored(v) < tier]
    if not upgrades:
        return 0
    pops = r.mget([f"pop:{c}" for _, c in upgrades])
    pipe = r.pipeline(transaction=False)
    for (surface, cand), pop in zip(upgrades, pops):
        indeg = int(pop) if pop else 0
        final = tier * (1.0 + math.log10(1.0 + indeg))
        pipe.hset(f"sf:{surface}", cand, f"{tier:g}|{final:.4f}")
    pipe.execute()
    return len(upgrades)


def repair(r: redis.Redis, stem: str, parse, tier: float, limit: int | None) -> tuple[int, int]:
    path = find_dump(stem)
    upgraded = lines = 0
    pairs: list[tuple[str, str]] = []
    with open_dump(path) as f:
        for line in tqdm(f, desc=f"repair {stem} (tier {tier:g})", unit=" lines", unit_scale=True):
            lines += 1
            if lines % 1_000_000 == 0:
                tqdm.write(f"  {lines:,} lines, {upgraded:,} upgrades so far")
            parsed = parse(line)
            if parsed and parsed[0] and parsed[1]:
                pairs.append(parsed)
                if len(pairs) >= BATCH:
                    upgraded += flush(r, pairs, tier)
                    pairs = []
            if limit is not None and lines >= limit:
                break
    if pairs:
        upgraded += flush(r, pairs, tier)
    return lines, upgraded


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only process the first N lines of each file (for testing)")
    args = ap.parse_args()

    r = redis.Redis(port=REDIS_PORT, decode_responses=True)
    r.ping()
    for stem, parse, tier in [("labels_lang=en", parse_label, 1.0),
                              ("redirects_lang=en", parse_redirect, 0.9)]:
        lines, upgraded = repair(r, stem, parse, tier, args.limit)
        print(f"{stem}: {lines:,} lines, {upgraded:,} tiers upgraded to {tier:g}")


if __name__ == "__main__":
    main()
