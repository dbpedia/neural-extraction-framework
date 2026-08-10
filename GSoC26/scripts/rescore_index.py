#!/usr/bin/env python3
"""Rescore every sf:* candidate with the in-degree popularity prior.

New stored value format (per candidate field in each sf:* hash):

    <tier>|<final>      e.g.  "0.9|5.8974"

where  final = tier * (1 + log10(1 + indegree))  and tier is the original
load-tier weight (1.0 labels / 0.9 redirects / 0.8 disambiguations).

The tier is stored alongside the final score so that:
  - ranking can keep tier DOMINANT (lookup sorts by tier first, then final —
    in-degree only breaks ties within a tier), and
  - re-running is idempotent: the tier is re-read from the stored value, never
    re-derived from an already-multiplied score, so scores never compound.

Candidates with no pop:* entry get in-degree 0 (final = tier exactly).
"""
import argparse
import math

import redis
from tqdm import tqdm

REDIS_PORT = 6380
KEY_BATCH = 2_000          # sf:* keys fetched/written per pipeline round-trip
RAW_TIERS = {"1.0", "0.9", "0.8"}


def parse_tier(value: str) -> float:
    """Original tier weight from either a raw build_index value or a rescored one."""
    if "|" in value:
        return float(value.split("|", 1)[0])
    return float(value)


def rescore_batch(r: redis.Redis, keys: list[str]) -> tuple[int, int]:
    """Rescore one batch of sf:* keys. Returns (keys_written, candidates_written)."""
    pipe = r.pipeline(transaction=False)
    for key in keys:
        pipe.hgetall(key)
    hashes = pipe.execute()

    # one pop: fetch per distinct candidate in the batch
    names = {name for h in hashes for name in h}
    order = list(names)
    pops = r.mget([f"pop:{n}" for n in order]) if order else []
    indegree = {n: int(v) if v else 0 for n, v in zip(order, pops)}

    pipe = r.pipeline(transaction=False)
    keys_written = cands_written = 0
    for key, h in zip(keys, hashes):
        if not h:
            continue
        mapping = {}
        for name, value in h.items():
            tier = parse_tier(value)
            final = tier * (1.0 + math.log10(1.0 + indegree[name]))
            mapping[name] = f"{tier:g}|{final:.4f}"
        pipe.hset(key, mapping=mapping)
        keys_written += 1
        cands_written += len(mapping)
    pipe.execute()
    return keys_written, cands_written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only rescore the first N sf:* keys (for testing)")
    args = ap.parse_args()

    r = redis.Redis(port=REDIS_PORT, decode_responses=True)
    r.ping()

    keys_done = cands_done = 0
    batch: list[str] = []
    bar = tqdm(desc="rescoring sf:*", unit=" keys", unit_scale=True)
    for key in r.scan_iter(match="sf:*", count=10_000):
        batch.append(key)
        if len(batch) >= KEY_BATCH:
            kw, cw = rescore_batch(r, batch)
            keys_done += kw
            cands_done += cw
            bar.update(len(batch))
            if keys_done // 1_000_000 != (keys_done - kw) // 1_000_000:
                tqdm.write(f"  {keys_done:,} keys rescored")
            batch = []
        if args.limit is not None and keys_done + len(batch) >= args.limit:
            break
    if batch:
        kw, cw = rescore_batch(r, batch)
        keys_done += kw
        cands_done += cw
        bar.update(len(batch))
    bar.close()

    print(f"\nrescored {keys_done:,} sf:* keys / {cands_done:,} candidates")


if __name__ == "__main__":
    main()
