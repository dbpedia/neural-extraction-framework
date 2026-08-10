#!/usr/bin/env python3
"""Build an in-degree popularity prior from the DBpedia wikilinks dump.

Streams wikilinks_lang=en.ttl.bz2 (~800M triples — the modern Databus name for
the old page_links_en dataset, predicate dbo:wikiPageWikiLink) and counts
INBOUND links per target URI:

    key:   pop:<uri local name>    value: <int in-degree>

Memory stays bounded: counts accumulate in a local dict that is flushed to
Redis (pipelined INCRBY) whenever it reaches FLUSH_KEYS distinct entities.

Idempotent: existing pop:* keys are deleted before counting starts, so a
re-run (or a full run after a --limit trial) never double-counts.
"""
import argparse
import bz2
import glob
import gzip
import heapq
import os
import sys
from urllib.parse import unquote

import redis
from tqdm import tqdm

REDIS_PORT = 6380
FLUSH_KEYS = 1_500_000     # distinct entities buffered before a pipelined flush
PIPE_BATCH = 10_000
DUMPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
RESOURCE_PREFIX = "http://dbpedia.org/resource/"
WIKILINK_PRED = "http://dbpedia.org/ontology/wikiPageWikiLink"


def open_dump(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def find_dump() -> str:
    matches = sorted(glob.glob(os.path.join(DUMPS_DIR, "wikilinks_lang=en*.ttl*")))
    if not matches:
        sys.exit(f"no wikilinks_lang=en*.ttl* in {DUMPS_DIR} — "
                 "run: python3 download_dumps.py --artifacts wikilinks --version 2022.12.01 --no-count")
    return matches[0]


def clear_existing(r: redis.Redis) -> int:
    removed = 0
    batch = []
    for key in r.scan_iter(match="pop:*", count=PIPE_BATCH):
        batch.append(key)
        if len(batch) >= PIPE_BATCH:
            removed += len(batch)
            r.unlink(*batch)
            batch = []
    if batch:
        removed += len(batch)
        r.unlink(*batch)
    return removed


def flush_counts(r: redis.Redis, counts: dict) -> None:
    pipe = r.pipeline(transaction=False)
    pending = 0
    for name, n in counts.items():
        pipe.incrby(f"pop:{name}", n)
        pending += 1
        if pending >= PIPE_BATCH:
            pipe.execute()
            pending = 0
    if pending:
        pipe.execute()
    counts.clear()


def count_indegrees(r: redis.Redis, path: str, limit: int | None) -> int:
    counts: dict[str, int] = {}
    lines = 0
    with open_dump(path) as f:
        for line in tqdm(f, desc=os.path.basename(path), unit=" lines", unit_scale=True):
            lines += 1
            if lines % 1_000_000 == 0:
                tqdm.write(f"  {lines:,} lines processed ({len(counts):,} buffered)")
            # <subj> <wikiPageWikiLink> <obj> .  — count the OBJECT (inbound)
            parts = line.split("> <", 2)
            if len(parts) == 3 and parts[1] == WIKILINK_PRED:
                obj = parts[2]
                end = obj.find(">")
                target = obj[:end]
                if target.startswith(RESOURCE_PREFIX):
                    name = target[len(RESOURCE_PREFIX):]
                    if "%" in name:
                        name = unquote(name)
                    if name:
                        counts[name] = counts.get(name, 0) + 1
            if len(counts) >= FLUSH_KEYS:
                tqdm.write(f"  flushing {len(counts):,} buffered counts to Redis...")
                flush_counts(r, counts)
            if limit is not None and lines >= limit:
                break
    if counts:
        flush_counts(r, counts)
    return lines


def report(r: redis.Redis) -> None:
    """Count pop:* keys and report the top 20 by in-degree."""
    total = 0
    top: list[tuple[int, str]] = []   # min-heap of (indegree, name), size <= 20
    batch = []
    print("scanning pop:* keys for totals and top-20...", flush=True)

    def consume(keys: list) -> None:
        nonlocal total
        vals = r.mget(keys)
        for key, val in zip(keys, vals):
            if val is None:
                continue
            total += 1
            item = (int(val), key[4:])
            if len(top) < 20:
                heapq.heappush(top, item)
            elif item > top[0]:
                heapq.heapreplace(top, item)

    for key in r.scan_iter(match="pop:*", count=PIPE_BATCH):
        batch.append(key)
        if len(batch) >= PIPE_BATCH:
            consume(batch)
            batch = []
    if batch:
        consume(batch)

    print(f"\npop:* keys written: {total:,}")
    print("top 20 by in-degree:")
    for n, name in sorted(top, reverse=True):
        print(f"  {n:>9,}  {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only process the first N lines (for testing)")
    args = ap.parse_args()

    r = redis.Redis(port=REDIS_PORT, decode_responses=True)
    r.ping()
    path = find_dump()

    removed = clear_existing(r)
    if removed:
        print(f"cleared {removed:,} existing pop:* keys (idempotent re-run)")

    lines = count_indegrees(r, path, args.limit)
    print(f"\n{lines:,} lines read")
    report(r)


if __name__ == "__main__":
    main()
