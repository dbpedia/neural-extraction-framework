#!/usr/bin/env python3
"""Import an sf_index_export.ndjson.gz produced by export_index.py into Redis.

Safe on a shared instance: writes only sf:* keys, never flushes anything.
Idempotent: HSET with identical values; re-running converges to the same state.
Verifies the imported key count against the export's trailing _count record.
"""
import argparse
import gzip
import json
import sys

import redis
from tqdm import tqdm

BATCH = 10_000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="path to sf_index_export.ndjson.gz")
    ap.add_argument("--port", type=int, default=6380)
    args = ap.parse_args()

    r = redis.Redis(port=args.port, decode_responses=True)
    r.ping()

    expected = expected_rd = None
    imported = rd_imported = 0
    pipe = r.pipeline(transaction=False)
    pending = 0

    with gzip.open(args.dump, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="importing sf:*/rd:*", unit=" keys", unit_scale=True):
            rec = json.loads(line)
            if "_meta" in rec:
                print(f"dump metadata: {rec['_meta']}")
                continue
            if "_count" in rec:
                expected = rec["_count"]
                expected_rd = rec.get("_rd_count")
                continue
            if "s" in rec:
                pipe.hset(f"sf:{rec['s']}", mapping=rec["c"])
                imported += 1
            elif "k" in rec:
                pipe.set(f"rd:{rec['k']}", rec["v"])
                rd_imported += 1
            pending += 1
            if pending >= BATCH:
                pipe.execute()
                pending = 0
            if imported % 1_000_000 == 0 and imported:
                tqdm.write(f"  {imported:,} sf keys imported")
    if pending:
        pipe.execute()

    print(f"\nimported {imported:,} sf:* keys + {rd_imported:,} rd:* keys")
    if expected is None:
        sys.exit("WARNING: dump had no trailing _count record — file may be truncated")
    if imported != expected:
        sys.exit(f"MISMATCH: dump says {expected:,} sf keys, imported {imported:,}")
    if expected_rd is not None and rd_imported != expected_rd:
        sys.exit(f"MISMATCH: dump says {expected_rd:,} rd keys, imported {rd_imported:,}")
    print(f"counts match dump record ({expected:,} sf / {expected_rd or 0:,} rd) ✓")


if __name__ == "__main__":
    main()
