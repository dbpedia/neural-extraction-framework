#!/usr/bin/env python3
"""Export the sf:* surface-form index to a single compressed file for transfer.

Format: gzipped newline-delimited JSON (portable, inspectable with zcat):
    line 1:     {"_meta": {"format": "dbpedia-sf-index", "version": 2, ...}}
    sf records: {"s": "<surface form>", "c": {"<local name>": "<tier|final>", ...}}
    rd records: {"k": "<comma-alias local name>", "v": "<canonical target>"}
    last line:  {"_count": <sf keys>, "_rd_count": <rd keys>}

rd:* keys are the comma-compound redirect aliases used by the pipeline's
CANONICALIZE_COMMA_COMPOUNDS emission rule.

Import with import_index.py on the target machine.
"""
import argparse
import gzip
import json
import os

import redis
from tqdm import tqdm

BATCH = 5_000
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sf_index_export.ndjson.gz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=6380)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    r = redis.Redis(port=args.port, decode_responses=True)
    r.ping()

    exported = rd_exported = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        meta = {"_meta": {"format": "dbpedia-sf-index", "version": 2,
                          "release": "2022.12.01", "value_format": "tier|final"}}
        out.write(json.dumps(meta) + "\n")

        batch: list[str] = []

        def flush() -> None:
            nonlocal exported
            pipe = r.pipeline(transaction=False)
            for key in batch:
                pipe.hgetall(key)
            for key, h in zip(batch, pipe.execute()):
                if h:
                    out.write(json.dumps({"s": key[3:], "c": h}, ensure_ascii=False) + "\n")
                    exported += 1
            batch.clear()

        for key in tqdm(r.scan_iter(match="sf:*", count=10_000),
                        desc="exporting sf:*", unit=" keys", unit_scale=True):
            batch.append(key)
            if len(batch) >= BATCH:
                flush()
                if exported % 1_000_000 < BATCH:
                    tqdm.write(f"  {exported:,} keys exported")
        flush()

        rd_batch: list[str] = []

        def flush_rd() -> None:
            nonlocal rd_exported
            vals = r.mget(rd_batch)
            for key, val in zip(rd_batch, vals):
                if val is not None:
                    out.write(json.dumps({"k": key[3:], "v": val}, ensure_ascii=False) + "\n")
                    rd_exported += 1
            rd_batch.clear()

        for key in tqdm(r.scan_iter(match="rd:*", count=10_000),
                        desc="exporting rd:*", unit=" keys", unit_scale=True):
            rd_batch.append(key)
            if len(rd_batch) >= BATCH:
                flush_rd()
        if rd_batch:
            flush_rd()

        out.write(json.dumps({"_count": exported, "_rd_count": rd_exported}) + "\n")

    size = os.path.getsize(args.out)
    print(f"\nexported {exported:,} sf:* keys + {rd_exported:,} rd:* keys -> {args.out}")
    print(f"file size: {size:,} bytes ({size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
