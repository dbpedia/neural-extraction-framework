#!/usr/bin/env python3
"""Build a Redis surface-form index from DBpedia dumps.

Streams the (multi-GB when decompressed) dumps line by line — nothing is loaded
into RAM — and populates Redis on port 6380 with:

    key:   sf:<surface form, lowercased, whitespace-collapsed>
    value: HASH { <dbpedia URI local name>: <score> }

Load order (later files ADD candidates via HSETNX, never overwrite):
    1. labels_en          score 1.0   (rdfs:label literal -> subject)
    2. redirects_en       score 0.9   (alias local name  -> canonical target)
    3. disambiguations_en score 0.8   (page title sans "(disambiguation)" -> candidates)

Idempotent: HSETNX means re-running never changes existing scores, and the
sf: prefix keeps the index away from any other keys in the shared instance.
"""
import argparse
import bz2
import glob
import gzip
import os
import re
import sys
from urllib.parse import unquote

import redis
from tqdm import tqdm

REDIS_PORT = 6380
BATCH_SIZE = 10_000
DUMPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
RESOURCE_PREFIX = "http://dbpedia.org/resource/"

LABEL_RE = re.compile(
    r'^<([^>]+)>\s+<http://www\.w3\.org/2000/01/rdf-schema#label>\s+"(.*)"@en\s+\.\s*$'
)
REDIRECT_RE = re.compile(
    r'^<([^>]+)>\s+<http://dbpedia\.org/ontology/wikiPageRedirects>\s+<([^>]+)>\s+\.\s*$'
)
DISAMBIG_RE = re.compile(
    r'^<([^>]+)>\s+<http://dbpedia\.org/ontology/wikiPageDisambiguates>\s+<([^>]+)>\s+\.\s*$'
)
NT_ESCAPE_RE = re.compile(r"\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|.)")
DISAMBIG_SUFFIX_RE = re.compile(r"\s*\(disambiguation\)$", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def nt_unescape(s: str) -> str:
    """Decode N-Triples string escapes (\\uXXXX, \\", \\\\, ...)."""
    def repl(m: re.Match) -> str:
        g = m.group(1)
        if g[0] in "uU":
            return chr(int(g[1:], 16))
        return {"t": "\t", "n": "\n", "r": "\r"}.get(g, g)

    return NT_ESCAPE_RE.sub(repl, s)


def normalize(sf: str) -> str:
    return WS_RE.sub(" ", sf).strip().lower()


def local_name(uri: str) -> str:
    if not uri.startswith(RESOURCE_PREFIX):
        return ""
    return unquote(uri[len(RESOURCE_PREFIX):])


def open_dump(path: str):
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def find_dump(stem: str) -> str:
    matches = sorted(glob.glob(os.path.join(DUMPS_DIR, f"{stem}*.ttl*")))
    if not matches:
        sys.exit(f"no dump matching {stem}*.ttl* in {DUMPS_DIR} — run download_dumps.py first")
    return matches[0]


def parse_label(line: str):
    m = LABEL_RE.match(line)
    if not m:
        return None
    uri, literal = m.groups()
    candidate = local_name(uri)
    if not candidate:
        return None
    return normalize(nt_unescape(literal)), candidate


def parse_redirect(line: str):
    m = REDIRECT_RE.match(line)
    if not m:
        return None
    alias_uri, target_uri = m.groups()
    alias, target = local_name(alias_uri), local_name(target_uri)
    if not alias or not target:
        return None
    return normalize(alias.replace("_", " ")), target


def parse_disambiguation(line: str):
    m = DISAMBIG_RE.match(line)
    if not m:
        return None
    page_uri, candidate_uri = m.groups()
    page, candidate = local_name(page_uri), local_name(candidate_uri)
    if not page or not candidate:
        return None
    surface = DISAMBIG_SUFFIX_RE.sub("", page.replace("_", " "))
    return normalize(surface), candidate


def load_file(r: redis.Redis, path: str, parse, score: float, limit: int | None) -> tuple[int, int]:
    """Stream path into Redis. Returns (lines_read, candidates_added)."""
    added = 0
    lines = 0
    pipe = r.pipeline(transaction=False)
    pending = 0

    def flush() -> int:
        nonlocal pending
        results = pipe.execute()
        pending = 0
        return sum(results)  # HSETNX returns 1 only for newly added fields

    with open_dump(path) as f:
        for line in tqdm(f, desc=os.path.basename(path), unit=" lines", unit_scale=True):
            lines += 1
            if lines % 1_000_000 == 0:
                tqdm.write(f"  {os.path.basename(path)}: {lines:,} lines processed")
            parsed = parse(line)
            if parsed:
                surface, candidate = parsed
                if surface and candidate:
                    pipe.hsetnx(f"sf:{surface}", candidate, score)
                    pending += 1
                    if pending >= BATCH_SIZE:
                        added += flush()
            if limit is not None and lines >= limit:
                break
    if pending:
        added += flush()
    return lines, added


def count_sf_keys(r: redis.Redis) -> int:
    n = 0
    for _ in r.scan_iter(match="sf:*", count=10_000):
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only process the first N lines of each file (for testing)")
    args = ap.parse_args()

    r = redis.Redis(port=REDIS_PORT, decode_responses=True)
    r.ping()

    stages = [
        ("labels_lang=en", parse_label, 1.0),
        ("redirects_lang=en", parse_redirect, 0.9),
        ("disambiguations_lang=en", parse_disambiguation, 0.8),
    ]

    total_added = 0
    for stem, parse, score in stages:
        path = find_dump(stem)
        print(f"\n=== loading {os.path.basename(path)} (score {score}) ===")
        lines, added = load_file(r, path, parse, score, args.limit)
        total_added += added
        print(f"  {lines:,} lines read, {added:,} new candidates added")

    print("\ncounting sf:* keys...", flush=True)
    keys = count_sf_keys(r)
    print(f"index keys (sf:*): {keys:,}")
    print(f"candidates added this run: {total_added:,}")


if __name__ == "__main__":
    main()
