#!/usr/bin/env python3
"""Local surface-form lookup against the Redis index built by build_index.py.

Drop-in replacement for DBpedia Lookup API calls:

    from surface_index import lookup
    lookup("USA")  ->  [("United_States", 0.9), ...]
"""
import os
import re
import string

import redis

REDIS_HOST = os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6380"))
_WS_RE = re.compile(r"\s+")
_STRIP_CHARS = string.punctuation + string.whitespace

_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _normalize(mention: str) -> str:
    return _WS_RE.sub(" ", mention).strip().lower()


def _variants(mention: str) -> list[str]:
    exact = _normalize(mention)
    # Stripping surrounding punctuation would truncate a trailing ")" from
    # parenthesized qualifiers like "Populous (company)", so try exact first.
    stripped = exact.strip(_STRIP_CHARS)
    bases = [exact, stripped]
    # Internal-punctuation fallbacks: "108 St. Georges Terrace" (period) and
    # hyphen/space disagreements ("Al-Asad Airbase" vs "Al Asad Airbase").
    # Appended AFTER the existing bases so they only fire when every current
    # variant misses — existing hits keep their exact behaviour.
    for base in (exact, stripped):
        for pv in (base.replace(".", ""),
                   base.replace("-", " "),
                   base.replace(".", "").replace("-", " ")):
            pv = _WS_RE.sub(" ", pv).strip()
            if pv:
                bases.append(pv)
    out = []
    for base in bases:
        for v in (
            base,
            base.replace(" ", "_"),        # underscore-joined
            base.replace("_", " "),        # underscores back to spaces
            base[4:] if base.startswith("the ") else base,  # "the "-stripped
        ):
            if v and v not in out:
                out.append(v)
    return out


def _parse_score(value: str) -> tuple[float, float]:
    """Stored values are either "<tier>|<final>" (after rescore_index.py) or a
    raw tier weight (fresh build_index.py entries). Returns (tier, final)."""
    if "|" in value:
        tier, final = value.split("|", 1)
        return float(tier), float(final)
    raw = float(value)
    return raw, raw


def candidates(mention: str) -> list[tuple[str, float, float]]:
    """Return the full unranked candidate pool for a mention as
    (uri_local_name, tier, final) tuples. Empty list if unknown."""
    if not mention:
        return []
    for variant in _variants(mention):
        pool = _r.hgetall(f"sf:{variant}")
        if pool:
            return [(uri, *_parse_score(v)) for uri, v in pool.items()]
    return []


def lookup(mention: str, k: int = 15) -> list[tuple[str, float]]:
    """Return up to k (uri_local_name, score) candidates for a mention.
    Ranking: tier (labels > redirects > disambiguations) is dominant; the
    popularity-weighted final score only breaks ties within a tier."""
    pool = candidates(mention)
    pool.sort(key=lambda c: (-c[1], -c[2], c[0]))
    return [(uri, final) for uri, _, final in pool[:k]]


if __name__ == "__main__":
    import sys
    for mention in sys.argv[1:]:
        print(f"{mention!r} -> {lookup(mention)}")
