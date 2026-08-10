#!/usr/bin/env python3
"""Sanity tests for the Redis surface-form index. Prints pass/fail per case
instead of dying on the first failure; exits 1 if any case fails."""
import sys

from surface_index import lookup

LONG_ADDRESS = ("In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, "
                "Hessarghatta Main Road, Bangalore - 560090.")

failures = 0


def check(description: str, ok: bool, detail: str) -> None:
    global failures
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures += 1
    print(f"[{status}] {description}")
    print(f"       {detail}")


def contains(mention: str, expected: str) -> None:
    results = lookup(mention)
    uris = [uri for uri, _ in results]
    check(f'lookup({mention!r}) contains {expected!r}',
          expected in uris,
          f"got {len(results)} candidates: {uris[:5]}{'...' if len(uris) > 5 else ''}")


def non_empty(mention: str) -> None:
    results = lookup(mention)
    check(f'lookup({mention!r}) is non-empty',
          len(results) > 0,
          f"got {results[:5]}")


def empty(mention: str) -> None:
    results = lookup(mention)
    check(f'lookup({mention[:40]!r}...) == []',
          results == [],
          f"got {results[:5]}")


contains("Perth", "Perth")
contains("USA", "United_States")
contains("Arion", "Arion_(comicsCharacter)")
non_empty("108 St Georges Terrace")
empty(LONG_ADDRESS)

print(f"\n{5 - failures}/5 passed")
sys.exit(1 if failures else 0)
