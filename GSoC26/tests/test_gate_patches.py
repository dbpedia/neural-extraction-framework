"""Offline tests for the 2026-08-04 gate patches (no network, no LLM calls).

Patch 1 — TIER16_RANGE_SWAP flag: range-membership swaps disabled by default;
          inverted-exact swaps unaffected; flag=True restores old behaviour.
Patch 2 — OBJ_MISSING sentinel: subject-real/object-absent demotes the object
          to a quoted literal instead of deleting the triple.

Run:  USE_TF=0 python3 test_gate_patches.py
"""
import os, re, sys
os.environ.setdefault("USE_TF", "0")

import autonomous_pipeline_v13 as pl

SUB  = "http://dbpedia.org/resource/SUBJ_ENTITY"
PRED = "http://dbpedia.org/ontology/somePred"
OBJ  = "http://dbpedia.org/resource/OBJJ_ENTITY"
RULE = {"range": ["Place", "Agent"]}


class _Resp:
    status_code = 200
    def __init__(self, b): self._b = b
    def json(self): return {"boolean": self._b}


def make_fake_get(truths):
    """truths: ordered list of (substring, bool). First match wins, else False."""
    def fake_get(url, params=None, timeout=None, headers=None):
        q = (params or {}).get("query", "")
        for frag, val in truths:
            if frag in q:
                return _Resp(val)
        return _Resp(False)
    return fake_get


def run_gate(truths, any_relation=False):
    orig_get, orig_rel = pl.requests.get, pl.check_any_relation
    pl.requests.get = make_fake_get(truths)
    pl.check_any_relation = lambda *a, **k: any_relation
    try:
        return pl.verify_override_schema(SUB, PRED, OBJ, RULE)
    finally:
        pl.requests.get, pl.check_any_relation = orig_get, orig_rel


# Query fragments (unambiguous substrings of the queries the gate builds)
EXACT      = f"<{SUB}> <{PRED}> <{OBJ}>"
INVERTED   = f"<{OBJ}> <{PRED}> <{SUB}>"
SUB_RANGE  = f"<{SUB}> a <http://dbpedia.org/ontology/"
OBJ_RANGE  = f"<{OBJ}> a <http://dbpedia.org/ontology/"
SUB_EXISTS = f"<{SUB}> ?p ?o"
OBJ_EXISTS = f"<{OBJ}> ?p ?o"

results = []
def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")

print("== Patch 1: TIER16_RANGE_SWAP ==")
# Range asymmetry present (obj NOT in range, sub IS), no stored fact either way,
# both URIs exist -> old code swapped; new default must fall through to Tier 2.
asym = [(EXACT, False), (INVERTED, False),
        (OBJ_RANGE, False), (SUB_RANGE, True),
        (SUB_EXISTS, True), (OBJ_EXISTS, True)]
assert pl.TIER16_RANGE_SWAP is False, "default must be False"
check("range-asymmetry alone no longer swaps (flag off)", run_gate(asym), True)

pl.TIER16_RANGE_SWAP = True
check("flag=True restores range swap", run_gate(asym), "INVERTED")
pl.TIER16_RANGE_SWAP = False

# Inverted-exact evidence must still swap with the flag off.
inv = [(EXACT, False), (INVERTED, True)]
check("inverted-exact still swaps (flag off)", run_gate(inv), "INVERTED")

# Tier 1 exact must still short-circuit everything.
check("exact triple still passes Tier 1", run_gate([(EXACT, True)]), True)

print("== Patch 2: OBJ_MISSING ==")
# Subject real, object URI absent, no relation -> demote sentinel (was: False).
miss_obj = [(EXACT, False), (INVERTED, False),
            (SUB_EXISTS, True), (OBJ_EXISTS, False)]
check("subject-real / object-absent returns OBJ_MISSING", run_gate(miss_obj), "OBJ_MISSING")

# Subject absent must still hard-block (nothing to demote a subject to).
miss_sub = [(EXACT, False), (INVERTED, False),
            (SUB_EXISTS, False), (OBJ_EXISTS, True)]
check("subject-absent still blocks", run_gate(miss_sub), False)

# Both absent must still hard-block.
miss_both = [(EXACT, False), (INVERTED, False),
             (SUB_EXISTS, False), (OBJ_EXISTS, False)]
check("both-absent still blocks", run_gate(miss_both), False)

# Both real, no edge, FAITHFUL mode -> unchanged Tier-2 accept.
check("Tier 2 faithful accept unchanged", run_gate(asym), True)

# Tier 1.5 any-relation accept unchanged.
check("Tier 1.5 accept unchanged", run_gate(asym, any_relation=True), True)

print("== Demoted-literal string round-trip ==")
# The caller emits `<sub> <pred> "raw words"`; the harness literal regex
# (autonomous_pipeline_v13 result parsing) must classify it as a literal triple
# so _normalize_literal / quote hints apply.
demoted = f'<{SUB}> <{PRED}> "Government of Addis Ababa"'
m = (re.match(r'^<([^>]*)>\s+<([^>]*)>\s+"(.*)"\s*$', demoted)
     or re.match(r'^<([^>]*)>\s+<([^>]*)>\s+<"(.*)">\s*$', demoted))
check("demoted literal matches harness literal regex",
      bool(m) and m.group(3), "Government of Addis Ababa")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
