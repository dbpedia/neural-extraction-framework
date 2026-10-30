"""Offline tests for the /extract response and per-triple verification status
(no network, no LLM, no Redis).

1. The SPARQL gate reports VERIFIED only when the exact fact is stored
   (Tier 1, or the inverted-exact ASK); every other accept path is unverified.
2. extract_all_triples() tags each kept triple "verified" / "faithful-unverified".
3. POST /extract returns the list of kept triples, not the whole record.

Run (from GSoC26/):  USE_TF=0 PYTHONPATH=src:deploy python3 tests/test_extract_status.py
"""
import os, sys
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")   # pass serve.py's 503 guard; no LLM is called

import numpy as np
import sentence_transformers


class _OfflineEmbedder:
    """Stands in for all-MiniLM-L6-v2 so importing the pipeline downloads nothing.
    These tests never compare embeddings; they only need encode() to return arrays."""
    def __init__(self, *args, **kwargs): pass
    def encode(self, texts, *args, **kwargs):
        if isinstance(texts, str):
            return np.zeros(384, dtype=np.float32)
        return np.zeros((len(texts), 384), dtype=np.float32)


sentence_transformers.SentenceTransformer = _OfflineEmbedder   # before the pipeline import

import autonomous_pipeline_v13 as pl
from langgraph.graph import StateGraph, END

R, O = "http://dbpedia.org/resource/", "http://dbpedia.org/ontology/"
SUB, PRED, OBJ = R + "Mermaid_(Train_song)", O + "artist", R + "Train_(band)"
EXACT    = f"<{SUB}> <{PRED}> <{OBJ}>"
INVERTED = f"<{OBJ}> <{PRED}> <{SUB}>"


class _Resp:
    status_code = 200
    def __init__(self, b): self._b = b
    def json(self): return {"boolean": self._b}


def run_gate(truths, any_relation=False, unreachable=False):
    """truths: {query substring: bool}. Returns (gate result, trace)."""
    def fake_get(url, params=None, timeout=None, headers=None):
        if unreachable:
            raise ConnectionError("endpoint down")
        q = (params or {}).get("query", "")
        return _Resp(next((v for k, v in truths.items() if k in q), False))
    orig_get, orig_rel = pl.requests.get, pl.check_any_relation
    pl.requests.get = fake_get
    pl.check_any_relation = lambda *a, **k: any_relation
    orig_sleep, pl.time.sleep = pl.time.sleep, lambda s: None
    trace = {}
    try:
        return pl.verify_override_schema(SUB, PRED, OBJ, {"range": []}, trace=trace), trace
    finally:
        pl.requests.get, pl.check_any_relation, pl.time.sleep = orig_get, orig_rel, orig_sleep


results = []
def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")

print("== gate trace ==")
both_exist = {f"<{SUB}> ?p ?o": True, f"<{OBJ}> ?p ?o": True}
res, tr = run_gate({EXACT: True})
check("Tier 1 exact -> verified", (res, tr.get("verified")), (True, True))
res, tr = run_gate({INVERTED: True})
check("inverted exact -> verified (caller swaps)", (res, tr.get("verified")), ("INVERTED", True))
res, tr = run_gate(both_exist, any_relation=True)
check("Tier 1.5 connected -> not verified", (res, tr.get("verified")), (True, None))
res, tr = run_gate(both_exist)
check("Tier 2 faithful -> not verified", (res, tr.get("verified")), (True, None))
res, tr = run_gate({}, unreachable=True)
check("endpoint down, fail open -> not verified", (res, tr.get("verified")), (True, None))
check("trace is optional (old call signature)",
      pl.verify_override_schema.__defaults__, (None,))

print("== gate_verified survives the LangGraph state ==")
# A key missing from GraphState would be dropped from the graph's output.
g = StateGraph(pl.GraphState)
g.add_node("gate", lambda st: {"gate_verified": True})
g.set_entry_point("gate")
g.add_edge("gate", END)
check("graph output keeps gate_verified",
      g.compile().invoke({"sentence": "x"}).get("gate_verified"), True)

print("== extract_all_triples status ==")
# Stub the LLM nodes and the per-triple graph; extract_all_triples runs as is.
graph_outs = [
    {"validation_status": "APPROVED", "final_triple": EXACT, "gate_verified": True},
    {"validation_status": "APPROVED", "final_triple": f"<{SUB}> <{O}producer> <{R}Espionage_(production_team)>",
     "gate_verified": False},
    # literal object: gate is skipped, so never verified even if a stale flag is set
    {"validation_status": "APPROVED", "final_triple": f'<{SUB}> <{O}releaseDate> "2012"',
     "gate_verified": True},
]
extracted = [{"subject": "Mermaid", "predicate": "artist", "object": "Train"},
             {"subject": "Mermaid", "predicate": "producer", "object": "Espionage"},
             {"subject": "Mermaid", "predicate": "releaseDate", "object": "2012"}]

class _Graph:
    def __init__(self): self.calls = iter(graph_outs)
    def invoke(self, state): return next(self.calls)

orig = (pl.node_0_preprocessor, pl.node_1_extractor, pl.compiled_triple_graph)
pl.node_0_preprocessor = lambda st: {"sentence": st["sentence"]}
pl.node_1_extractor   = lambda st: {"extracted_triples": extracted}
pl.compiled_triple_graph = _Graph()
out = pl.extract_all_triples("The song Mermaid is by the band Train.", verbose=False)
status = {t["rel"]: t["status"] for t in out["triples"]}
check("exact fact -> verified", status.get("artist"), "verified")
check("unverified URI triple -> faithful-unverified", status.get("producer"), "faithful-unverified")
check("literal -> faithful-unverified", status.get("releaseDate"), "faithful-unverified")
check("sub/rel/obj unchanged", out["triples"][0]["obj"], "Train_(band)")

# The gate verified <artist>, but an alias remap emits a different predicate,
# so the emitted triple is not the verified fact.
pl.compiled_triple_graph = _Graph()
out = pl.extract_all_triples("The song Mermaid is by the band Train.", verbose=False,
                             predicate_aliases={"artist": "musicalArtist"})
status = {t["rel"]: t["status"] for t in out["triples"]}
check("alias-remapped triple -> faithful-unverified",
      status.get("musicalArtist"), "faithful-unverified")

print("== POST /extract ==")
import serve
from fastapi.testclient import TestClient
pl.compiled_triple_graph = _Graph()
body = TestClient(serve.app).post(
    "/extract", json={"text": "The song Mermaid is by the band Train."}).json()
check("triples is a list", type(body["triples"]).__name__, "list")
check("only the triples are returned", sorted(body), ["triples"])
check("each triple has sub/rel/obj/status",
      [sorted(t) for t in body["triples"]], [["obj", "rel", "status", "sub"]] * 3)
pl.node_0_preprocessor, pl.node_1_extractor, pl.compiled_triple_graph = orig

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
