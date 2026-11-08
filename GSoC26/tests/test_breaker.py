"""Offline tests for the per-sentence circuit breaker (no network, no LLM, no model).

chunked_runner.py calls extract_all_triples with predicate_aliases, value_formats
and fewshot_examples. Once per_sentence_timeout.py has replaced it with
extract_all_triples_safe, those arguments must reach the real pipeline, or every
sentence raises TypeError and the runner scores it as F1=0.

Run (from GSoC26/):  python3 tests/test_breaker.py
"""
import os, sys, time

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
BREAKER = open(os.path.join(SRC, "per_sentence_timeout.py")).read()

results = []
def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")


calls = []
def extract_all_triples(sentence, allowed_predicates=None, literal_predicates=None,
                        predicate_ranges=None, quote_hints=None, predicate_aliases=None,
                        value_formats=None, fewshot_examples=None, verbose=True):
    """Fake pipeline with the real signature; records what it was called with."""
    calls.append(dict(allowed_predicates=allowed_predicates,
                      predicate_aliases=predicate_aliases,
                      value_formats=value_formats,
                      fewshot_examples=fewshot_examples, verbose=verbose))
    return {"triples": [{"sub": "A", "rel": "r", "obj": "B"}]}


def load_breaker(pipeline):
    ns = {"extract_all_triples": pipeline}
    exec(BREAKER, ns)
    return ns


print("== the arguments chunked_runner passes reach the pipeline ==")
ns = load_breaker(extract_all_triples)
out = ns["extract_all_triples"]("s", allowed_predicates={"r"}, literal_predicates=set(),
                                predicate_ranges={}, quote_hints={},
                                predicate_aliases={"currentclub": "club"},
                                value_formats={"r": "x"}, fewshot_examples=["ex"],
                                verbose=False)
check("triples returned (no TypeError)", out["triples"], [{"sub": "A", "rel": "r", "obj": "B"}])
check("predicate_aliases forwarded", calls[-1]["predicate_aliases"], {"currentclub": "club"})
check("value_formats forwarded", calls[-1]["value_formats"], {"r": "x"})
check("fewshot_examples forwarded", calls[-1]["fewshot_examples"], ["ex"])

print("== old call styles still work ==")
ns["extract_all_triples"]("s", {"r"})
check("positional allowed_predicates", calls[-1]["allowed_predicates"], {"r"})
check("verbose defaults to False", calls[-1]["verbose"], False)

print("== timeout and re-running the cell ==")
def slow(sentence, **kwargs):
    time.sleep(1)
    return {"triples": [{"sub": "late", "rel": "r", "obj": "x"}]}
ns = load_breaker(slow)
ns["_HARD_LIMIT"] = 0.1
check("hung sentence returns empty triples", ns["extract_all_triples"]("s")["triples"], [])
exec(BREAKER, ns)   # re-run the cell: must not wrap the wrapper
check("re-run keeps the real pipeline", ns["_orig_extract_all_triples"], slow)

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
