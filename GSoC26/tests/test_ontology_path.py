"""Offline test: dbpedia.owl must load regardless of the working directory
(no network beyond the embedding model, no LLM calls).

Before the fix the loader looked for a bare "dbpedia.owl" in the cwd, missed
GSoC26/data/dbpedia.owl, and only printed a warning — so every other test ran
with ontology grounding silently disabled.

Run (from GSoC26/):  USE_TF=0 PYTHONPATH=src python3 tests/test_ontology_path.py
"""
import os, sys, tempfile, types
os.environ.setdefault("USE_TF", "0")
os.environ.pop("DBPEDIA_OWL_PATH", None)

# Import from a directory that does NOT contain dbpedia.owl.
os.chdir(tempfile.mkdtemp())
import autonomous_pipeline_v13 as pl

results = []
def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")

print("== ontology loads from an unrelated cwd ==")
n = len(pl.ontology_vectors)
check(f"properties loaded >= {pl.EXPECTED_MIN_ONTOLOGY_PROPS} (got {n})",
      n >= pl.EXPECTED_MIN_ONTOLOGY_PROPS, True)
check("datatype properties loaded", len(pl.DATATYPE_PROPERTIES) > 0, True)
check("resolved file is the repo snapshot",
      os.path.relpath(pl._find_ontology_file(), os.path.dirname(pl.__file__)),
      os.path.join("..", "data", "dbpedia.owl"))

print("== notebook cell (no __file__) run from GSoC26/ ==")
gsoc26 = os.path.dirname(os.path.dirname(os.path.abspath(pl.__file__)))
os.chdir(gsoc26)
no_file = {k: v for k, v in vars(pl).items() if k != "__file__"}
find_in_cell = types.FunctionType(pl._find_ontology_file.__code__, no_file)
check("finds data/dbpedia.owl via the cwd fallback", find_in_cell(),
      os.path.join(gsoc26, "data", "dbpedia.owl"))

print("== DBPEDIA_OWL_PATH override ==")
os.environ["DBPEDIA_OWL_PATH"] = "~/elsewhere/dbpedia.owl"
check("env var wins and '~' is expanded", pl._find_ontology_file(),
      os.path.join(os.path.expanduser("~"), "elsewhere", "dbpedia.owl"))
del os.environ["DBPEDIA_OWL_PATH"]

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
