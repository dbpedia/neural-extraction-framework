# RUNBOOK — bare server → full 19-domain Text2KGBench run

Everything below assumes the GSoC server layout
`/root/projects/GSoC26_Extraction_Nakul/knowledge_graph_pipeline/`.
Copy the entire folder from the Mac (excluding `dumps/` — not needed at
runtime) plus `sf_index_export.ndjson.gz`.

## 1. System prerequisites

```
# Redis on port 6380 (do NOT flush an existing shared instance)
redis-server --port 6380 --daemonize yes

# Python deps
python3 -m pip install redis tqdm requests rdflib numpy \
    sentence-transformers langchain-core langchain-openai langgraph
```

## 2. Import the index (~10 min)

```
python3 import_index.py sf_index_export.ndjson.gz --port 6380
```

Expect: `imported 16,161,334 sf:* keys + 831,508 rd:* keys` and
`counts match dump record ✓`. The importer verifies against the dump's
trailing count record and writes only `sf:*`/`rd:*` — shared-instance safe.
Redis memory afterwards: ~2.6 GiB.

## 3. Verify the index

```
USE_TF=0 python3 test_index.py       # expect 4/5 — the Arion case fails BY
                                     # DESIGN (WebNLG artifact URI, see README)
USE_TF=0 python3 eval_ranking.py     # needs ~/Text2KGBench; expect ~97.4%
                                     # recall@15 AND rank@1 on the 4 domains
```

## 4. Environment

```
export OPENROUTER_API_KEY=sk-or-...   # or put it in the shell profile
export USE_TF=0                       # REQUIRED if TensorFlow is installed and
                                      # broken/mismatched — it deadlocks
                                      # transformers' lazy TF import at C level
```

`dbpedia.owl` must sit in the working directory — **see §7, this file differs
between machines and it matters.**

## 5. Flags (module-level constants in autonomous_pipeline_v13.py)

| flag | run value | meaning |
|---|---|---|
| `USE_LOCAL_INDEX` | `True` | Node 2 uses the local sf:* index. `False` = the untouched web-API path (the paper's ablation baseline). |
| `CANONICALIZE_COMMA_COMPOUNDS` | `True` | narrow redirect canonicalization at emission (needs rd:* keys). |
| `TARGET_MODEL` | `openai/gpt-5.6-luna` | set per experiment. |

## 6. Full 19-domain run

```
USE_TF=0 python3 - << 'EOF'
import autonomous_pipeline_v13 as pl
pl.TARGET_MODEL = "openai/gpt-5.6-luna"
pl.USE_LOCAL_INDEX = True
ns = {"extract_all_triples": pl.extract_all_triples}
exec(compile(open("chunked_runner.py").read(), "chunked_runner.py", "exec"), ns)
from text2kg_harness import all_domains, macro_average
scores = {}
for slug in sorted(all_domains(), key=lambda s: int(s.split("_")[0])):
    P, R, F, rows = ns["chunked_domain"](slug)
    scores[slug] = (P, R, F)
print("MACRO:", macro_average(scores))
EOF
```

Each domain saves `results_<slug>.json` after every chunk, so a crash never
costs more than one chunk; re-running a domain overwrites its file.
Budget ~30–120 s/sentence depending on dbpedia.org's mood (the SPARQL gate
and topology checks hit the live endpoint).

## 7. THE dbpedia.owl CAVEAT (server numbers may differ from the Mac's)

The Mac used the **Archivo 2024.08.01 snapshot: 1,194 ObjectProperties /
2,041 DatatypeProperties** (printed at import as
`Semantic Vector Space built for 3235 properties`). The server's own
dbpedia.owl is a different snapshot. This file feeds three things:

1. **Predicate linking space** (Node 3) — a pid missing from the OWL is
   embedded on the fly from its camelCase label instead of its rdfs:label +
   comment, which slightly changes ranking.
2. **DATATYPE_PROPERTIES** — decides the literal short-circuit. A predicate
   that is a DatatypeProperty in one snapshot and absent in the other will
   route its objects differently (entity-linking vs literal).
3. **ONTOLOGY_RULES domain/range** — drives the schema-inversion Tier 1.6
   check and the type-match penalties.

**After import, check:** the startup line `Semantic Vector Space built for N
properties (X object, Y datatype)`. The Mac's exact dbpedia.owl is INCLUDED
in the transfer bundle (Archivo 2024.08.01; verify with the sha256 in the
bundle manifest) — use it, and expect exactly
`3235 properties (1194 object, 2041 datatype)`. For paper-final numbers, use
the SAME dbpedia.owl on every machine involved.

## 8. Sanity re-run (optional but recommended)

Reproduce the Mac's 4_building A/B before the full sweep:
`USE_LOCAL_INDEX=False` expect ≈0.45–0.48, `True` expect ≈0.50–0.54
(gpt-5.6-luna; single-run noise is ±0.03 — see CHANGES.md).
