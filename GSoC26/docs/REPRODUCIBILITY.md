# Reproducibility Bundle — final benchmark run

Everything needed to regenerate the reported Text2KGBench (dbpedia_webnlg) numbers.

## Code
- **Repository tag:** `v14-final` (commit `bf6bad0`) — the frozen pipeline version; all 19 final domain scores come from this exact code.
- **Entry point:** `chunked_runner.py` → `chunked_domain(<slug>)`; pipeline in `autonomous_pipeline_v13.py` (v14 patches included, see `CHANGES.md`).
- **Prompts:** embedded in `autonomous_pipeline_v13.py` (Node 0 pre-processor, Node 1 extractor, Judge).

## Model
- **LLM:** `openai/gpt-5.6-luna` via OpenRouter (`https://openrouter.ai/api/v1`), temperature 0, max_retries 1.
- Set `OPENROUTER_API_KEY` in the environment. No fine-tuning anywhere; the LLM is the only non-local, paid component.

## Benchmark
- **Text2KGBench** dbpedia_webnlg split: 19 domains, 2,014 test sentences, official exact-triple metric (per-sentence F1, macro-averaged per domain).
- Local path expected: `~/Text2KGBench/data/dbpedia_webnlg` (train + test + per-domain ontologies).
- **Version:** commit `50a3d255371b8817cdff70fd88459ac82b339cfe` (2024-05-05) of the Text2KGBench repository.
- Train split is used only for learned conventions (quoting, predicate aliases, value formats, few-shot retrieval) — never for weights.

## Local surface-form index (Redis, port 6380)
- 16.2M mention entries built offline from DBpedia dump files (labels, redirects, anchor texts) by `build_index.py`; popularity scores by `build_popularity.py`; import/export via `import_index.py` / `export_index.py`.
- **Dump sources:** DBpedia Databus `generic` artifacts `labels`, `redirects`,
  `disambiguations` (lang=en), resolved to the latest release at build time by
  `download_dumps.py` via the Databus SPARQL endpoint (the script logs the
  resolved version URLs when run).

## Local SPARQL store (oxigraph, port 7878)
- Local DBpedia triple load served at `http://localhost:7878/query`; configured via `pl.DBPEDIA_SPARQL_ENDPOINT`.
- **Loaded dumps** (bulk load completed 2026-06-27; store 18 GB at load,
  15 GB after `optimize.sh` compaction): `dbpedia_text2sparql_ontology.nt`,
  `instance_types_en.ttl`, `instance_types_transitive_en.ttl`,
  `persondata_en.ttl`, `labels_en.ttl`, `short_abstracts_en.ttl`,
  `mappingbased_literals_en.ttl`, `mappingbased_objects_en.ttl`,
  `infobox_properties_en.ttl` (English DBpedia; load script `load.sh` and full
  `load.log` retained on the build host).

## Ontology snapshot
- `dbpedia.owl` snapshot in repo — affects predicate linking, the datatype short-circuit, and gate range rules.

## Configuration (as run)
```python
os.environ["USE_TF"] = "0"
pl.TARGET_MODEL = "openai/gpt-5.6-luna"
pl.USE_LOCAL_INDEX = True
pl.DBPEDIA_SPARQL_ENDPOINT = "http://localhost:7878/query"
# In-code flags at v14-final defaults: TIER16_RANGE_SWAP=False,
# ENFORCE_ENTITY_LOCK=True, FAITHFUL_EXTRACTION_MODE=True.
```

## Run command
```python
ns = {"extract_all_triples": pl.extract_all_triples}
exec(open("chunked_runner.py").read(), ns)
P, R, F, rows = ns["chunked_domain"]("13_food")   # per domain; results_<slug>.json saved incrementally
```

## Determinism notes
- Temperature 0; index, ranking, gate, and formatting are fully deterministic.
- Residual variance source: OpenRouter may route between providers of the same model. Measured reproducibility: 10_comicscharacter full re-run (Mac vs server) differed by 0.001 F1.

## Fix scope (Ara's question 3)
| Fix (v14) | Code path | Scope |
|---|---|---|
| Per-domain literal flags (leak fix) | Node 3 datatype short-circuit | Generic mechanism; behavior learned per-domain from train (by design) |
| Context-qualified sense lookup + qualifier bonus | Node 2 candidate fetch, Node 3 ranking | Generic — driven by sentence words only |
| Index cross-check before literal demotion | SPARQL gate | Generic |
| Symmetric-predicate swap guard | SPARQL gate | Generic (fixed predicate list from DBpedia semantics) |
| Year/currency literal polish | Emitter | Generic |

No fix references any benchmark domain by name.
