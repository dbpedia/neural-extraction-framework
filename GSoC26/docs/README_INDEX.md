# Local DBpedia Surface-Form Index

Redis-backed replacement for live DBpedia Lookup API calls. Maps a text
mention to ranked DBpedia entity candidates, fully offline.

Built from the DBpedia Databus **generic** group, release **`2022.12.01`**
(English). Artifact names (note the modern Databus renames):

| dataset (classic name) | Databus artifact | file | used for |
|---|---|---|---|
| labels_en | `labels` | `labels_lang=en.ttl.bz2` | surface forms, tier 1.0 |
| redirects_en | `redirects` | `redirects_lang=en.ttl.bz2` | alias → canonical, tier 0.9 |
| disambiguations_en | `disambiguations` | `disambiguations_lang=en.ttl.bz2` | ambiguous → candidates, tier 0.8 |
| **page_links_en** | **`wikilinks`** | `wikilinks_lang=en.ttl.bz2` | in-degree popularity prior |

## Redis layout

- Port **6380** by default (all scripts take/assume this; shared-instance safe:
  only `sf:*` and `pop:*` namespaces are ever written, nothing is flushed).
- `sf:<surface form>` — HASH. Field = DBpedia URI local name (percent-decoded),
  value = **`<tier>|<final>`**, e.g. `0.9|5.8974`.
  - `tier` ∈ {1.0 labels, 0.9 redirects, 0.8 disambiguations} — the source
    dataset the pair came from, max-tier-wins.
  - `final = tier * (1 + log10(1 + indegree))` — popularity-weighted score.
  - Surface forms are lowercased with whitespace collapsed.
  - Values written by a fresh `build_index.py` run before rescoring are a bare
    tier (`1.0`); `surface_index.py` accepts both formats.
- `pop:<local name>` — STRING int, wikilink in-degree. **Only needed to
  (re)compute scores** — `lookup()` never reads it. Deletable after rescoring
  to save ~1.1 GiB; rebuild with `build_popularity.py` if scores must be
  recomputed later.

Memory (measured on the full English index, Redis `used_memory`):
~3.7 GiB total = ~1.8 GiB `sf:*` (16.16 M keys) + ~1.1 GiB `pop:*` (21.68 M
keys) + allocator overhead. Without `pop:*`: ~2.5 GiB.

## Scripts

| script | purpose |
|---|---|
| `download_dumps.py` | Resolve current dump URLs live from the Databus SPARQL endpoint and download with resume. `--artifacts`, `--version` (pin, e.g. `2022.12.01`), `--no-count`. |
| `build_index.py` | Stream the three dumps into `sf:*` (HSETNX, batched pipeline). `--limit N` for trials — **run `repair_tiers.py` after mixing a trial with a full run** (HSETNX means a trial's lower-tier write can block the full run's higher tier). |
| `build_popularity.py` | Stream `wikilinks` and count inbound links per target into `pop:*`. Idempotent (clears `pop:*` first). `--limit N`. |
| `rescore_index.py` | Rewrite every `sf:*` value to `tier|final` using `pop:*`. Idempotent (tier is stored, never re-derived from a rescored value). |
| `repair_tiers.py` | Re-stream labels + redirects and raise any stored tier that is too low (max-tier-wins repair). Idempotent. |
| `surface_index.py` | The lookup library (see below). |
| `test_index.py` | 5 smoke assertions. (The `Arion_(comicsCharacter)` case fails by design — that URI is a WebNLG annotation artifact that never existed in DBpedia.) |
| `eval_ranking.py` | Text2KGBench recall@15 / rank@1 / rank@3 per domain; `--compare` evaluates alternative sort keys. Needs `~/Text2KGBench`. |
| `export_index.py` / `import_index.py` | Transfer the finished `sf:*` index between machines (gzipped NDJSON, count-verified). |

Dependencies: `python3 -m pip install redis tqdm requests` (requests only for
downloading; tqdm only for progress bars).

## lookup()

```python
from surface_index import lookup
lookup("USA")        # -> [("United_States", 5.87), ...]  up to k=15
lookup("nonsense strings return")  # -> []
```

Mention normalisation: lowercase, collapse whitespace, then try in order:
exact form, punctuation-stripped form, then internal-punctuation fallbacks
(periods removed, hyphens→spaces — "108 St. Georges Terrace", "Al-Asad
Airbase"); each base also gets an underscore-joined variant (`" "`→`"_"`),
an underscores-to-spaces variant, and a leading `"the "`-stripped variant.
First variant with a non-empty pool wins.
Exact-before-stripped matters: stripping would truncate the trailing `)` of
`"Populous (company)"`.

Ranking: **tier is dominant** — candidates sort by (tier desc, final desc,
name). Popularity only breaks ties within a tier. Validated on Text2KGBench
(4 domains, 2291 mentions): strict tier dominance scored 97.4% rank@1 vs
95.1% for the best alternative (merged label+redirect tier) and 93.4% for
popularity-only. Returned score is `final`, so a lower-tier candidate can
display a larger number than a higher-ranked one — sort position, not the
printed score, is the ranking.

## Transferring the index to another machine

On the source: `python3 export_index.py` → `sf_index_export.ndjson.gz`.
On the target (fresh or shared Redis — only `sf:*` is written):

```
python3 import_index.py sf_index_export.ndjson.gz --port 6380
```

The import verifies the key count against the dump's trailing `_count`
record. `pop:*` is intentionally not exported (scores are already baked in).

## Verifying an import

```
python3 test_index.py       # expect 4/5 (see Arion note above)
python3 eval_ranking.py     # needs Text2KGBench; expect ≈ the table below
```

Reference numbers (release 2022.12.01, after rescore + repair):

```
domain                 recall@15   rank@1   rank@3
4_building                 98.4%    98.4%    98.4%
10_comicscharacter         74.5%    74.5%    74.5%   (gap = WebNLG artifact URIs)
13_food                   100.0%   100.0%   100.0%
1_university              100.0%   100.0%   100.0%
OVERALL                    97.4%    97.4%    97.4%
```

Quick no-benchmark spot checks: `lookup("Perth")` → `Perth` first;
`lookup("USA")` → `United_States` in the top 3; `lookup("Java")` →
`Java` (island) first, `JAVA` second (tier dominance in action).
