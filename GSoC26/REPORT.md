# Neural Extraction Framework 2.0 — Final Benchmark Report

**GSoC 2026 · DBpedia · Nakul Singh** · mentors: Ara, Tommaso Soru
Code frozen at tag [`v14-final`](https://github.com/NakulSingh156/dbpedia-neuro-symbolic-extraction/releases/tag/v14-final) · Image: `ghcr.io/nakulsingh156/neural-extraction-framework:v2.0`

## Summary

An LLM + knowledge-base pipeline that converts natural-language sentences into
verified DBpedia triples. On **Text2KGBench (dbpedia_webnlg)** — all 19 domains,
2,014 sentences, official exact-triple metric — it reaches **macro F1 = 0.6317**,
matching the published GPT-4o NEF baseline (**0.628**) on a model **~20× cheaper
per token**, at **~3–4× lower end-to-end benchmark cost** (≈$8–10 vs ≈$30–35),
with every non-LLM component running locally. The pipeline is packaged as a
publicly pullable Docker service.

## 1. Task and benchmark

Text2KGBench pins each sentence to a per-domain ontology and scores exact triple
matches (per-sentence F1, macro-averaged over the 19 domains). The train split may
be used for prompting/conventions; we use **no fine-tuning** anywhere.

## 2. Final results

One frozen pipeline version (`v14-final`), one host environment, full sweep.
Per-sentence outputs: [`results/final_clean/`](results/final_clean/).

| Domain | Ours | NEF (GPT-4o) | Δ |
|---|---|---|---|
| politician | 0.7991 | 0.722 | +0.077 |
| scientist | 0.7732 | 0.561 | **+0.212** |
| astronaut | 0.7377 | 0.730 | +0.008 |
| writtenwork | 0.7309 | 0.768 | −0.037 |
| celestialbody | 0.7233 | 0.707 | +0.016 |
| film | 0.7177 | 0.753 | −0.035 |
| sportsteam | 0.7103 | 0.679 | +0.031 |
| food | 0.7066 | 0.765 | −0.058 |
| city | 0.6853 | 0.747 | −0.062 |
| company | 0.6804 | 0.494 | **+0.186** |
| building | 0.6667 | 0.648 | +0.019 |
| musicalwork | 0.6063 | 0.443 | **+0.163** |
| airport | 0.6051 | 0.721 | −0.116 |
| university | 0.5935 | 0.604 | −0.011 |
| athlete | 0.5566 | 0.577 | −0.020 |
| artist | 0.5264 | 0.609 | −0.083 |
| meanoftransportation | 0.5014 | 0.523 | −0.022 |
| comicscharacter | 0.4000 | 0.519 | −0.119 |
| monument | 0.2820 | 0.370 | −0.088 |
| **MACRO** | **0.6317** | **0.628** | **+0.004** |

We lead the baseline on 8/19 domains — including **musicalwork (+0.163), NEF's
single worst domain**. The macro margin is within measured run-to-run variance
(per-domain ±0.02, macro ±0.005; temperature 0 throughout — residual variance is
provider-side serving nondeterminism), so the defensible claim is **parity with
GPT-4o at ~5 % of its cost**, not superiority.

**Measurement integrity notes.** Two domains (university, airport) were first
measured on a memory-exhausted host and re-measured after the fault was fixed,
under a pre-declared identical-conditions policy; the re-measured values are
reported. 3 of 2,014 sentences returned empty predictions from transient API
failures — documented, not retried (≤ 0.001 macro impact). All intermediate
results are archived. Reproducibility spot-check: a full domain re-run on a
different machine differed by 0.001 F1.

## 3. Cost and runtime

| | Ours (GPT-5.6 Luna) | NEF (GPT-4o) |
|---|---|---|
| Model price (in/out per M tokens) | $0.10 / $0.60 | $2.50 / $10.00 (~20× ours) |
| LLM calls per sentence | 8–12 (extract + judge + repair loops) | ~1–2 |
| Per sentence | ≈ $0.004–0.005 | ≈ $0.015–0.017 |
| Full benchmark (2,014 sentences) | **≈ $8–10** | ≈ $30–35 (as reported by mentors) |
| Wall-clock | ~27 s/sentence; ~16 h unattended on one 8 GB VPS | — |

Two comparisons, kept separate deliberately: **per token** the model is ~20×
cheaper; **system vs system** the full benchmark run is ~3–4× cheaper. The gap
between the two ratios is by design — this pipeline spends ~10× more LLM calls
per sentence than the baseline, funding judging and verification with the saved
model cost, and still lands cheaper overall at the same accuracy. (Running this
call-heavy pipeline unchanged at GPT-4o prices would cost ≈$180–200 — the cheap
model is what makes verification-heavy extraction affordable.)

The LLM is the only paid or online component. The endpoint is OpenAI-compatible
and swappable — a local vLLM/Ollama model gives a 100 % offline, $0/sentence
deployment.

## 4. Architecture

Six stages (diagram in [README](README.md)): **Extract** (LLM, few-shot retrieved
from train) → **Candidates** (local Redis surface-form index: 16.2M mentions,
~1 ms lookups) → **Rank** (vector + lexical + type-sanity + sentence-qualifier
bonus) → **Judge** (LLM sentence-fidelity check, ontology-bounded predicates) →
**Verify** (local oxigraph SPARQL gate; tiered: exact → inverted-direction →
any-relation → existence) → **Format** (train-learned quoting/value conventions).

Two design principles carry the result:

1. **Faithfulness over KB-membership.** A triple that is true per the sentence
   but absent from DBpedia is kept as a *faithful extraction* and tagged
   unverified — not deleted as a hallucination. The benchmark scores sentence
   fidelity; so do downstream users, who get a trust tag on every triple.
2. **The KB as a service, not a dependency.** Replacing the live Lookup API and
   dbpedia.org SPARQL with local equivalents made obscure entities resolvable,
   the entity-vs-literal decision deterministic, verification ~100× faster, and
   the whole system rate-limit- and outage-proof.

## 5. What moved the score (engineering journey)

| Stage | Macro | Key change |
|---|---|---|
| Open-weight baseline (LLaMA-3.3-70B, live APIs) | 0.4387 (14 domains) | starting point; candidate generation identified as the bottleneck (49 % of binned errors were entity mis-resolution) |
| + local surface-form index | e.g. building 0.34 → 0.45 → | long-tail entities become candidates at all |
| + local SPARQL gate, tiered verification, learned conventions, GPT-5.6 Luna | 0.598 (19 domains) | first full sweep; local vs remote gate alone measured +0.037 |
| + v14 fixes (below) | **0.6317** | final |

**The v14 fixes** came from an offline autopsy of the three weakest domains
(zero LLM cost), which found two root causes:

- **Cross-domain literal-flag leak** — per-domain "this predicate is a literal"
  flags accumulated in a global set across a multi-domain process; monument's
  convention (quote `country`) silently destroyed 127 gold triples in food and
  86 in writtenwork. Fix: strictly per-domain flags. Food 0.49 → 0.71,
  writtenwork 0.57 → 0.73.
- **Bare-title entity senses** — 35/70 musicalwork and 39/45 film zeros were
  links like `Mermaid` where gold wants `Mermaid_(Train_song)`, even though the
  sentence names the disambiguator. Fix: context-qualified index lookups
  generated from sentence words + a qualifier-support bonus in ranking.
  Film 0.48 → 0.72, musicalwork 0.52 → 0.61.

Every fix is a generic code path; nothing references a benchmark domain by name.
Full fix-by-fix history with measured effects: [`CHANGES.md`](docs/CHANGES.md).

## 6. Reproducibility

[`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) records the complete bundle: code tag,
model string + config, Text2KGBench commit, the Databus artifacts behind the
index, the nine DBpedia dumps loaded into the store, and the run command.

## 7. Deployment

The stack ships as three containers (Redis index · oxigraph store · pipeline
API) behind a single endpoint — [`deploy/DEPLOY.md`](deploy/DEPLOY.md):

```
POST /extract {"text": "The song Mermaid is by the band Train."}
→ {"triples": [{"sub": "Mermaid_(Train_song)", "rel": "artist", "obj": "Train_(band)"}]}
```

Verified end-to-end on a fresh host. Data snapshots (index 1.4 GB, store 15 GB)
are planned for publication on the DBpedia Databus so setup is fully self-serve.

## 8. Limitations

- **Annotation-artifact ceilings**: comicscharacter gold uses `_(comicsCharacter)`
  URIs that never existed in DBpedia; monument is 19 sentences with a case-twin
  entity trap. These bound the two lowest domains, for every system.
- **Run variance**: API-served LLMs at temperature 0 still vary between serving
  providers; macro-level effect measured at ±0.005.
- **Single benchmark split** so far (dbpedia_webnlg).

## 9. Future work

- **wikidata_tekgen** (the other Text2KGBench half) — generalisation beyond DBpedia.
- **Anonymous-node grounding** — treat unnamed entities ("Marie Antoinette's
  husband") as blank nodes with constraints, i.e. as queries; ground against the
  local store only when the answer is unique. Scoped implementation + purpose-built
  testbed planned as the flagship contribution of a paper.
- Databus publication of the data snapshots; image slimming.

## Acknowledgements

Google Summer of Code 2026 & DBpedia. Mentors **Ara** and **Tommaso Soru** —
including Tsoru's framing of the anonymous-node problem. Benchmark:
Text2KGBench (ISWC 2023); baseline: DBpedia NEF (GPT-4o).
