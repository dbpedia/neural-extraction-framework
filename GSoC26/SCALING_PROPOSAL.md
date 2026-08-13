# NEF 2.0 at Scale — Extracting RDF from Wikipedia Abstracts

**A resource, cost, and time study for DBpedia** · Nakul Singh, GSoC 2026 · mentors: Ara, Tommaso Soru
*Status: draft for mentor review. Every number below is either measured from our
benchmark runs, counted from our loaded DBpedia snapshot (query shown), or
carries a dated external source.*

## 1. What we're proposing

Run the NEF 2.0 pipeline (this repository; Docker image
`ghcr.io/nakulsingh156/neural-extraction-framework:v2.0`) over English Wikipedia abstracts
to produce a new, provenance-tagged RDF dataset for the DBpedia Databus —
**triples only on the Databus, the stack stays in the Docker image**, per
Tsoru's guidance. Phased rollout with review gates, starting with a
mentor-reviewed example batch, so DBpedia commits resources only after seeing
output quality at each step.

## 2. Why this is worth DBpedia's sponsorship

1. **Abstracts contain facts infoboxes don't.** DBpedia's mapped triples come
   overwhelmingly from infoboxes; the prose abstract of an article routinely
   asserts relations that never appear in any infobox field. This pipeline
   reads the prose.
2. **Every triple carries a trust tag.** The verification gate labels each
   output `verified` (already supported by DBpedia) or `faithful-unverified`
   (asserted by the sentence, absent from DBpedia — i.e. **the new knowledge**).
   Reviewers and downstream users can filter by trust level; the unverified
   slice *is the product*.
3. **Benchmark-validated quality.** Macro F1 0.6317 on Text2KGBench
   dbpedia_webnlg — level with the GPT-4o NEF 1.0 baseline (0.628) at 3–4×
   lower end-to-end cost ([REPORT.md](REPORT.md)). *Note: that comparison
   varies both method and base model, so it does not by itself establish that
   the method is responsible for the result — see the controlled experiment in
   §4. For the purposes of this proposal what matters is the absolute quality
   level, which is what the Phase 0/1 audits re-measure in the open domain
   anyway.*
4. **Fully open, reproducible stack.** Local surface-form index + local SPARQL
   store, both built from Databus artifacts; the only external call is an
   OpenAI-compatible LLM endpoint — swappable for a self-hosted open model.

## 3. The workload, measured (not estimated)

Counted directly from our loaded snapshot (oxigraph, `short_abstracts_en.ttl`,
load 2026-06-27):

```sparql
SELECT (COUNT(*) AS ?c) WHERE { ?s rdfs:comment ?o }        # → 4,643,098
```

Sentence statistics from a 400-abstract sample (`STRLEN > 40` filter to skip
parse-artifact literals; sentence split on terminal punctuation):

| Quantity | Value | Source |
|---|---|---|
| English short abstracts | **4,643,098** | count query above, our snapshot |
| Sentences per abstract | **2.26 mean / 2 median** | 400-abstract sample |
| Words per abstract | 44.8 mean | same sample |
| **Total sentences (full corpus)** | **≈ 10.5 M** | 4.64 M × 2.26 |

*Scope note:* these are **short** abstracts (lead paragraph). Long abstracts
(`long_abstracts_en.ttl`) run ≈2–3× more sentences; everything below scales
linearly if DBpedia prefers that target.

## 4. Cost model

**Measured unit cost** (2,014-sentence benchmark + 602-sentence rerun, GPT-5.6
Luna via OpenRouter at $0.10/$0.60 per M tokens, 8–12 calls/sentence):
**≈ $0.004–0.005 per sentence**. [Exact OpenRouter invoice total to be attached
from the account's Activity page.]

### Option A — API-served model (turnkey)

| Scope | Abstracts | Sentences | LLM cost | Wall-clock @ 50 concurrent |
|---|---|---|---|---|
| **Phase 0 — review batch** | 100 | ≈226 | **< $2** | ≈ 1 hour |
| **Phase 1 — pilot** | 100 k | ≈226 k | **≈ $0.9–1.1 k** | ≈ 1.5 days |
| **Phase 2** | 1 M | ≈2.26 M | ≈ $9–11 k | ≈ 2 weeks |
| **Phase 3 — full corpus** | 4.64 M | ≈10.5 M | **≈ $42–53 k** | ≈ 9 weeks |

Throughput math: measured ≈27 s/sentence end-to-end is LLM-latency-bound (the
local index and store answer in milliseconds), so it parallelises to ≈C/27
sentences/s at concurrency C. C = 50 → ≈ 160 k sentences/day per orchestrator;
scales linearly with workers/API limits. Infrastructure beyond the LLM: the
current 8 GB VPS already hosts index + store + pipeline (≈ €10–15/month class).

### Option B — self-hosted open model (cheaper at scale, needs a quality pilot)

The endpoint is OpenAI-compatible; a vLLM-served open model on one rented GPU
(A100/H100 class, ≈$1.5–3/hr on Runpod/Lambda-class pricing) turns cost from
per-token to per-hour. At realistic batched-serving throughput this lands the
**full corpus in the low-thousands of dollars** — but output quality with an
open model on the v14 pipeline is unmeasured (our LLaMA-3.3-70B number, 0.4387,
predates the index/gate/fixes). **Phase 1 should include a benchmark re-run of
one open model (≈$0 API cost if self-hosted) before committing Option B.**

### Controlled model experiment (required for publication, ≈ $180–200)

The headline benchmark comparison changes two variables at once — method
(NEF 1.0 → NEF 2.0) *and* base model (GPT-4o → GPT-5.6 Luna) — so it cannot
separate "the pipeline is better" from "the model is better". Holding the model
fixed resolves it:

| | NEF 1.0 (method) | NEF 2.0 (method) |
|---|---|---|
| **GPT-4o** | 0.628 (published) | **to be measured** |
| **GPT-5.6 Luna** | optional 4th cell | 0.6317 (measured) |

Running NEF 2.0 on GPT-4o over the full benchmark (≈$180–200, ≈16 h) gives a
clean method-vs-method comparison at a fixed model, and turns the Luna run into
a separate, honest efficiency result rather than a confounded quality claim.
Recommended sequencing: a 3-domain pilot (≈$25) first, since prompts and
thresholds were tuned with Luna in the loop.

This is a benchmark experiment, not part of the abstract-extraction budget, but
it belongs in the same funding conversation.

## 5. Quality control plan

- **Gate provenance**: every triple tagged `verified` / `faithful-unverified`;
  publishable as a quality annotation in the dataset.
- **Sampled precision audit per phase**: N = 200 random triples manually
  reviewed (mentors + contributor) before the next phase unlocks; publish the
  audited precision with the dataset.
- **Open-domain caveat, stated honestly**: benchmark F1 was measured with
  per-domain ontology bounding; production runs rank over the full DBpedia
  ontology (already supported). Phase 0/1 exist precisely to measure quality in
  the open-domain setting before real money is spent.

## 6. Output and publication

- Expected yield ≈ 2–3 extracted triples/sentence → **≈ 20–30 M triples** for
  the full corpus, ≈ 3–5 GB as N-Triples.
- Published to the **DBpedia Databus as a versioned dataset (triples only)**,
  with per-triple provenance (source sentence ID + verification status);
  regenerable per Wikipedia/DBpedia release cycle.
- The stack remains distributed as the Docker image (GHCR), reproducible from
  this repository + Databus source artifacts.

## 7. Phased plan with decision gates

| Phase | Deliverable | Cost | Gate to proceed |
|---|---|---|---|
| 0 | 100 abstracts through the deployed stack; traced outputs shared for mentor review | < $2 | mentors judge output quality |
| 1 | 100 k abstracts + 200-triple precision audit + open-model benchmark re-run | ≈ $1 k | audited precision acceptable; Option A vs B decision |
| 2 | 1 M abstracts, first Databus release (beta) | ≈ $10 k | community/DBpedia review of beta dataset |
| 3 | Full corpus + versioned Databus release | ≈ $45–55 k (A) or ≈10× less (B) | — |

**The ask to DBpedia**: sponsor Phase 1 (≈ $1 k + review time). Phases 2–3 only
on the strength of Phase 1's audited numbers.

## 8. Risks, honestly

| Risk | Mitigation |
|---|---|
| Open-domain quality below benchmark quality | Phase 0/1 measure it before scale; precision audits gate every phase |
| API provider variance / rate limits | measured at macro ±0.005 on the benchmark; self-hosted Option B removes it entirely |
| Cost estimate drift | unit cost is measured, not modeled; invoices attached per phase |
| Abstract parsing artifacts (junk literals observed in snapshot) | length/character filters (already applied in our sampling); counted into yield estimates |

## Appendix A — traced examples (first Phase-0 batch, run 2026-08-10)

Three abstracts — a person, a place, an organisation — through the **deployed
container** (`nef2` image, real index + store). Raw request/response JSON:
[`examples/phase0_results.json`](examples/phase0_results.json).
Timing: 30–74 s/abstract, consistent with the benchmark's ≈27 s/sentence.

**1 · Person (long-tail, diacritics, dates)** — *"Józef Franciszek Darzyn
Ciemiński (Borzyszkowy, 4 August 1867 – Winona, 1959) was a Polish-born Roman
Catholic priest…"* → 8/8 triples, all gate-approved:

```
Józef_Franciszek_Darzyn_Ciemiński  birthPlace   Borzyszkowy
Józef_Franciszek_Darzyn_Ciemiński  birthDate    1867-08-04
Józef_Franciszek_Darzyn_Ciemiński  deathPlace   Winona
Józef_Franciszek_Darzyn_Ciemiński  deathDate    1959
Józef_Franciszek_Darzyn_Ciemiński  nationality  Polish
Józef_Franciszek_Darzyn_Ciemiński  religion     Roman_Catholic
Józef_Franciszek_Darzyn_Ciemiński  occupation   Roman_Catholic_priest
Józef_Franciszek_Darzyn_Ciemiński  location     Minnesota
```

An obscure entity, diacritics intact, prose date normalised to `1867-08-04` —
the exact profile infobox mappings miss.

**2 · Place** — *"Medamarthy is a village in Srikakulam district of Andhra
Pradesh in India."* → the full containment chain, 4/4 approved:

```
Medamarthy           type      Village
Medamarthy           isPartOf  Srikakulam_district
Srikakulam_district  isPartOf  Andhra_Pradesh
Andhra_Pradesh       country   India
```

**3 · Organisation (honest case)** — *"Winston-Salem State University (WSSU), a
constituent institution of the University of North Carolina…"* → 9 extracted,
2 rejected by the gate, 7 kept — of which 4 are solid
(`isPartOf University_of_North_Carolina`, `location`, `country`, `member`) and
**2–3 are marginal**: open-domain predicate drift produced
`programCost "baccalaureate programs, graduate programs"` and a vacuous
`student "diverse student population"`.

**Reading for reviewers.** Entity linking and place/person relations hold up at
benchmark quality on real abstracts; the known gap is predicate selection
without per-domain ontology bounding (the benchmark supplied bounded predicate
lists; open-domain ranking over the full ontology is looser). This is precisely
what the Phase-1 precision audit measures — and the audit gate exists so that
this class of triple is quantified, filtered (e.g. minimum-informativeness rule
for literal objects), or fixed before any large spend.

## Appendix B — sources

- Workload counts/samples: SPARQL queries in §3 against our snapshot (load
  manifest in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)).
- Unit cost: measured benchmark spend (OpenRouter pricing page, 2026-08;
  invoice to be attached).
- Benchmark result and variance: [REPORT.md](REPORT.md), tag `final-results-0.6317`.
- NEF 1.0 baseline cost: mentor-reported (≈$30–35 full benchmark), exact figure
  to be confirmed by mentors.
