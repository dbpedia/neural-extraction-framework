# Deploying the extraction service

A fully local text→DBpedia-triples extraction stack in three containers:

| Service | What it does | Data it needs |
|---|---|---|
| `redis` | Surface-form index: "which entity can this text mean?" (16.2M mentions, ~1 ms lookups) | `dump.rdb` (~1.4 GB) |
| `oxigraph` | Local DBpedia triple store: verifies extracted facts in milliseconds | oxigraph data dir |
| `pipeline` | The extraction pipeline behind `POST /extract` | — |

The only external call the stack ever makes is to the LLM. Everything else —
candidate lookup, fact verification, formatting — is local. Your text never
leaves the host except for the LLM request.

## Quick start

```bash
git clone <repo> && cd <repo>/deploy

# 1. Get the two data snapshots into deploy/data/
#      data/dump.rdb        <- surface-form index snapshot
#      data/oxigraph/       <- triple-store snapshot
#    Published on the DBpedia Databus (artifact URIs added on release):
#      https://databus.dbpedia.org/<account>/surface-form-index/<version>
#      https://databus.dbpedia.org/<account>/extraction-store/<version>
#    (or copy them from your build machine)

# 2. Seed the Docker volumes (one time)
./setup.sh

# 3. Configure the LLM
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# 4. Up
docker compose up
```

First start takes ~3 minutes: Redis loads the index dump into memory, and the
compose healthcheck holds the pipeline back until the index is queryable.

## Use it

```bash
curl -X POST localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "The song Mermaid is by the band Train."}'
```

Every returned triple carries its verification status — `verified` (the fact
exists in DBpedia) or faithful-unverified (asserted by the sentence but absent
from DBpedia) — so consumers can filter by trust level.

## Choosing the LLM

The pipeline speaks the OpenAI-compatible API, so the model is a config choice:

```bash
# .env
TARGET_MODEL=openai/gpt-5.6-luna          # benchmark configuration (default)
# Any OpenRouter model id works. For a 100% offline deployment, point
# OPENROUTER_BASE_URL at a local vLLM/Ollama server and use its model name.
```

Benchmark reference (Text2KGBench dbpedia_webnlg, 19 domains): macro F1 0.6317
with gpt-5.6-luna at ≈$0.005/sentence — on par with the published GPT-4o
baseline (0.628) at ~1/20th the cost. See `REPRODUCIBILITY.md`.

## Resource requirements

- **RAM:** ≥6 GB for the stack (Redis holds the index in memory: ~3 GB; the
  pipeline loads an embedding model: ~1 GB). 8 GB host minimum recommended.
- **Disk:** index snapshot 1.4 GB + oxigraph store size.
- Run Redis only via this compose file (never as a pipeline child process),
  and never run a second Redis on the host — measured failure mode: memory
  exhaustion degrades extraction quality via per-sentence timeouts.

## Prebuilt image

No build needed — the image is published:

```bash
docker pull ghcr.io/nakulsingh156/neural-extraction-framework:v2.0
```

**Different API port** (e.g. 8000 is taken on your host): append a uvicorn
command to the `docker run`, or override `command:` in compose:

```bash
docker run ... neural-extraction-framework:v2.0 uvicorn serve:app --host 0.0.0.0 --port 8010
```

## Operational notes

- `redis` runs with `--save ""` — the index is read-only at runtime and can
  never be overwritten by a background save.
- `oxigraph` runs `serve-read-only` — the store cannot be mutated via HTTP.
- Health: `GET localhost:8000/health` returns `{ok, model}`.
