# Contributing to the Neural Extraction Framework

Welcome — this guide is written for first-time contributors. It is based on a
fresh setup performed on 27 November 2026 (Python 3.12, Linux, no Docker); every
command below was either run successfully or is marked with what it needs that
this environment did not have. If a step does not work for you, open an issue
and say exactly which command failed and what it printed.

Please be kind and constructive in issues, pull requests and on the forum.
There is no separate `CODE_OF_CONDUCT.md` yet; treat this paragraph as the
standing expectation until the maintainers adopt one.

---

## 1. Prerequisites

| What | Why | Notes |
|---|---|---|
| Python 3.8+ (3.11+ recommended) | All pipeline code is Python | The GSoC26 container uses Python 3.11; this guide was verified on 3.12 |
| Git | Cloning, branching, PRs | |
| Docker + Docker Compose | Running the GSoC26 extraction service | Only needed for the full service, not for docs or offline tests |
| Redis | Surface-form index (GSoC25/26) | Only needed for index-backed runs/tests |
| Java | `fact-extractor` (separate repo) | Not needed for this repo |
| An LLM API key | The pipeline's only non-local, paid component | GSoC26: `OPENROUTER_API_KEY`; GSoC25: `GEMINI_API_KEY` |

---

## 2. Fork, branch, pull request

```bash
# 1. Fork https://github.com/dbpedia/neural-extraction-framework on GitHub,
#    then clone YOUR fork:
git clone https://github.com/<your-username>/neural-extraction-framework.git
cd neural-extraction-framework

# 2. Create a feature branch (one branch per issue / change):
git checkout -b docs/onboarding-guide

# 3. Make your changes, then commit and push:
git add <changed files>
git commit -m "Docs: add onboarding guide (fixes #50)"
git push origin docs/onboarding-guide

# 4. Open a Pull Request against dbpedia:main from your branch on GitHub.
#    In the PR description, link the issue it fixes, e.g. "Fixes #50",
#    and say how you tested the change.
```

Keep PRs focused: one issue per PR, no unrelated reformatting. If you are
unsure whether an approach is wanted, comment on the issue first — maintainers
reply on GitHub and on the [forum](https://forum.dbpedia.org/).

---

## 3. Base setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: the **root** `requirements.txt` only installs `streamlit`. Every GSoC
project folder carries its own dependencies — always install the folder's
requirements (or follow its Dockerfile) instead of assuming the root file is
enough:

| Folder | What it is | Setup |
|---|---|---|
| `GSoC26/` | NEF 2.0 — the current pipeline (LLM + local surface-form index + SPARQL verification gate) | `GSoC26/README.md` + `GSoC26/deploy/DEPLOY.md`; deps from `GSoC26/deploy/Dockerfile` (see §4) |
| `GSoC25/` | English IE pipeline (2025) | `GSoC25/Readme.md`; `pip install -r GSoC25/requirements.txt`; needs `GEMINI_API_KEY` + Redis |
| `GSoC25_H/` | Hindi IE pipeline (2025) | `GSoC25_H/requirements.txt` |
| `GSoC26_H/` | Hindi chapter work (2026) | `GSoC26_H/requirements.txt` |
| `GSoC24/`, `GSoC23/`, `GSoC21/` | Earlier pipelines — reference / prior work | Each folder has its own README + `requirements.txt` |

### Installing the GSoC26 dependencies (verified)

The GSoC26 pipeline has no `requirements.txt` of its own; its dependency list
lives in `GSoC26/deploy/Dockerfile`:

```bash
pip install rdflib redis pydantic requests numpy \
    langchain-core langchain-openai langgraph \
    sentence-transformers rank-bm25 fastapi uvicorn \
    scikit-learn tqdm
```

Two things that bite on a fresh machine:

- **Torch is big.** `sentence-transformers` pulls in PyTorch (~2 GB of CUDA
  wheels). On a machine without a GPU, install the CPU build first — it is
  much smaller and is all the pipeline needs:
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cpu torch
  pip install sentence-transformers
  ```
- **pip's temp dir.** pip unpacks wheels under `$TMPDIR` (usually `/tmp`,
  which is often a small tmpfs). If a torch install fails with
  `OSError: [Errno 28] No space left on device` even though your disk has
  room, point `TMPDIR` at a directory on disk first:
  ```bash
  mkdir -p .tmp && TMPDIR=$PWD/.tmp pip install sentence-transformers
  ```

---

## 4. Running the GSoC26 pipeline

### As a service (Docker — the supported path)

See [`GSoC26/deploy/DEPLOY.md`](GSoC26/deploy/DEPLOY.md). In short, three
containers (Redis index, oxigraph store, pipeline API) behind one endpoint.
Before starting the stack:

1. Put the data snapshots in `deploy/data/` (`dump.rdb` + `oxigraph/`),
   published on the DBpedia Databus (links in DEPLOY.md).
2. Set `OPENROUTER_API_KEY` in `.env`.

Then start the stack — run `./setup.sh` once to seed the Docker volumes:

```bash
cd GSoC26/deploy
./setup.sh
docker compose up -d
```

Once the stack is up, send a test extraction request:

```bash
curl -X POST localhost:8000/extract -H "Content-Type: application/json" \
  -d '{"text": "The song Mermaid is by the band Train."}'
# -> {"triples": [{"sub": "Mermaid_(Train_song)", "rel": "artist",
#                  "obj": "Train_(band)"}], ...}
```

Requirements for the full stack: ≥6 GB RAM (Redis holds the ~3 GB index in
memory), the ~1.4 GB `dump.rdb` snapshot, the oxigraph store, and an
OpenRouter key. The only external call the stack makes is to the LLM.

### Smoke test without any local setup

You can check the two public services the pipeline talks to with plain curl —
useful before you install anything:

```bash
# SPARQL endpoint — ask for Berlin's English label:
curl "https://dbpedia.org/sparql?query=SELECT%20%3Flabel%20WHERE%20%7B%20%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2FBerlin%3E%20rdfs%3Alabel%20%3Flabel%20.%20FILTER%20(lang(%3Flabel)%20%3D%20%27en%27)%20%7D&format=application%2Fsparql-results%2Bjson"

# Entity Lookup — resolve the string "Berlin" to a DBpedia resource:
curl "https://lookup.dbpedia.org/api/search?query=Berlin&format=json"
```

---

## 5. Running the tests

There is no repo-wide test runner yet — tests live next to the code they
cover, and each folder's tests have different requirements. The GSoC26 tests
are the best starting point:

```bash
cd GSoC26/tests
USE_TF=0 PYTHONPATH=../src python3 test_gate_patches.py
```

| Test | Needs | Status (verified 27 Nov 2026) |
|---|---|---|
| GSoC26/tests/test_gate_patches.py | Python deps (§3); all-MiniLM-L6-v2 must be cached or downloadable; no LLM, no Redis | ✅ 10/10 passed |
| `GSoC26/tests/test_date_style.py` | Above + a `Text2KGBench` checkout (`git clone https://github.com/cenguix/Text2KGBench.git`, then point the harness at `Text2KGBench/data/dbpedia_webnlg`) | ✅ **12/12 passed** against a real Text2KGBench checkout |
| `GSoC26/tests/test_index.py` | A live Redis on port 6380 with the 16.2M-entry surface-form index imported (`GSoC26/scripts/import_index.py`, ~10 min, ~2.6 GiB RAM — see `GSoC26/docs/RUNBOOK.md`) | Not runnable without the index; expect 4/5 — the Arion case fails by design (a WebNLG artifact URI) |
| `GSoC25/Data/test_collector_functions.py`, `GSoC25/EntityLinking/*`, `GSoC25/RelationExtraction/*`, `GSoC25/TestFiles/*` | GSoC25 deps + Redis and (for integration tests) API keys | Per-folder; run from inside `GSoC25/` with its requirements installed |

### What to expect on first run

- **First import downloads the embedding model.** Importing
  `autonomous_pipeline_v13` loads `SentenceTransformer('all-MiniLM-L6-v2')`
  (~90 MB from HuggingFace) on first use. This is normal; later runs reuse
  the cache.
- **`OPENROUTER_API_KEY not set` warning is harmless for offline tests.**
  The key is only needed for actual LLM calls.
- **Redis auto-boot fails gracefully.** If `redis-server` is not installed
  you will see `Redis offline ... Attempting Auto-Boot ... Auto-Boot Error`.
  Offline tests continue without it; index-backed tests will not work.
- **Run tests from the right directory.** The pipeline resolves
  `dbpedia.owl` relative to the current working directory (see §6) — the
  offline gate tests do not need the ontology, but anything that does must be
  run from `GSoC26/data/` or with the file copied next to your working
  directory.

---

## 6. Known gotchas (found while setting up — please fix, don't work around silently)

- **`dbpedia.owl` is only found when the working directory contains it**
  (issue [#76](https://github.com/dbpedia/neural-extraction-framework/issues/76)).
  The ontology snapshot ships at `GSoC26/data/dbpedia.owl`, but
  `load_full_dbpedia_ontology()` defaults to a bare relative path, so running
  from anywhere else prints `dbpedia.owl not found. Ontology grounding
  DISABLED` and predicate linking silently falls back to naive slugification.
  Workaround for now: `cd GSoC26/data` or copy `dbpedia.owl` into your
  working directory. The real fix is a `Path(__file__)`-based lookup.
- **No `requirements.txt` in `GSoC26/`** — the dependency list lives only in
  `GSoC26/deploy/Dockerfile` (§3 above). If you add dependencies, update the
  Dockerfile and this guide together.
- **GSoC25 docs use notebook syntax.** `GSoC25/Readme.md` shows commands with
  a leading `!` (`!pip install ...`) — drop the `!` when running them in a
  terminal.
- **Folder README filenames differ by year** (`README.md` vs `Readme.md` vs
  `ReadME.md`) — check the folder listing rather than guessing the name.

---

## 7. Where to ask questions

- **Forum (primary):** [forum.dbpedia.org](https://forum.dbpedia.org/) —
  introduce yourself in the project thread, don't open a new one. For this
  repo: the [Neural Extraction Framework GSoC 2026 thread](https://forum.dbpedia.org/t/towards-a-neural-extraction-framework-gsoc-2026/4778)
  and the [Agentic QA thread](https://forum.dbpedia.org/t/agentic-question-answering-over-dbpedia-gsoc-2026/4968).
- **Slack:** [dbpedia.slack.com](https://dbpedia.slack.com/)
- **Mailing list:** dbpedia-gsoc@lists.dbpedia.org
- **GSoC ideas:** [forum.dbpedia.org/c/projects/gsoc/8](https://forum.dbpedia.org/c/projects/gsoc/8)

Show warm-up work before asking for a project: the contributors who got
mentor attention posted concrete progress (a pipeline audit, a prototype, a
first PR). Ask specific questions that reference the exact file or function
you are stuck on.

---

## 8. For GSoC applicants

- Read the [GSoC ideas list](https://forum.dbpedia.org/c/projects/gsoc/8)
  and the linked warm-up tasks before picking a project.
- The established proposal loop: draft your proposal as a Google Doc in the
  format of the accepted-proposal example linked in the idea thread, then
  invite the mentor as an editor for feedback well before the deadline
  (mid-March, not the last week).
- Your contribution history on GitHub and the forum is what mentors look at
  first — small merged PRs (docs, tests, small bug fixes) count.
