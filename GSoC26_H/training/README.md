<div align="center">

# DBpedia Hindi Chapter

### Fine-tuning a small language model for Hindi triple extraction

[![GSoC](https://img.shields.io/badge/GSoC-2026-8A2BE2?style=flat-square)](https://summerofcode.withgoogle.com/)
[![DBpedia](https://img.shields.io/badge/DBpedia-Hindi%20Chapter-FF6600?style=flat-square)](https://dbpedia.org)
[![Model](https://img.shields.io/badge/Model-Gemma%203%204B-4285F4?style=flat-square)](https://huggingface.co/google/gemma-3-4b-it)
[![Method](https://img.shields.io/badge/Method-QLoRA-00A67E?style=flat-square)](https://arxiv.org/abs/2305.14314)
[![Config](https://img.shields.io/badge/Config-Hydra-89b8cd?style=flat-square)](https://hydra.cc/)
[![Tracking](https://img.shields.io/badge/Tracking-W%26B-FFBE00?style=flat-square)](https://wandb.ai/)

</div>

---

We fine-tune a small language model to read a Hindi sentence and output subject-relation-object triples, aligned to the DBpedia ontology — the same structured knowledge that powers DBpedia's English-language graph, now for Hindi.

In this folder, we provide:

- **A quality-scored training pipeline** — every annotated example is scored 1-10 by an LLM judge against a four-part rubric (source quality, span exactness, semantic correctness, property-relation quality) before it enters the dataset.
- **A DBpedia ontology alignment benchmark** — 112 Hindi sentences (139 triples), hand-verified, used to validate relation-to-property mapping via `e5-large-instruct` embeddings and LLM disambiguation.
- **A QLoRA fine-tuning setup for Gemma 3 4B** — fully configured through Hydra, tracked in Weights & Biases, evaluated on generation quality (pass@1, valid-format rate) rather than loss alone.

---

## Pipeline

```mermaid
flowchart TD
    A["Raw Hindi text<br/>Wikipedia + synthetic sources"] --> B["Triplet annotation<br/>LLM-based extraction"]
    B --> C["Quality scoring<br/>1-10 rubric, LLM-as-judge"]
    C --> D["Train / validation split<br/>by quality score"]
    D --> E["Slug-format conversion<br/>Optimal + Chain-of-Thought"]
    E --> F["QLoRA fine-tuning<br/>Gemma 3 4B"]
    F --> G["Evaluation<br/>pass@1, valid-format rate"]
```

---

## Code Organization

1. `prepare_data.py` — combines all sources, applies the train/validation split, converts to slug format
2. `train.py` — QLoRA fine-tuning entry point; reads Hydra configs, trains, evaluates every 0.25 epoch
3. `evaluate.py` — pass@1 / valid-format-rate evaluation on a saved checkpoint, standalone from training
4. `configs/` — Hydra config groups for model / data / training / logging
5. `scripts/` — launch commands for the smoke test and the two learning-rate runs

---

## Data sources

| Source | Description |
|---|---|
| Synthetic dataset | LLM-generated Hindi sentences with triple annotations |
| Noisy dataset | Additional synthetically generated sentences, mixed quality |
| Wikipedia | Real-world Hindi sentences scraped from Wikipedia articles |

## Annotation

Every sentence is annotated with subject-relation-object triples, including a dedicated `property` relation type for adjectives, possessives, and attributes — following exact-span extraction rules.

## Quality scoring

Each annotated example is scored 1-10 by an LLM judge against four weighted criteria:

| Criterion | Weight |
|:---|:---:|
| Source sentence quality | 20% |
| Span exactness | 30% |
| Semantic correctness of core triplets | 30% |
| Property relation quality | 20% |

## Ontology alignment

A benchmark of **112 Hindi sentences** (**139 triples**) validates relation-to-property alignment. Each extracted relation is embedded with `e5-large-instruct`, matched against the DBpedia ontology by cosine similarity, and disambiguated by an LLM against the sentence context.

## Training data format

Triples are represented in pipe-separated **slug format** rather than JSON — the model is trained to generate this directly, with no post-processing conversion step:

```
subject | relation | object
```

Two trace types are generated per example:

| Trace type | Description |
|---|---|
| **Optimal** | Direct triple output |
| **Chain-of-thought (CoT)** | Step-by-step reasoning, then the triple output |

---

## Fine-tuning — Experiment 1

| Setting | Value |
|:---|:---|
| Base model | Gemma 3 4B |
| Method | QLoRA (4-bit quantized LoRA) |
| Learning rates | `2e-4` and `1e-5` — compared separately |
| Warmup ratio | `0.01` |
| Evaluation frequency | every `0.25` epoch, with checkpointing |
| Validation set | high-scoring Wikipedia sentences |
| Training set | all remaining data |
| Config management | Hydra |
| Experiment tracking | Weights & Biases |

### Running an experiment

```bash
# Build the training/validation files
python3 prepare_data.py

# Smoke test — confirm the pipeline works end to end on a small sample
bash scripts/smoke_test.sh

# Full runs — one per learning rate
bash scripts/train_exp1_lr2e4.sh
bash scripts/train_exp1_lr1e5.sh
```

---

## Folder structure

```
training/
├── README.md
├── prepare_data.py
├── train.py
├── evaluate.py
├── configs/
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   ├── training/
│   └── logging/
└── scripts/
    ├── smoke_test.sh
    ├── train_exp1_lr2e4.sh
    └── train_exp1_lr1e5.sh
```

---

## Methodology references

| Paper | Venue | Contribution |
|---|---|---|
| **KaLLM** | ACL 2024 | Small fine-tuned models outperforming larger general-purpose models on triple extraction |
| **REBEL** | EMNLP 2021 | End-to-end sequence generation instead of pipeline-based extraction |
| **GenIE** | NAACL 2022 | Schema-constrained generation for large-scale ontologies |

---

## Status

- [x] Data collection
- [x] Annotation pipeline
- [x] Quality scoring pipeline
- [x] Ontology alignment benchmark
- [x] Wikipedia scoring
- [ ] Final training/validation set assembled
- [x] Hydra configuration
- [x] Training script
- [ ] Fine-tuning run
- [ ] Evaluation

---

<div align="center">

### Google Summer of Code 2026 — DBpedia
</div>
