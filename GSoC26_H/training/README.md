# Training — DBpedia Hindi Chapter

**Google Summer of Code 2026 — DBpedia**

This folder contains the training pipeline for the Hindi triple-extraction model: data preparation, Hydra configuration, QLoRA fine-tuning, and evaluation.

---

## Overview

This pipeline teaches a small language model to extract subject-relation-object triples from Hindi sentences and align them to the DBpedia ontology — the same kind of triples that power DBpedia's English-language knowledge graph, but for Hindi.

The pipeline covers four stages:

1. **Data generation** — collecting and annotating Hindi sentences with subject-relation-object triples
2. **Quality scoring** — an LLM-as-judge rubric that scores every annotated example for span exactness, semantic correctness, and property-relation quality
3. **Ontology alignment** — mapping extracted relations to real DBpedia (`dbo:`) properties, validated against a manually curated benchmark
4. **Fine-tuning** — training a QLoRA-adapted small language model on the resulting dataset, tracked and evaluated systematically

---

## Pipeline

```
Raw Hindi text (Wikipedia + synthetic sources)
        |
        v
Triplet annotation (LLM-based extraction)
        |
        v
Quality scoring (1-10 rubric, LLM-as-judge)
        |
        v
Train / validation split (by quality score)
        |
        v
Slug-format conversion (Optimal + Chain-of-Thought traces)
        |
        v
QLoRA fine-tuning (Gemma 3 4B)
        |
        v
Evaluation (pass@1, valid-format rate, BenchIE ground truth)
```

### 1. Data sources

| Source | Description |
|---|---|
| Synthetic dataset | LLM-generated Hindi sentences with triple annotations |
| Noisy dataset | Additional synthetically generated sentences, mixed quality |
| Wikipedia | Real-world Hindi sentences scraped from Wikipedia articles |

### 2. Annotation

Every sentence is annotated with subject-relation-object triples, including a dedicated `property` relation type for adjectives, possessives, and attributes — following exact-span extraction rules (no paraphrasing, no dropped postpositions).

### 3. Quality scoring

Each annotated example is scored 1-10 by an LLM judge against four weighted criteria:

| Criterion | Weight |
|---|---|
| Source sentence quality | 20% |
| Span exactness | 30% |
| Semantic correctness of core triplets | 30% |
| Property relation quality | 20% |

### 4. Ontology alignment (ground truth)

A benchmark of 112 Hindi sentences (139 triples) is used to validate relation-to-property alignment: each extracted relation is embedded with `e5-large-instruct`, matched against the DBpedia ontology by cosine similarity, and disambiguated by an LLM against the sentence context.

### 5. Training data format

Triples are represented in pipe-separated **slug format** rather than JSON:

```
subject | relation | object
```

Two trace types are generated per example:
- **Optimal** — direct triple output
- **Chain-of-thought (CoT)** — step-by-step reasoning followed by the triple output

### 6. Fine-tuning — Experiment 1

| Setting | Value |
|---|---|
| Base model | Gemma 3 4B |
| Method | QLoRA (4-bit quantized LoRA) |
| Learning rates | `2e-4` and `1e-5` (compared separately) |
| Warmup ratio | 0.01 |
| Evaluation frequency | every 0.25 epoch, with checkpointing |
| Validation set | high-scoring Wikipedia sentences |
| Training set | all remaining data |
| Config management | Hydra |
| Experiment tracking | Weights & Biases |

---

## Folder structure

```
training/
├── README.md
├── configs/            # Hydra configuration groups
│   ├── config.yaml
│   ├── model/
│   ├── data/
│   ├── training/
│   └── logging/
├── prepare_data.py     # builds the train/validation split
├── train.py             # QLoRA fine-tuning entry point
└── evaluate.py           # pass@1 / valid-format-rate evaluation
```

---

## Methodology references

This project's approach — pipe-separated slug output, chain-of-thought augmentation, and LoRA fine-tuning of small models for triple extraction — draws on:

- **KaLLM** (ACL 2024) — fine-tuned small models outperforming larger general-purpose models on triple extraction
- **REBEL** (EMNLP 2021) — end-to-end sequence-to-sequence relation extraction
- **GenIE** (NAACL 2022) — schema-constrained generative information extraction

---

## Acknowledgments

**Google Summer of Code 2026 — DBpedia**
