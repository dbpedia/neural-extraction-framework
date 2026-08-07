# DBpedia Hindi Chapter — Neural Relational Triple Extraction

> Building a Hindi knowledge graph from Wikipedia, one sentence at a time.

[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=flat-square)](https://github.com/singhhnitin/neural-extraction-framework/blob/gsoc26h-development/GSoC26_H)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![DBpedia](https://img.shields.io/badge/DBpedia-Hindi%20Chapter-0066CC?style=flat-square)](https://www.dbpedia.org/)
[![Model](https://img.shields.io/badge/Extraction-Gemma%203%204B-4285F4?style=flat-square)](https://huggingface.co/google/gemma-3-4b-it)
[![Model](https://img.shields.io/badge/Normalization-F2LLM--1.7B-00A67E?style=flat-square)](https://github.com/singhhnitin/neural-extraction-framework/blob/gsoc26h-development/GSoC26_H)

---

## The Problem

Hindi is spoken by over 600 million people, and Hindi Wikipedia holds vast amounts of factual knowledge — but almost all of it sits in free text, not structured infoboxes. DBpedia's Hindi chapter is sparse as a direct result: a machine looking for a clean table finds nothing, even when a fact is stated plainly in a sentence.

```
ताजमहल का निर्माण शाहजहाँ ने करवाया था।
(The Taj Mahal was built by Shah Jahan.)
```

A human reads this and knows: **Taj Mahal — builder → Shah Jahan**. A machine needs that fact turned into a structured triple before it can query, reason over, or link it to anything else.

## What This Project Does

An end-to-end pipeline that reads a raw Hindi sentence and outputs a DBpedia-compatible triple, ready for the knowledge graph:

```
Hindi sentence → Extraction → Normalization → Human review → Knowledge graph
```

For the example above, the pipeline produces `Taj_Mahal | dbo:builder | Shah_Jahan`.

---

## Published Models & Datasets

All models and datasets generated this GSoC are published on Hugging Face, publicly accessible.

### Models

| Model | Description |
|---|---|
| [dbpedia-hindi-gemma3-4b-lr2e4](https://huggingface.co/Nitin1211/dbpedia-hindi-gemma3-4b-lr2e4) | Extraction model, lr=2e-4 (**selected, recommended**) |
| [dbpedia-hindi-gemma3-4b-lr1e5](https://huggingface.co/Nitin1211/dbpedia-hindi-gemma3-4b-lr1e5) | Extraction model, lr=1e-5 (comparison run) |

### Datasets

| Dataset | Description |
|---|---|
| [dbpedia-hindi-training-data](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-training-data) | 39,621-example training set (Optimal-trace) |
| [dbpedia-hindi-validation-data](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-validation-data) | 3,634-example validation set (real Wikipedia) |
| [dbpedia-hindi-predicate-linking-gold](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-predicate-linking-gold) | 8,029-entry predicate → DBpedia property gold set |
| [dbpedia-hindi-benchie-ground-truth](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-benchie-ground-truth) | 139-triple BenchIE DBpedia ground truth |
| [dbpedia-hindi-noisy-training-data](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-noisy-training-data) | 15,581-example noisy synthetic set (curriculum training) |
| [dbpedia-hindi-cot-training-data](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-cot-training-data) | 39,621-example Chain-of-Thought traces (generated, **not used** in final training — see dataset card) |

---

## Pipeline

```
Hindi sentence
   │
   ▼
[1] EXTRACTION — Gemma 3 4B, QLoRA fine-tuned (lr=2e-4)
    Reads the full sentence, outputs every (subject, relation, object)
    triple directly — no separate rule-based argument-extraction step.
    Every triple is also tagged as either a real relation, or "property"
    (a non-relational fragment like an adjective+noun pair).
   │
   ├──► "property"-type triples ──► set aside for dedicated review
   │
   ▼
[2] NORMALIZATION — F2LLM-1.7B, QLoRA fine-tuned, + GPT-OSS-120B fallback
    Each real relation is checked against a cache of already-resolved
    predicates first. If not cached, F2LLM-1.7B retrieves the top-k
    candidate DBpedia properties by embedding similarity, and
    GPT-OSS-120B disambiguates between them using the sentence context.
   │
   ▼
[3] HUMAN REVIEW (HITL) — Streamlit app, password-protected
    A reviewer sees the sentence, the extracted triple, and the
    suggested DBpedia property, and can Accept / Modify / Reject —
    including selecting the correct subject/object directly from
    dropdown options populated with words from the sentence, and
    choosing from the real top-40 DBpedia property candidates
    (precomputed per predicate) rather than a generic list.
   │
   ▼
[4] FEEDBACK LOOP — merge_hitl_feedback.py
    Pulls confirmed corrections and folds them back into the
    normalization cache and future training data.
```

**A deliberate design choice worth stating plainly:** normalization does not use a fixed confidence-score cutoff to decide what needs human review. The design checks simply whether a DBpedia property was found at all — every real relation gets a chance at review, maximizing reviewable coverage.

---

## Development Journey — The Full Process

This section walks through the complete process, start to finish, in the order it actually happened.

### 1. Embedding Model Selection

Ontology alignment began with a curated set of 73 DBpedia properties paired with a multilingual embedding model (MiniLM), establishing a solid initial baseline. A newer embedding model (KaLM-Embedding) was evaluated as a potential upgrade across four different configurations; MiniLM held up as the stronger, more dependable choice within that curated scope.

For the broader predicate-linking task against the full DBpedia ontology, three models were compared directly: **F2LLM-1.7B**, **F2LLM-500M**, and **e5-large-instruct**. F2LLM-1.7B came out ahead at every precision level tested and was selected as the model to fine-tune going forward.

### 2. Synthetic Dataset Generation

An initial synthetic training dataset was generated using an LLM, producing Hindi sentences paired with subject-relation-object triple annotations. A second, deliberately "noisy" dataset (15,581 examples, [published](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-noisy-training-data)) was generated separately — seeded from lower-scoring originals, using a mix of `openai/gpt-oss-120b` (86.1%) and `meta/llama-3.2-3b-instruct` (13.9%) — to give the model realistic variation and extraction mistakes to learn from, supporting a staged (curriculum-style) training approach.

### 3. Wikipedia Scraping

Real Hindi Wikipedia articles were scraped and filtered down to sentences matching the BenchIE benchmark's natural sentence-length distribution (median ~15 words, range 10–23), grounding the training data in genuine, naturally-occurring Hindi text rather than model-generated sentences alone.

### 4. Scoring

Every candidate sentence — synthetic and real Wikipedia alike — was scored 1–10 by an LLM judge against a weighted quality rubric (source sentence quality, span exactness, semantic correctness, property-relation quality). An early version of the scoring prompt undervalued genuine Wikipedia sentences relative to synthetic ones; refining it with Hindi-specific few-shot examples and explicit property-density weighting raised the rate of high-quality (score ≥9) sentences from 8.5% to 68.8%.

### 5. Train / Validation Split

Sentences scoring 9 or above were held out as the validation set; everything else became the training set. Pronouns appearing in core triplets were resolved to their referenced entities via an LLM-based coreference pass (`coref_resolve.py`) before finalizing the splits, with a refined recovery rule added for "self-contained" cases (e.g. a demonstrative pronoun paired with a noun) to retain otherwise-recoverable examples.

**Final splits (also [published on Hugging Face](#published-models--datasets)):**

| Dataset            | Size                 | Purpose                                                           |
| ------------------- | -------------------- | ------------------------------------------------------------------ |
| Training set        | **39,621** examples  | Fine-tuning Gemma 3 4B (extraction), Optimal-trace format          |
| Validation set       | **3,634** examples    | High-scoring (≥9) Wikipedia sentences, held out during training    |
| Held-out test set    | **585** examples       | Never used in training; source of the precision@k numbers below    |

A Chain-of-Thought variant of the training set was also generated ([published](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-cot-training-data)) but not used in final training — determined unnecessary at this training stage.

### 6. Extraction Fine-Tuning — Two Learning Rates

Gemma 3 4B was fine-tuned via QLoRA (4-bit quantized LoRA) at two learning rates, compared directly against each other. Both checkpoints are [published on Hugging Face](#published-models--datasets):

| Learning rate       | Wikipedia F1 | Train F1  |
| ------------------- | ------------ | --------- |
| **2e-4 (selected)** | **0.692**    | **0.795** |
| 1e-5                | 0.613        | 0.471     |

valid_format_rate: 98–100%. Evaluation combined LLM-as-judge scoring with triple-level precision/recall/F1, on the held-out validation set described above.

### 7. Predicate-Linking Gold Set Construction (F2LLM-8B + GPT-OSS-120B)

To fine-tune the lightweight normalization model (F2LLM-1.7B), a gold-standard set of predicate-to-DBpedia-property mappings ([published](https://huggingface.co/datasets/Nitin1211/dbpedia-hindi-predicate-linking-gold)) was built using a two-stage pipeline across 8,029 unique predicates:

1. **Retrieval — F2LLM-8B.** For each predicate, F2LLM-8B retrieved the top candidate DBpedia properties by embedding similarity against the full ontology.
2. **Disambiguation — GPT-OSS-120B.** The retrieved candidates, together with full sentence context, were passed to GPT-OSS-120B to select the correct property, or confirm that no real DBpedia property applies.

This produced 5,855 confirmed real DBO mappings and 2,174 initially unmatched predicates. A later recovery pass — searching ranks 51–100 instead of stopping earlier — recovered 1,249 of those (57.5%), raising total gold-set coverage to 88.4%.

### 8. Normalization Fine-Tuning — F2LLM-1.7B on the Gold Set

F2LLM-1.7B was fine-tuned via QLoRA on the gold set above, across two rounds (3 epochs, then 6 more for 9 total). Every checkpoint was re-evaluated using a single, consistent encoding method (SentenceTransformer) to ensure genuinely comparable results across the training arc. A parallel plain-LoRA run confirmed QLoRA as the stronger approach at every precision threshold.

Evaluated on the true 585-item held-out set — checking whether the correct DBpedia property appears in the top-k candidates:

| k      | Precision@k |
| ------ | ----------- |
| 1      | 37.6%       |
| 5      | 72.1%       |
| 10     | 78.8%       |
| 20     | 84.1%       |
| 30     | 87.7%       |
| **40** | **89.9%**   |

QLoRA outperformed plain LoRA by 0.5–1.5 percentage points at every k (e.g. p@1: 37.6% vs 36.2%).

### 9. Human-in-the-Loop Interface

Built as a Streamlit app supporting Accept / Modify / Reject decisions with confidence-coded badges and a structured error taxonomy, then connected directly to real pipeline output. The interface includes stable per-triple IDs (so reviewed items persist correctly across reloads), password protection, dropdown-based subject/object correction (options constrained to words actually present in the sentence), a real top-40 DBpedia property candidate list per predicate (precomputed via the same F2LLM-1.7B retrieval used in production, not a generic fixed list), and automatic syncing of every decision straight to GitHub.

### 10. Full-Scale Evaluation

Complete pipeline — extraction → normalization → final property — run across the entire evaluation set:

| Source    | Precision | Recall | F1        | N     |
| --------- | --------- | ------ | --------- | ----- |
| Wikipedia | 0.498     | 0.493  | **0.493** | 1,817 |
| Train     | 0.550     | 0.530  | **0.537** | 50    |
| BenchIE   | 0.192     | 0.165  | **0.173** | 112   |

Total predicates normalized: 4,967. BenchIE evaluates the pipeline against an independent, out-of-domain benchmark (ground truth constructed via the same top-k + LLM disambiguation approach as the gold set above), providing a valuable real-world signal alongside the in-domain Wikipedia and Train results.

The Wikipedia figure above reflects a rerun (`rerun_wikipedia_f1_clean.py`) excluding 79 sentences found to be corrupted by leftover coreference-resolution artifacts, using the fully-updated normalization cache.

**Extraction-only vs. full-pipeline (with predicate linking):**

| Source    | Extraction-only F1 | Full-pipeline F1 |
| --------- | ------------------- | ------------------ |
| Wikipedia | 0.554                | 0.493               |
| Train     | 0.610                | 0.537               |
| BenchIE   | 0.056                | 0.173               |

BenchIE's extraction-only F1 is notably lower than its full-pipeline F1 — the opposite pattern from Wikipedia/Train. Manual inspection found the model consistently trims Hindi postpositions (का/की/में) that BenchIE's gold spans retain, causing exact-match failures despite the core relation being correctly identified in nearly every case checked. Predicate linking, which matches at the DBpedia-property level rather than exact text, is unaffected by this and recovers the real matches.

**Baselines, for comparison (extraction-only F1):**

| System                              | Wikipedia | Train    | BenchIE |
| ------------------------------------- | --------- | -------- | ------- |
| Fine-tuned Gemma 3 4B (zero-shot)      | 0.554     | 0.610    | 0.056   |
| Base Gemma 3 4B (untrained)            | 0.024     | 0.014    | 0.016   |
| IndIE (external, rule-based)           | 0.021     | 0.008\*  | 0.076   |

\*IndIE's Train figure uses the full 39,621-sentence training set, not the 50-sample subset used elsewhere — not directly comparable to the other rows. Fine-tuning improved Wikipedia F1 by roughly 23x and Train F1 by roughly 44x over the untrained base model.

### 11. Data Quality Auditing

A dedicated audit pipeline verifies data integrity end to end: detecting and filtering corrupted sentences (leftover artifacts from the coreference-resolution step), and independently auditing every triple labeled "property" (non-relational) using GPT-OSS-120B to confirm that classification is correct.

**Property-triple audit — final results**, across all 6,189 audited property-type triples:

| Category                                          | Count | %     |
| --------------------------------------------------- | ----- | ----- |
| Genuine (correctly non-relational)                   | 3,974 | 64.2% |
| **Mislabeled (real relation, wrongly discarded)**    | **1,837** | **29.7%** |
| Errors                                                | 378   | 6.1%  |

Nearly a third of triples labeled "property" (non-relational) were actually genuine relational facts that never reached DBpedia matching. A real bug was found and fixed during this audit: the LLM judge's response was being truncated before reaching its final verdict (too low a token limit for a reasoning-style model), and the verdict-parsing logic was checking the start of the response rather than its end — together causing an earlier run to silently report 0% mislabeled. Both issues are fixed in the current script.

### 12. Project Explainer

An interactive, stage-by-stage walkthrough of the full project — architecture, results, and the human review process — was built and deployed as a public-facing page for sharing with mentors and the DBpedia community.

### 13. Cross-Lingual Robustness Test

To check the pipeline's behavior on Indic languages it was never trained on, the extraction model was run on 50 real Gujarati sentences (sourced from Gujarati Wikipedia) and 50 Rajasthani sentences (constructed using verified real Rajasthani grammatical markers, given the lack of a substantial standalone Rajasthani text corpus).

**Result: 100/100 sentences produced output, zero crashes.** As expected, output quality is poor — the model attempts to force Gujarati/Rajasthani sentence structure into Hindi patterns, frequently code-mixing scripts within a single triple and substituting a Hindi verb for the source sentence's actual verb. This is consistent with a model fine-tuned exclusively on Hindi: it generalizes the *task* (find subject-relation-object structure) but not the *language*. The pipeline is robust — it does not fail on unfamiliar input — but is genuinely Hindi-specific, not incidentally multilingual.

---

## Human-in-the-Loop Review

Live at a password-protected Streamlit app, synced directly to GitHub. The review queue surfaces triples with a suggested DBpedia property for confirmation, correction, or rejection, with dropdown-based subject/object correction and real top-40 property candidates per predicate.

Review data (`hitl_corrections.jsonl`) is stored with both the original extracted subject/object and the reviewer's corrected version, preserving full traceability of every correction.

---

## Repository Structure

```
GSoC26_H/
├── README.md
├── requirements.txt
├── configs/
│   └── lora_config.yaml
│
├── training/                QLoRA fine-tuning for both models
│   ├── README.md                      Detailed training methodology, data pipeline, quality scoring
│   ├── train.py                       Gemma 3 4B extraction fine-tuning entry point
│   ├── prepare_data.py                Builds train/validation splits
│   ├── evaluate.py                    Standalone checkpoint evaluation
│   ├── finetune_f2lm.py               F2LLM-1.7B QLoRA fine-tuning
│   ├── finetune_f2lm_lora_only.py     F2LLM-1.7B plain LoRA (comparison run)
│   ├── merge_lora_only.py             Merges LoRA adapter into base weights
│   ├── build_gold_set_chunk0.py       Predicate-linking gold set construction (chunk 1/2)
│   ├── build_gold_set_chunk1.py       Predicate-linking gold set construction (chunk 2/2)
│   ├── embed_catalog_f2lm.py          Pre-computes DBpedia property catalog embeddings
│   ├── evaluate_150_lr1e5.py          150-sample eval for the lr=1e-5 checkpoint
│   ├── evaluate_150_mixed.py          150-sample mixed-source eval
│   ├── evaluate_checkpoints.py        Compares checkpoints across training
│   ├── peek_predictions.py            Quick manual inspection of model outputs
│   ├── configs/                       Hydra configs (model / data / training / logging)
│   └── scripts/
│       ├── smoke_test.sh              End-to-end pipeline verification on a small sample
│       ├── train_exp1_lr2e4.sh        Full training run, lr=2e-4 (selected)
│       └── train_exp1_lr1e5.sh        Full training run, lr=1e-5 (comparison)
│
├── inference/                Running the trained models at scale
│   ├── evaluate_full_scale.py         Full-scale zero-shot extraction (1,979 sentences)
│   ├── evaluate_few_shot.py           Full-scale few-shot extraction
│   ├── normalize_full_scale.py        Full-scale predicate normalization
│   ├── normalize_triples.py           Standalone normalization utility
│   └── retry_none_predicates.py       NONE-predicate recovery (ranks 51-100)
│
├── evaluation/                Held-out and checkpoint comparison evaluation
│   ├── evaluate_held_out.py           Precision@k on the true held-out set
│   ├── evaluate_all_checkpoints.py    Consistent re-evaluation across all checkpoints
│   ├── evaluate_finetuned_fair.py     Fair QLoRA-vs-LoRA comparison with consistent encoding
│   └── rerun_wikipedia_f1_clean.py    Recomputes Wikipedia F1 excluding corrupted sentences
│
├── data_quality/              Data integrity tooling
│   ├── check_all_property_triples.py  Full audit of "property"-type triples
│   ├── filter_hitl_corrupted.py       Removes corrupted sentences from HITL data
│   ├── filter_optimal_only.py         Filters training traces to optimal-only
│   ├── scan_corrupted_sentences.py    Detects coreference-reasoning leakage in sentences
│   ├── scan_corrupted_v2.py           Refined corrupted-sentence detection
│   └── scoring_and_coref/
│       ├── coref_resolve.py           LLM-based coreference resolution pass
│       ├── score_chunk.py             LLM-as-judge quality scoring (chunked)
│       └── score_wikipedia.py         Wikipedia sentence quality scoring
│
├── hitl/                      Human review interface
│   ├── README.md
│   ├── hitl_app.py                    Streamlit review app
│   ├── generate_hitl_data.py          Builds the review queue from pipeline output
│   ├── merge_hitl_feedback.py         Folds corrections back into training data
│   └── requirement.txt                HITL-specific dependencies
│
├── data/                      Ontology reference and ground truth
│   ├── ontology/
│   │   └── dbpedia_properties.json    DBpedia property catalog
│   ├── ground_truth/
│   │   └── ground_truth_benchie_triples.jsonl   BenchIE gold triples
│   └── wikipedia_synthetic_data_clean.jsonl     Cleaned Wikipedia annotation data
│
├── results/                   Evaluation outputs and summaries
│   ├── alignment_results_full_20k.jsonl    Full-scale normalized triples (HITL source data)
│   ├── hitl_corrections.jsonl              Human review decisions
│   ├── hitl_top40_candidates.json          Precomputed real top-40 DBO candidates per predicate
│   ├── embedding_model_comparison.md       F2LLM vs e5-large-instruct comparison
│   ├── ground_truth_summary.md             BenchIE ground-truth construction summary
│   ├── noisy_dataset_summary.md            Noisy training data generation summary
│   ├── phase1_ablation_table.md / .csv     Phase 1 baseline comparison
│   └── wikipedia_generation_summary.md     Wikipedia scraping/annotation summary
│
├── src/                        Phase 1 baseline code — foundational work that
│                                established the case for fine-tuning
│   ├── baseline/
│   │   └── gemma_zero_shot.py         Zero-shot Gemma 3 baseline
│   ├── ontology/
│   │   └── alignment.py               Early ontology-alignment prototype
│   ├── evaluation/
│   │   └── error_taxonomy.py          Original 5-type error taxonomy
│   └── finetune/
│       └── generate_noisy_data.py     Noisy training data generator
│
└── notebooks/                  Phase-by-phase exploratory notebooks
    ├── 01_week1_baselines.ipynb
    └── 02_week2_error_analysis.ipynb
```

---

## Getting Started

### Setup

```bash
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework/GSoC26_H
pip install -r requirements.txt
```

### Smoke test

Verifies the training pipeline end to end — model loading, LoRA application, data loading, and training steps — on a small sample:

```bash
cd training
bash scripts/smoke_test.sh
```

### Step-by-step: running the full process yourself

**1. Prepare training data** (combines synthetic, noisy, and scored Wikipedia sources into train/validation splits):
```bash
cd training
python3 prepare_data.py
```

**2. Fine-tune the extraction model** (Gemma 3 4B, both learning rates):
```bash
bash scripts/train_exp1_lr2e4.sh
bash scripts/train_exp1_lr1e5.sh
```

**3. Build the predicate-linking gold set** (F2LLM-8B retrieval + GPT-OSS-120B disambiguation):
```bash
python3 build_gold_set_chunk0.py
python3 build_gold_set_chunk1.py
```

**4. Fine-tune the normalization model** (F2LLM-1.7B on the gold set):
```bash
python3 finetune_f2lm.py
```

**5. Run full-scale extraction and normalization:**
```bash
cd ../inference
python3 evaluate_full_scale.py
python3 normalize_full_scale.py
```

**6. Build and launch the human review interface:**
```bash
cd ../hitl
python3 generate_hitl_data.py
streamlit run hitl_app.py
```

### Loading the extraction model (Gemma 3 4B)

The extraction model is a QLoRA adapter, [published on Hugging Face](https://huggingface.co/Nitin1211/dbpedia-hindi-gemma3-4b-lr2e4), on top of the base Gemma 3 4B model, loaded in 4-bit:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
ADAPTER = "Nitin1211/dbpedia-hindi-gemma3-4b-lr2e4"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()
```

Generation stops on either the base `eos_token_id` or Gemma's `<end_of_turn>` token. See `inference/evaluate_full_scale.py` for the full manual decode loop.

### Loading the normalization model (F2LLM-1.7B)

The normalization model is loaded as a merged checkpoint via SentenceTransformer:

```python
from sentence_transformers import SentenceTransformer

FINETUNED_MODEL = "path/to/f2lm_finetuned_v2_merged"  # not yet published separately
model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)
```

**Hardware used:** NVIDIA A2, 16GB VRAM, 4-bit quantization throughout.

---

## Contributing

Part of Google Summer of Code 2026 with the DBpedia Association.

- Forum: [forum.dbpedia.org](https://forum.dbpedia.org/)
- Project home: [dbpedia.org](https://www.dbpedia.org/)

---

<p align="center">
  <i>Part of the DBpedia Neural Extraction Framework</i><br>
  <a href="https://www.dbpedia.org/">DBpedia Association</a>
</p>

<p align="center">
  GSoC 2026 Contributor — <a href="https://github.com/singhhnitin">@singhhnitin</a>
</p>
