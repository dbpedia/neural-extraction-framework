# DBpedia Hindi Chapter — Neural Relational Triple Extraction

> Building a Hindi knowledge graph from Wikipedia, one sentence at a time.

[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![DBpedia](https://img.shields.io/badge/DBpedia-Hindi%20Chapter-0066CC?style=flat-square)](https://www.dbpedia.org/)
[![Model](https://img.shields.io/badge/Extraction-Gemma%203%204B-4285F4?style=flat-square)](https://huggingface.co/google/gemma-3-4b-it)
[![Model](https://img.shields.io/badge/Normalization-F2LLM--1.7B-00A67E?style=flat-square)]()

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
   ├──► "property"-type triples ──► set aside, never normalized
   │     (currently ~57% of all extracted triples — see "Open Items")
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
    including editing the subject/object text directly if the
    extraction span was wrong. Every triple gets a stable ID so the
    same item is never re-reviewed on reload.
   │
   ▼
[4] FEEDBACK LOOP — merge_hitl_feedback.py
    Pulls confirmed corrections and folds them back into the
    normalization cache and future training data.
```

**A deliberate design choice worth stating plainly:** normalization does not use a fixed confidence-score cutoff to decide what needs human review. An earlier version did, and it was silently discarding a large share of predicates from ever being reviewable. The current design instead checks simply whether a DBpedia property was found at all — every real relation gets a chance at review.

---

## Dataset

All sizes below are line counts confirmed directly against the live data files.

| Dataset | Size | Purpose |
|---|---|---|
| Training set | **79,242** examples | Fine-tuning Gemma 3 4B (extraction) |
| Validation set | **3,634** examples | High-scoring (≥9) Wikipedia sentences, held out during training |
| Held-out test set | **585** examples | Never used in training; source of the precision@k numbers below |
| Predicate-linking gold set | **8,029** entries | Built via F2LLM-8B retrieval + GPT-OSS-120B disambiguation (5,855 real DBO mappings, 2,174 initially NONE) |

Training data is drawn from three sources: an original LLM-generated synthetic dataset, a separately generated "noisy" dataset (deliberately imperfect examples for robustness), and real Hindi Wikipedia sentences (scraped and filtered to match the BenchIE benchmark's natural sentence-length distribution).

---

## Real Results

All numbers below are from the current fine-tuned checkpoints, run on the actual evaluation set (1,817 Wikipedia + 112 BenchIE + 50 Train sentences), not estimates or a small sample.

### Extraction (Gemma 3 4B, QLoRA)

Two learning rates were fine-tuned and compared on an LLM-as-judge + triple-level F1 evaluation:

| Learning rate | Wikipedia F1 | Train F1 |
|---|---|---|
| **2e-4 (selected)** | **0.692** | **0.795** |
| 1e-5 | 0.613 | 0.471 |

valid_format_rate: 98–100% (after fixing a bug where the model's `<end_of_turn>` stop token wasn't being recognized, which had been causing valid outputs to be scored as invalid).

### Predicate Normalization (F2LLM-1.7B, QLoRA, Round 2 — 9 epochs)

Evaluated on the true 585-item held-out set, checking whether the correct DBpedia property appears in the top-k candidates:

| k | Precision@k |
|---|---|
| 1 | 37.6% |
| 5 | 72.1% |
| 10 | 78.8% |
| 20 | 84.1% |
| 30 | 87.7% |
| **40** | **89.9%** |

QLoRA outperformed plain LoRA by 0.5–1.5 percentage points at every k (e.g. p@1: 37.6% vs 36.2%).

**NONE-predicate recovery:** of 2,174 predicates that initially returned no DBpedia match, retrying against ranks 51–100 (instead of stopping earlier) recovered 1,249 (57.5%) — raising total ontology coverage from 72.9% to 88.4%.

### Full-Scale End-to-End Pipeline (zero-shot prompting)

Complete pipeline — extraction → normalization → final property — run across the entire evaluation set:

| Source | Precision | Recall | F1 | N |
|---|---|---|---|---|
| Wikipedia | 0.493 | 0.490 | **0.490** | 1,817 |
| Train | 0.550 | 0.530 | **0.537** | 50 |
| BenchIE | 0.192 | 0.165 | **0.173** | 112 |

Total predicates normalized: 4,967 (out of 11,461 total extracted triples — the remainder are "property"-type, see Open Items below).

**BenchIE's much lower score is a known, actively investigated issue** — not a result being presented as final. See Open Items.

---

## Human-in-the-Loop Review

Live at a password-protected Streamlit app, synced directly to GitHub. Currently shows 1,564 triples for review — every one already has a suggested DBpedia property (only triples with a confirmed match are surfaced; the design decision on whether "no match" predicates should also be human-reviewable is still open).

Review data (`hitl_corrections.jsonl`) is stored with both the original extracted subject/object and the reviewer's corrected version, so no information is lost in a correction.

---

## Open Items — Stated Honestly

This project is under active development. Rather than presenting only finished results, here is exactly what's still open:

- **Few-shot extraction evaluation is currently running.** A few-shot version of the extraction prompt (using real examples from the training data, not synthetic ones) is being compared against the zero-shot results above. Results not yet final.
- **Property-triple audit is currently running.** Roughly 57% of all extracted triples are labeled "property" (non-relational) and never reach normalization. A manual spot-check of 40 such triples found 1 mislabeled — a real fact wrongly discarded. A full audit of all property-type triples using GPT-OSS-120B is in progress to get an exact count; an earlier run of this audit had a measurement bug (verdict parsing checked the start of the model's reasoning trace instead of its final answer) which has since been fixed and the audit restarted.
- **BenchIE's F1 (0.173) is substantially lower than Wikipedia/Train and is not yet root-caused.** The property-triple audit above may partially explain it if BenchIE has a higher mislabeling rate than other sources.
- **1,055 of the 1,249 NONE-predicate-recovery matches have not yet been merged into the normalization cache**, meaning they exist as verified correct answers but are not yet reflected in the HITL review file or the full-scale F1 numbers above.
- **`merge_hitl_feedback.py` is built but has not yet been run on a real batch of corrections** — there aren't yet enough accumulated HITL reviews to make running it worthwhile.
- **79 Wikipedia validation sentences (4.3%) were found to be corrupted** (leftover reasoning text from an earlier coreference-resolution step instead of the actual sentence). These have been removed from HITL, but the full-scale F1 numbers above were computed before this cleanup — a rerun on cleaned data is a pending decision.

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
│       ├── smoke_test.sh              50 examples, 1 epoch — verifies the pipeline
│       │                              runs end to end before committing to a full run
│       ├── train_exp1_lr2e4.sh        Full training run, lr=2e-4 (selected)
│       └── train_exp1_lr1e5.sh        Full training run, lr=1e-5 (comparison)
│
├── inference/                Running the trained models at scale
│   ├── evaluate_full_scale.py         Full-scale zero-shot extraction (1,979 sentences)
│   ├── evaluate_few_shot.py           Full-scale few-shot extraction (in progress)
│   ├── normalize_full_scale.py        Full-scale predicate normalization
│   ├── normalize_triples.py           Standalone normalization utility
│   └── retry_none_predicates.py       NONE-predicate recovery (ranks 51-100)
│
├── evaluation/                Held-out and checkpoint comparison evaluation
│   ├── evaluate_held_out.py           Precision@k on the true held-out set
│   ├── evaluate_all_checkpoints.py    Consistent re-evaluation across all checkpoints
│   └── evaluate_finetuned_fair.py     Fair QLoRA-vs-LoRA comparison (fixed encoding bug)
│
├── data_quality/              Data integrity checks
│   ├── check_all_property_triples.py  Full audit of "property"-type triples (in progress)
│   ├── filter_hitl_corrupted.py       Removes corrupted sentences from HITL data
│   ├── filter_optimal_only.py         Filters training traces to optimal-only
│   ├── scan_corrupted_sentences.py    Detects coreference-reasoning leakage in sentences
│   └── scan_corrupted_v2.py           Refined corrupted-sentence detection
│
├── hitl/                      Human review interface
│   ├── README.md
│   ├── hitl_app.py                    Streamlit review app
│   ├── generate_hitl_data.py          Builds the review queue from pipeline output
│   ├── merge_hitl_feedback.py         Folds corrections back into training data (not yet run)
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
│   ├── embedding_model_comparison.md       F2LLM vs e5-large-instruct comparison
│   ├── ground_truth_summary.md             BenchIE ground-truth construction summary
│   ├── noisy_dataset_summary.md            Noisy training data generation summary
│   ├── phase1_ablation_table.md / .csv     Phase 1 baseline comparison
│   └── wikipedia_generation_summary.md     Wikipedia scraping/annotation summary
│
├── src/                        Phase 1 baseline code (historical — superseded by the
│                                pipeline above, kept for reference and comparison)
│   ├── baseline/
│   │   └── gemma_zero_shot.py         Zero-shot Gemma 3 baseline, pre-fine-tuning
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

**Note on `src/`:** this folder holds the original Phase 1 baseline work, predating the fine-tuned pipeline above. Its own code comments record the zero-shot baseline's weak starting point (0/5 correct predicates on an early 5-sentence check) — exactly the gap that motivated fine-tuning Gemma 3 4B in the first place. Kept for historical reference, not part of the current pipeline.

---

## Getting Started

### Setup

```bash
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework/GSoC26_H
pip install -r requirements.txt
```

### Smoke test (recommended before any full run)

Verifies the training pipeline works end to end — model loading, LoRA application, data loading, and training steps — on a small sample before committing to a full run:

```bash
cd training
bash scripts/smoke_test.sh
```

### Loading the extraction model (Gemma 3 4B)

The extraction model is a QLoRA adapter on top of the base Gemma 3 4B model, loaded in 4-bit:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()
```

Generation must stop on either the base `eos_token_id` **or** Gemma's `<end_of_turn>` token — missing this causes valid outputs to loop and get scored as invalid. See `inference/evaluate_full_scale.py` for the full manual decode loop.

### Loading the normalization model (F2LLM-1.7B)

The normalization model is loaded as a merged checkpoint via SentenceTransformer. **Note:** confirm the exact checkpoint path on your own setup with `grep -n "FINETUNED_MODEL\s*=" inference/normalize_full_scale.py` before relying on the placeholder below.

```python
from sentence_transformers import SentenceTransformer

FINETUNED_MODEL = "path/to/f2lm_finetuned_v2_merged"  # confirm exact path before use
model = SentenceTransformer(FINETUNED_MODEL, trust_remote_code=True)
```

### Running the full pipeline

```bash
# 1. Extraction
python3 inference/evaluate_full_scale.py

# 2. Normalization
python3 inference/normalize_full_scale.py

# 3. Build the HITL review file
python3 hitl/generate_hitl_data.py

# 4. Launch the review app
streamlit run hitl/hitl_app.py
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
