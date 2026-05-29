# GSoC 2026 — DBpedia Hindi Chapter
## Fine-Tuning Indic Models for Hindi Relational Triple Extraction + Human-in-the-Loop Feedback

[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-fbbc05?style=flat-square&logo=google)](https://summerofcode.withgoogle.com/)
[![DBpedia](https://img.shields.io/badge/DBpedia-Hindi%20Chapter-0066CC?style=flat-square)](https://www.dbpedia.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

---

## 📋 Project Overview

The DBpedia Hindi Chapter aims to expand the multilingual depth of DBpedia by extracting structured relational triples (subject → predicate → object) from Hindi Wikipedia and integrating them into the DBpedia knowledge graph.

While DBpedia's existing extraction framework provides strong support for infobox-based triples, the extraction of relations from free-text Hindi sentences using neural and NLP-driven methods remains underdeveloped. A large portion of relational knowledge present in Hindi Wikipedia articles is not yet represented in structured form within DBpedia.

This project addresses that gap through **two integrated contributions**:

1. **Fine-tuning a Small Language Model (Gemma-3) with LoRA/QLoRA** for reliable relational triple extraction from Hindi text — improving over both prompt-only LLM baselines and rule-based extractors (IndIE).
2. **A lightweight Human-in-the-Loop (HITL) feedback interface** that turns reviewer corrections into an iteratively growing training dataset for active learning.

---

## 👥 Team

| Role | Person |
|---|---|
| **Contributor** | Nitin Singh ([@singhhnitin](https://github.com/singhhnitin)) — KIIT University, B.Tech CSE |
| **Mentor** | Sanju Tiwari ([@tiwarisanju18](https://github.com/tiwarisanju18)) |
| **Mentor** | Aditya Venkatesh |
| **Mentor** | Debarghya Dutta |
| **Mentor** | Ronak Panchal |

**Project Forum:** [DBpedia Hindi Chapter 2026 — Discussion](https://forum.dbpedia.org/t/dbpedia-hindi-chapter-2026-fine-tuning-indic-models-for-hindi-relational-triple-extraction-human-in-the-loop-feedback-gsoc-2026/4788)

---

## 🎯 Project Status

- [x] **Pre-application warm-up** — Zero-shot baseline, error taxonomy, ontology alignment prototype, HITL Streamlit prototype
- [x] **Phase 1 (Weeks 1–2)** — Baselines & Ablation *(in progress)*
  - [x] GSoC25_H pipeline reproduced on full Hindi-BenchIE
  - [x] IndIE baseline metrics recorded
  - [x] Zero-shot Gemma-3-1B baseline run on full dataset
  - [ ] Per-error-type breakdown across all systems
  - [ ] Ontology alignment layer evaluated on full set
- [ ] **Phase 2 (Weeks 3–6)** — Fine-tuning Gemma-3 with LoRA + Ontology Alignment
- [ ] **Phase 3 (Weeks 7–9)** — HITL Feedback UI + Active Learning
- [ ] **Phase 4 (Weeks 10–12)** — Iteration, Documentation, and Dataset Release

---

## 📊 Baseline Results (Hindi-BenchIE)

*Updated after Phase 1 evaluation. Numbers below are placeholders — see `results/baseline_table.csv` for current values.*

| System | Precision | Recall | F1 |
|---|---|---|---|
| IndIE (rule-based) | TBD | TBD | TBD |
| Zero-shot Gemma-3-1B | TBD | TBD | TBD |
| GSoC25_H (best system, Gemma-3-12B + IndIE + ReAct) | TBD | TBD | TBD |
| **Fine-tuned Gemma-3 (this work)** | — | — | — |
| **Fine-tuned + Ontology Alignment (this work)** | — | — | — |

---

## 🏗️ Architecture

```
Hindi Wikipedia Sentence
         ↓
IndIE (Rule-based Subject/Object Extraction)
         ↓
Fine-Tuned Gemma-3 (Predicate Extraction — MILIE-inspired iterative slot conditioning)
         ↓
Ontology Alignment Layer (Surface Predicate → DBpedia Property via multilingual embeddings)
         ↓
Confidence-Threshold Filter
         ↓
Human-in-the-Loop Review  (Accept / Edit / Reject) — for low-confidence triples only
         ↓
Final Validated Triple
         ↓
RDF Conversion (DBpedia Ontology Format)
         ↓
Integration into DBpedia Hindi Knowledge Graph
```

---

## 🔍 Error Taxonomy

Empirically identified from pre-application zero-shot experiments — used to make every evaluation interpretable at the failure-mode level rather than as a single aggregate number.

| Error Type | Description | Example |
|---|---|---|
| `predicate_normalization_failure` | Surface verb ≠ DBpedia property | `"का निर्माण"` extracted, should be `dbo:builder` |
| `language_mixing` | English predicate generated for Hindi input | `"was born in"` extracted, should be `dbo:birthPlace` |
| `implicit_relation_error` | Hindi copula extracted as predicate | `"है"` extracted, should be `dbo:capital` |
| `argument_span_error` | Wrong subject/object boundaries | — |
| `missing_triple` | No triple extracted | — |

---

## 📁 Repository Structure

```
GSoC26_H/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies (pinned)
├── .gitignore
│
├── notebooks/                       ← Phase-by-phase Colab notebooks
│   └── 01_week1_baselines.ipynb     ← IndIE / Gemma-3 zero-shot / GSoC25_H baselines
│
├── src/                             ← All Python source code
│   ├── baseline/
│   │   └── gemma_zero_shot.py       ← Zero-shot Gemma-3 runner (iterative + simultaneous prompts)
│   ├── ontology/
│   │   └── alignment.py             ← Surface predicate → DBpedia property alignment layer
│   ├── evaluation/
│   │   └── error_taxonomy.py        ← 5-type error classification + ablation table builder
│   ├── finetune/                    ← LoRA training (Phase 2)
│   └── pipeline/                    ← End-to-end pipeline + RDF converter (Phase 4)
│
├── hitl/                            ← Streamlit HITL annotation interface (Phase 3)
│
├── data/
│   ├── ontology/
│   │   └── dbpedia_properties.json  ← Curated DBpedia properties with Hindi surface forms
│   ├── training/                    ← Instruction-tuning pairs (Phase 2)
│   └── feedback/                    ← HITL JSONL correction outputs (Phase 3)
│
├── configs/
│   └── lora_config.yaml             ← LoRA/QLoRA training config (Colab T4 ready)
│
└── results/                         ← Evaluation outputs, ablation tables, error breakdowns
```

---

## 🚀 Quick Start

### Option 1: Google Colab (recommended)

```python
# In a new Colab notebook with T4 GPU runtime:
!git clone https://github.com/dbpedia/neural-extraction-framework.git
%cd neural-extraction-framework/GSoC26_H
!pip install -r requirements.txt
# Open notebooks/01_week1_baselines.ipynb and run all cells
```

### Option 2: Local installation

```bash
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework/GSoC26_H
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_week1_baselines.ipynb
```

---

## 📐 Design Decisions

**Why iterative slot prompting?**
Based on MILIE (Kotnis et al., ACL 2022): in zero-shot experiments, subjects and objects are extracted correctly while predicates fail completely. Fine-tuning prompts generate the predicate *after* subject + object are established in context, mirroring MILIE's iterative conditioning hypothesis.

**Why an ontology alignment layer?**
Zero-shot Gemma-3 produces surface Hindi verb phrases that are not aligned to DBpedia ontology properties. Pre-application experiments showed that adding a multilingual embedding-based alignment layer recovers 80% of predicate accuracy on a 5-sentence test set. The remaining low-confidence cases are flagged for HITL review rather than silently passed into the knowledge graph.

**Why confidence-based HITL routing (not random sampling)?**
Active learning principle: route uncertain extractions to human review rather than random samples. Each annotation round concentrates reviewer attention on cases where model uncertainty is highest, making annotation maximally informative for the next training iteration.

**Why LoRA/QLoRA instead of full fine-tuning?**
Colab T4 (16GB VRAM) cannot full-fine-tune Gemma-3-2B/4B. QLoRA (4-bit quantization + low-rank adapters) keeps memory usage under 10GB while updating only a small fraction of parameters — making the method reproducible by future contributors without specialized hardware.

---

## 📚 Key References

| Paper | Contribution to this project |
|---|---|
| **BenchIE** (Gashteovski et al., ACL 2022) | Evaluation protocol — fact synset matching, essential/compensatory triples |
| **MILIE** (Kotnis et al., ACL 2022) | Iterative slot extraction — predicates conditioned on subject/object |
| **OpenIE Survey** (Pai et al., EMNLP Findings 2024) | Motivates task-specific fine-tuning over prompt engineering |
| **IndIE** (Kothari et al.) | Underpins the current GSoC25_H rule-based component |
| **Hindi-BenchIE** | Primary evaluation dataset for this project |

---

## 🔗 Related GSoC Projects

This project builds directly on:
- **[GSoC25_H](../GSoC25_H/)** — Hindi Information Extraction pipeline (IndIE + LLM-IE + ReAct + predicate linking)
- **[GSoC24_H](../GSoC24_H/)** — Earlier Hindi DBpedia extraction work

And complements the parallel 2026 effort: *Stabilizing, Completing, and Upstreaming the Hindi DBpedia IE Pipeline*.

---

## 📞 Contact

- **GitHub:** [@singhhnitin](https://github.com/singhhnitin)
- **Email:** nitinsingh3323@gmail.com
- **DBpedia Forum:** [Project thread](https://forum.dbpedia.org/t/dbpedia-hindi-chapter-2026-fine-tuning-indic-models-for-hindi-relational-triple-extraction-human-in-the-loop-feedback-gsoc-2026/4788)
- **DBpedia Slack:** [dbpedia.slack.com](https://dbpedia.slack.com/)

---

*Built as part of Google Summer of Code 2026 with the DBpedia Association.*
