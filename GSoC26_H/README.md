# DBpedia Hindi Chapter — Neural Relational Triple Extraction

> Building a Hindi knowledge graph from Wikipedia, one sentence at a time.

[![Status](https://img.shields.io/badge/Status-Active%20Development-success?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![DBpedia](https://img.shields.io/badge/DBpedia-Hindi%20Chapter-0066CC?style=flat-square)](https://www.dbpedia.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)]()

---

## 🌏 The Problem

Hindi is spoken by over **600 million people**. Hindi Wikipedia has **160,000+ articles** packed with factual knowledge. Yet, when researchers, search engines, or AI systems need *structured* Hindi facts, they hit a wall.

DBpedia is the world's largest open knowledge graph extracted from Wikipedia — but its **Hindi chapter is sparse**. Why? Because the relational knowledge sitting inside Hindi Wikipedia articles is locked in **free text**, not structured infoboxes:

> *"ताज महल का निर्माण शाहजहाँ ने करवाया था।"*
> (The Taj Mahal was built by Shah Jahan.)

A human reads this and knows: **Taj Mahal — `builder` → Shah Jahan**.

A machine, looking for an infobox or a clean table? Finds nothing.

**Result:** A massive gap between what Hindi Wikipedia *contains* and what Hindi DBpedia *structures*.

---

## 🎯 What This Project Does

This project closes that gap by building an **end-to-end pipeline** that reads free-text Hindi sentences and outputs DBpedia-compatible structured triples — ready to be ingested into the knowledge graph.

```
Hindi Sentence  →  Extract  →  Align to Ontology  →  Validate  →  RDF Triple
```

For the example above, the pipeline produces:

```turtle
:Taj_Mahal  dbo:builder  :Shah_Jahan .
```

That's a structured fact a machine can query, reason over, and link to other knowledge.

---

## 🧩 Why Hindi Is Hard

Generic information extraction tools fail on Hindi for specific linguistic reasons:

| Hindi Feature | What It Means | Why It Breaks Generic IE |
|---|---|---|
| **Free word order** | Subject, object, verb can appear in any order | Pattern-based extractors miss arguments |
| **Postposition system** | Tiny words like *का, ने, में* carry the relation | English-trained models ignore them |
| **Pro-drop** | Subjects often implicit | Extractors return empty subjects |
| **Verb-final syntax** | The action comes at the end | Left-to-right parsers miss the predicate |
| **Copula relations** | *"है"* (is) hides the real relationship | Models extract *"है"* as the predicate — meaningless |

These aren't edge cases. They show up in nearly every Hindi Wikipedia sentence.

---

## 🏗️ The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                  HINDI WIKIPEDIA SENTENCE                        │
│       "ताज महल का निर्माण शाहजहाँ ने करवाया था।"                  │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  1️⃣  RULE-BASED EXTRACTION (IndIE)                                │
│      Identifies candidate arguments using Hindi syntax rules     │
│      Output: subject = "ताज महल", object = "शाहजहाँ"             │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2️⃣  FINE-TUNED SMALL LANGUAGE MODEL (Gemma-3 + LoRA)             │
│      Predicts the relation, conditioned on the arguments         │
│      Output: predicate = "का निर्माण"                            │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3️⃣  ONTOLOGY ALIGNMENT LAYER                                     │
│      Maps the Hindi surface predicate to a DBpedia property      │
│      using multilingual sentence embeddings                      │
│      Output: dbo:builder  (confidence: 0.87)                     │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
                  ┌───────────┴───────────┐
                  │  Confidence > 0.45?   │
                  └───────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────┐           ┌──────────────────────┐
    │   ✅ Accepted    │           │   🧑 HUMAN REVIEW     │
    │  Direct to KG    │           │  Reviewer corrects   │
    └────────┬────────┘           │  → retraining data   │
             │                    └──────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  4️⃣  RDF SERIALIZATION                                            │
│      :Taj_Mahal  dbo:builder  :Shah_Jahan .                      │
└─────────────────────────────┬────────────────────────────────────┘
                              ▼
                  📊 DBpedia Hindi Knowledge Graph
```

---

## 💡 The Key Ideas

### 1. Predicates are the hard part

Multilingual language models, even in zero-shot mode, are surprisingly good at identifying *who* and *what* in a Hindi sentence. They struggle with the **relation between them**.

This insight — that subjects and objects come for free, but predicates need work — shapes the entire architecture. Fine-tuning focuses where it matters; the easy slots are left alone.

### 2. Don't ask the model to do schema alignment

Asking a language model to output `dbo:birthPlace` from raw Hindi text is asking it to do two things at once: understand the relation **and** know the DBpedia ontology. Models hallucinate properties that don't exist.

A separate **ontology alignment layer** handles the second task. The model produces a Hindi surface form ("का जन्म"); a multilingual embedding model maps it to the right DBpedia property via cosine similarity. Each component does one job well.

### 3. Humans review what matters

A naive review pipeline asks a human to check every extraction — expensive and boring. A smarter one asks humans only when the model is **uncertain**. Confidence scores from the alignment layer become the trigger: high confidence → straight into the KG, low confidence → human review queue.

Every correction becomes new training data. The system gets better with use.

---

## 🔬 Error Taxonomy

When extraction fails, *how* it fails matters more than aggregate accuracy. Five failure modes are tracked across every evaluation:

| Error Type | What Happens | Example |
|---|---|---|
| **Predicate Normalization Failure** | Surface Hindi extracted instead of DBpedia property | `का निर्माण` → should be `dbo:builder` |
| **Language Mixing** | Model outputs English on Hindi input | `was born in` → should be `dbo:birthPlace` |
| **Implicit Relation Error** | Copula extracted as predicate | `है` → should be `dbo:capital` |
| **Argument Span Error** | Wrong subject/object boundaries | Captures *"ताज महल का"* instead of *"ताज महल"* |
| **Missing Triple** | No extraction at all | Sentence skipped entirely |

This taxonomy turns *"the model is 60% accurate"* into actionable diagnostics: *"language mixing dropped from 40% to 5%, but argument span errors are still our biggest loss."*

---

## 📦 What's Inside

```
GSoC26_H/
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📓 notebooks/             ← Phase-by-phase Colab notebooks
│
├── 🐍 src/
│   ├── baseline/             ← Rule-based and zero-shot baselines
│   ├── ontology/             ← Hindi → DBpedia property alignment
│   ├── finetune/             ← LoRA training pipeline
│   ├── evaluation/           ← Error taxonomy + metrics
│   └── pipeline/             ← End-to-end extractor + RDF export
│
├── 🖥️ hitl/                  ← Human-in-the-loop Streamlit interface
│
├── 📊 data/
│   ├── ontology/             ← DBpedia properties with Hindi surface forms
│   ├── training/             ← Instruction-tuning pairs
│   └── feedback/             ← Reviewer corrections (JSONL)
│
├── ⚙️ configs/                ← LoRA configs, hyperparameters
└── 📈 results/                ← Evaluation outputs, ablation tables
```

---

## 🚀 Getting Started

### Run in Google Colab (no setup)

The Phase 1 baseline notebook reproduces all three baselines on the full Hindi-BenchIE benchmark using a free T4 GPU.

```python
!git clone https://github.com/dbpedia/neural-extraction-framework.git
%cd neural-extraction-framework/GSoC26_H
!pip install -r requirements.txt
# Open notebooks/01_week1_baselines.ipynb and run all cells
```

### Run locally

```bash
git clone https://github.com/dbpedia/neural-extraction-framework.git
cd neural-extraction-framework/GSoC26_H

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/01_week1_baselines.ipynb
```

**Hardware:** Phase 1 runs on free Colab T4. Phase 2 (LoRA fine-tuning) needs ~10GB VRAM — Colab T4 is sufficient with 4-bit quantization.

---

## 📊 Evaluation

All systems are evaluated on **Hindi-BenchIE** using the official `BenchIEDetailedComparator` from the GSoC 2025 Hindi pipeline. This ensures numbers are directly comparable across years.

The evaluation uses **fact-cluster matching** instead of string overlap:
- Each gold annotation lists *all* valid surface forms of the same fact
- A prediction is correct if it matches *any* form in the cluster
- Avoids penalizing valid paraphrases — a known weakness of older OIE benchmarks

Results are reported as:
- **Aggregate**: Precision, Recall, F1 across the full benchmark
- **Per-system slot accuracy**: Subject / Predicate / Object separately
- **Per-error-type breakdown**: Counts and percentages for each of the 5 failure modes

---

## 🛣️ Roadmap

```
Phase 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Baselines & Ablation
  ✅ Reproduce IndIE, Gemma-3 zero-shot, GSoC25_H pipelines
  ✅ Aggregate P/R/F1 on Hindi-BenchIE
  🔄 Per-error-type breakdown across all systems

Phase 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Fine-Tuning + Ontology Alignment
  ⏳ Build training dataset from BenchIE gold annotations
  ⏳ LoRA fine-tune Gemma-3 with iterative slot prompting
  ⏳ Extend ontology alignment to full DBpedia property coverage

Phase 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Human-in-the-Loop Feedback
  ⏳ Production Streamlit annotation interface
  ⏳ Confidence-based review queue
  ⏳ Annotation round on 50–100 sentences

Phase 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Iteration & Release
  ⏳ Retrain with reviewer-corrected data
  ⏳ Final pipeline, dataset, evaluation report
  ⏳ Integration documentation for DBpedia maintainers
```

---

## 📚 Built On

This work stands on three pieces of recent research:

- **[BenchIE](https://aclanthology.org/2022.acl-long.307/)** — Fact-cluster evaluation that doesn't get fooled by surface paraphrase
- **[MILIE](https://aclanthology.org/2022.acl-long.555/)** — Iterative slot extraction; predicates conditioned on arguments
- **[OpenIE Survey 2024](https://aclanthology.org/2024.findings-emnlp.222/)** — State of the field, where neural methods help and where they don't
- **IndIE** — Rule-based Hindi OIE; provides the strong argument-extraction foundation

And on the prior DBpedia GSoC Hindi work in [`GSoC24_H/`](../GSoC24_H/) and [`GSoC25_H/`](../GSoC25_H/).

---

## 🌟 Why It Matters

Every triple extracted is one more queryable fact about Hindi-speaking history, geography, science, and culture — added to a knowledge graph used by researchers, search engines, voice assistants, and AI systems worldwide.

A pipeline that works for Hindi can be adapted for **other low-resource languages**: Bengali, Tamil, Telugu, Marathi. Eight hundred million more speakers, eight more low-resource Wikipedias waiting to become structured knowledge.

Closing the Hindi gap is the first step.

---

## 🤝 Contributing

This project is part of Google Summer of Code 2026 with the DBpedia Association.

- 💬 **Forum:** [forum.dbpedia.org](https://forum.dbpedia.org/)
- 💼 **Slack:** [dbpedia.slack.com](https://dbpedia.slack.com/)
- 🌐 **Project home:** [dbpedia.org](https://www.dbpedia.org/)

Contributions, issues, and discussion welcome.

---

<p align="center">
  <i>Part of the DBpedia Neural Extraction Framework</i><br>
  <a href="https://www.dbpedia.org/">🏛️ DBpedia Association</a>
</p>
