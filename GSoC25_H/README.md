# DBpedia Hindi Extraction Framework

**A comprehensive pipeline for extracting structured knowledge triplets from Hindi text, combining rule-based methods, neural models, and large language models.**


> **Note:** This documentation was updated to reflect the current state of the codebase (Feb 2026).

---

## Overview

This repository contains the DBpedia Hindi Extraction Framework, originally developed during GSoC 2024/2025 and currently being updated for the GSoC 2026 cycle.

---

## Directory Structure

```text
GSoC25_H/
├── IndIE/                          # Rule-based extraction with LLM enhancement
│   ├── chunking/                   # Sentence chunking models (XLM-R, CRF)
│   ├── hindi-benchie/              # Benchmark dataset and evaluation scripts
│   ├── templates/                  # Flask web interface templates
│   ├── main.py                     # Main extraction script
│   ├── llm_extractor.py            # LLM-based extraction fallback
│   ├── convert.py                  # H5 to text format converter
│   └── app.py                      # Flask web application
│
├── llm_IE/                         # LLM evaluation framework
│   ├── finetuning/                 # Synthetic data generation scripts
│   ├── config.py                   # Model and experiment configuration
│   ├── prompt_templates.py         # Multiple prompting strategies
│   ├── llm_interface.py            # Ollama API wrapper
│   ├── output_parser.py            # Response parsing logic
│   ├── full_dataset_evaluation.py  # Benchmark runner
│   └── detailed_comparison_using_benchIE.py  # Analysis script
│
├── ReAct/                          # Reasoning + Acting framework
│   ├── config.py                   # ReAct-specific configuration
│   ├── prompt_factory.py           # Prompt generation for tool calling
│   ├── llm_interface.py            # Function-calling interface
│   ├── data_loader.py              # Dataset loading utilities
│   └── evaluation.py               # Evaluation metrics
│
├── link_prediction/                # Knowledge graph link prediction
│   ├── data/                       # DBpedia Hindi dumps (TTL files)
│   ├── link_prediction.ipynb       # TransE and ConvE implementations
│   ├── hiwiki-analysis.ipynb       # Data exploration notebook
│   └── requirements.txt            # PyKEEN and dependencies
│
├── src/                            # Main pipeline components
│   ├── coref/                      # WL-Coref implementation
│   ├── chunking/                   # Chunking model wrapper
│   ├── wikipedia/                  # Wikipedia data reader
│   ├── start.py                    # CLI entry point for full pipeline
│   ├── demo.py                     # Streamlit interactive demo
│   ├── indIE.py                    # IndIE wrapper for pipeline
│   ├── llm_triplets.py             # LLM-based triplet extraction
│   ├── llm_coreference.py          # LLM-based coreference resolution
│   ├── entity_linking.py           # mGENRE entity linking
│   ├── el_normalize.py             # Wikidata to DBpedia URI conversion
│   ├── predicate_linking.py        # Hindi predicate to DBpedia property mapping
│   └── utils.py                    # Shared utilities
│
├── models/                         # Pre-trained model storage
│   ├── coref_model/                # WL-Coref multilingual model
│   ├── RE_model/                   # IndIE chunking models
│   ├── EL_model/                   # mGENRE entity linking model
│   ├── ontology/                   # DBpedia ontology embeddings (auto-generated)
│   └── download_models.sh          # Automated download script
│
├── ontology_input/                 # DBpedia ontology (TTL format)
│   └── ontology--DEV_type=parsed.ttl
│
├── assets/                         # Demo screenshots
├── requirements.txt                # Main Python dependencies
└── README.md                       # This file
```

---

## Components

### 1. IndIE (Rule-Based Extraction)

**Location:** `IndIE/`

A hybrid information extraction system that combines handwritten dependency rules with optional LLM enhancement. The pipeline consists of three stages:

1. **Chunking:** Breaks sentences into meaningful phrases using XLM-RoBERTa or CRF models
2. **Merged Dependency Tree (MDT):** Identifies grammatical relationships between chunks
3. **Extraction:** Applies 100+ handwritten rules or LLM-based extraction to generate triplets

**Key Files:**
- `main.py` - Entry point with configurable extraction modes
- `llm_extractor.py` - LLM fallback for zero-extraction sentences
- `convert.py` - Converts H5 output to tab-separated format
- `chunking/chunking_model.py` - XLM-RoBERTa chunking implementation
- `chunking/crf_chunker.py` - CRF-based chunking alternative

**Extraction Modes (configured in `main.py`):**
- `use_llm=True` - Replace rules with LLM entirely
- `llm_fallback=True` - Use LLM only when rules produce zero triplets
- `llm_enhancement=True` - Combine rule-based + LLM outputs
- `llm_filter_mode=True` - Filter outputs using LLM

### 2. llm_IE (LLM Evaluation Framework)

**Location:** `llm_IE/`

A plug-and-play framework for evaluating Small Language Models on Hindi information extraction using the Hindi-BenchIE benchmark (112 sentences).

**Features:**
- Six prompting strategies (few-shot, chain-of-thought, evidence-based reasoning)
- Supports any Ollama-compatible model
- Detailed precision/recall/F1 metrics with per-sentence TP/FP/FN breakdowns

**Key Files:**
- `config.py` - Model selection and hyperparameter settings
- `prompt_templates.py` - Implements all prompting strategies
- `llm_interface.py` - Ollama REST API wrapper
- `full_dataset_evaluation.py` - Runs complete benchmark evaluation
- `detailed_comparison_using_benchIE.py` - Generates detailed analysis

**Available Strategies:**
1. `few_shot` - English instructions with Hindi examples
2. `few_shot_hindi` - Hindi-only instruction set
3. `chain_of_thought` - Step-by-step reasoning in Hindi
4. `chain_of_thought_english_hindi` - Bilingual reasoning
5. `chain_of_thought_ER` - Evidence-based extraction in Hindi
6. `chain_of_thought_ER_english_hindi` - Bilingual evidence-based

### 3. ReAct (Reasoning + Acting)

**Location:** `ReAct/`

Implements the ReAct framework using LLM native function-calling capabilities for structured triplet extraction.

**Key Files:**
- `config.py` - Tool definitions and model configuration
- `prompt_factory.py` - Generates ReAct-style prompts
- `llm_interface.py` - Handles tool-calling API interactions
- `evaluation.py` - Evaluates ReAct outputs against benchmarks

### 4. Link Prediction

**Location:** `link_prediction/`

Knowledge Graph Embedding experiments for predicting missing links in Hindi DBpedia using TransE and ConvE models.

**Contents:**
- `link_prediction.ipynb` - Model training and evaluation
- `hiwiki-analysis.ipynb` - Hindi DBpedia statistics and exploration
- `data/` - 13 DBpedia Hindi TTL dumps (May 2025)

**Models Implemented:**
- **TransE:** Translational distance embeddings
- **ConvE:** Convolutional neural network embeddings
- **MURIL Integration:** Uses MuRIL embeddings as initialization

### 5. Main Pipeline

**Location:** `src/`

The production-ready extraction pipeline integrating all components.

**Pipeline Stages:**

1. **Coreference Resolution** (`coref/`)
   - Model: WL-Coref (multilingual XLM-R)
   - Resolves pronouns and entity mentions

2. **Relation Extraction** (`indIE.py`, `llm_triplets.py`)
   - Method 1: Rule-based (IndIE wrapper)
   - Method 2: LLM-based fallback

3. **Entity Linking** (`entity_linking.py`)
   - Model: mGENRE (multilingual GENRE)
   - Links Hindi entities to Wikipedia/DBpedia

4. **Entity Normalization** (`el_normalize.py`)
   - Converts mGENRE output (Wikidata IDs) to DBpedia URIs

5. **Predicate Linking** (`predicate_linking.py`)
   - Maps Hindi relation phrases to DBpedia properties
   - Hybrid scoring: graph evidence + embeddings + lexical + types

**Entry Points:**
- `start.py` - CLI for batch processing
- `demo.py` - Streamlit interactive interface

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, recommended)
- 16GB RAM minimum
- Ollama (for LLM components)

### Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd GSoC25_H

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download pre-trained models
cd models
bash download_models.sh
cd ..

# 5. Install Ollama (for LLM components)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# 6. Pull an LLM model
ollama pull gemma3:4b
```

### Important Notes

**Fairseq Compatibility (PyTorch 2.0+):**

If using PyTorch 2.0 or later, modify the fairseq checkpoint loader:

```python
# File: fairseq/fairseq/checkpoint_utils.py
# Line: 271
# Change:
state = torch.load(f, map_location=torch.device("cpu"))
# To:
state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
```

---

## Usage

### Run Full Pipeline (CLI)

```bash
python src/start.py \
  --input_dir input/ \
  --do_coref \
  --do_rel \
  --do_el \
  --do_prop_link \
  --verbose
```

**Flags:**
- `--do_coref` - Enable coreference resolution
- `--do_rel` - Enable relation extraction
- `--do_el` - Enable entity linking
- `--do_prop_link` - Enable predicate linking
- `--verbose` - Print detailed output

### Run Streamlit Demo

```bash
streamlit run src/demo.py
```

Open browser at `http://localhost:8501` and enter a Hindi Wikipedia article name.

### Run IndIE Extraction

```bash
cd IndIE

# Configure extraction mode in main.py (lines 45-50)
# Set: use_llm, llm_fallback, llm_enhancement, llm_filter_mode

# Run extraction
python main.py

# Convert output
python convert.py
```

### Evaluate LLMs on BenchIE

```bash
cd llm_IE

# Configure models and strategies in config.py
python full_dataset_evaluation.py

# Generate detailed analysis
python detailed_comparison_using_benchIE.py
```

### Run Link Prediction

```bash
cd link_prediction

# Open notebooks in Jupyter
jupyter notebook
```

---

## Model Downloads

The `models/download_models.sh` script downloads:

1. **Coreference Model** (WL-Coref) - ~1.2GB
2. **Relation Extraction Model** (IndIE Chunker) - ~500MB
3. **Entity Linking Model** (mGENRE) - ~2.5GB
4. **Entity Linking Trie** (Marisa Trie) - ~800MB

**Manual Download Links:**

| Component | Link |
|-----------|------|
| Coref | [Google Drive](https://drive.google.com/file/d/1ScVz_o4V3G7watezLriCC0vU5gT7FO7q) |
| RE | [Google Drive](https://drive.google.com/file/d/1UqOUdeK96m6EabI-cg2EeBz6p3IwrPZ6) |
| EL Model | [Facebook AI](https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz) |
| EL Trie | [Facebook AI](http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl) |

---

## Data Resources

### Datasets

- **Hindi-BenchIE:** 112-sentence gold standard for Hindi Open IE
  - Location: `hindi-benchie/hindi_benchie_gold.txt`
  - Format: Custom text with clusters and compensatory extractions

- **DBpedia Hindi (May 2025):** Knowledge graph dumps
  - Location: `link_prediction/data/*.ttl.bz2`
  - Size: 13 files, ~8GB compressed

- **DBpedia Ontology:** Property definitions
  - Location: `ontology_input/ontology--DEV_type=parsed.ttl`

### External Resources

- **Experiment Results:** [Google Drive](https://drive.google.com/drive/folders/1rYZbLRgZRwfyVJJvsxhqqODQvIrA1JCs)
- **Full Dataset:** [Google Drive](https://drive.google.com/drive/folders/1fgbZdGAnLhIASQRKEuyOwbBFvFZvJt_R)

---

## Performance Benchmarks

### LLM Evaluation Results (Hindi-BenchIE)

**Best Performing Configuration:**
- **Model:** Gemma 3 4B (Quantized)
- **Strategy:** `chain_of_thought_ER` (Evidence-Based Reasoning)
- **F1-Score:** **25.48%**
- **Precision:** 27.38%
- **Recall:** 23.83%

**Top 3 Model-Strategy Pairs:**

| Rank | Model | Strategy | F1 | Precision | Recall |
|------|-------|----------|----|-----------|--------|
| 1 | gemma3:4b | chain_of_thought_ER | 25.48% | 27.38% | 23.83% |
| 2 | gemma3:4b | chain_of_thought_ER_english_hindi | 24.92% | 28.87% | 21.93% |
| 3 | gemma3:4b | few_shot | 14.44% | 14.36% | 14.53% |

**Key Findings:**
1. **Evidence-Based Reasoning Wins:** The `chain_of_thought_ER` strategy yielded a **75% improvement** over standard few-shot prompting.
2. **Architecture > Scale:** The 4B parameter Gemma model outperformed the 7B Mistral model by **2.4x** on average F1 score.
3. **Language Agnostic Reasoning:** Structured reasoning strategies performed well regardless of whether the prompts were in Hindi or English, whereas unstructured chain-of-thought failed in both.

*For a detailed analysis of all 18 experiments, see `llm_IE/results_and_discussion.md`.*

### IndIE Enhancement Results

| Method | Recall | Notes |
|--------|--------|-------|
| IndIE (baseline) | 50% | Rule-based only |
| IndIE + LLM (fallback) | 66% | +32% improvement |

---

## Configuration

### IndIE Extraction Modes

Edit `IndIE/main.py` hyperparameters:

```python
hyper_params = {
    'use_llm': False,         # Replace rules with LLM
    'llm_fallback': True,     # LLM only for zero-triplet sentences
    'llm_enhancement': True,  # Combine rules + LLM
    'llm_filter_mode': False, # LLM-based filtering
    'llm_model': 'gemma3:4b',
    'chunker': 'XLM',         # 'XLM' or 'CRF'
}
```

### LLM Evaluation Setup

Edit `llm_IE/config.py`:

```python
self.experiment = ExperimentConfig(
    models=["gemma3:4b", "mistral:latest"],
    prompt_strategies=["chain_of_thought_ER", "few_shot"],
)
```

### Predicate Linking Weights

Edit `src/predicate_linking.py`:

```python
# Scoring weights (sum to 1.0)
w_graph = 0.4  # Direct DBpedia edges
w_emb = 0.3    # Semantic similarity
w_lex = 0.2    # Label matching
w_type = 0.1   # Type compatibility
```

---

## Known Issues

1. **Entity Linking Disambiguation:** Common names may link incorrectly
2. **Type Linking Not Implemented:** Currently handles only relational properties, not type assertions
3. **SPARQL Timeouts:** Occasional timeouts on complex predicate linking queries
4. **LLM Hallucination:** High false positive rates (50-70%) in pure LLM modes
5. **GPU Memory:** Larger Gemma models may require higher VRAM (use quantized models)

---

## References

### Related Projects

- **IndIE Paper:** [IJCNLP-AACL 2023 Findings](http://103.25.231.59:80)
- **WL-Coref:** [GitHub](https://github.com/vdobrovolskii/wl-coref)
- **mGENRE:** [Facebook Research](https://github.com/facebookresearch/GENRE)
- **Hindi-BenchIE:** [GitHub](https://github.com/ritwikmishra/hindi-benchie)

### DBpedia Resources

- **DBpedia Downloads:** [Databus](https://databus.dbpedia.org/)
- **Hindi Wikipedia Dumps:** [Wikimedia](https://dumps.wikimedia.org/hiwiki/)
- **Extraction Framework:** [GitHub](https://github.com/dbpedia/extraction-framework)

---

## License

This project integrates multiple open-source components with varying licenses. Please refer to individual component licenses:

- IndIE: (Original repository license)
- WL-Coref: MIT License
- mGENRE: CC-BY-NC 4.0

---

## Acknowledgments

- Google Summer of Code 2024/2025
- DBpedia Association
- IndIE Authors
- Meta AI (mGENRE)
- Hindi NLP Community (BenchIE benchmark)

---

**Repository Status:** Active  
**Last Updated:** February 2026  
**Primary Use Case:** Hindi Wikipedia knowledge extraction for DBpedia
