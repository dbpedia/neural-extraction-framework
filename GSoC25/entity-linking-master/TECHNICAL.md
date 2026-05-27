# Technical Documentation: Batch Entity Linking Preprocessing Toolkit

## Overview
This toolkit provides a robust, modular, and extensible batch entity linking pipeline for large-scale entity normalization, context disambiguation, and knowledge base linking using LLMs (Gemini) and DBpedia. It is designed for high-throughput, production-grade entity linking and annotation workflows.

---

## Architecture

### High-Level Flow

1. **Input**: List of entity-context pairs (or load from CSV/Excel/JSON)
2. **Batch Canonical Name Normalization**: Uses Gemini LLM to normalize mentions to canonical names
3. **Batch Context Analysis**: Uses Gemini LLM to analyze context and disambiguate entity type
4. **Batch DBpedia URI Lookup**: Uses SPARQL to map canonical names to DBpedia URIs
5. **Output**: Merged DataFrame/JSON with all results, including error reporting and export options

### Directory Structure

```
batch_preprocessing/
├── batch_canonical_name.py      # Batch canonical name normalization (Gemini)
├── batch_context_analysis.py    # Batch context disambiguation (Gemini)
├── batch_dbpedia_uri.py         # Batch DBpedia URI lookup (SPARQL)
├── full_batch_pipeline.py       # Orchestrates the full workflow
```

---
```
examples/*  # Contains all examples scripts for the batch preprocessing full pipeline
```

```
hybrid_linking/*   # Core initial implementation for DBpedia
```

## Module Responsibilities

### `batch_canonical_name.py`
- Accepts a list of entity mentions
- Splits into manageable chunks for Gemini
- Sends batch prompts to Gemini for canonical name normalization
- Handles errors, missing results, and deduplication
- Returns results as DataFrame, JSON, or list of dicts

### `batch_context_analysis.py`
- Accepts a list of dicts with 'mention' and 'context'
- Splits into manageable chunks for Gemini
- Sends batch prompts to Gemini for context analysis (entity type, confidence, keywords, description)
- Handles errors and missing results
- Returns results as DataFrame, JSON, or list of dicts

### `batch_dbpedia_uri.py`
- Accepts a list of canonical names
- Splits into manageable chunks for DBpedia SPARQL
- Uses VALUES clause to look up all URIs in each batch
- Handles errors and missing results
- Returns results as DataFrame, JSON, or list of dicts

### `full_batch_pipeline.py`
- Orchestrates the full workflow:
  1. Canonical name normalization
  2. Context analysis
  3. DBpedia URI lookup
- Merges all results into a single DataFrame
- Provides utility functions for loading/saving input/output
- Reports errors and progress

---

## Data Flow

1. **Input**: List of dicts (or DataFrame) with 'mention' and 'context'
2. **Canonical Name Normalization**: Adds 'canonical_name' column
3. **Context Analysis**: Adds 'entity_type', 'confidence', 'keywords', 'description' columns
4. **DBpedia URI Lookup**: Adds 'dbpedia_uri' column
5. **Output**: DataFrame/JSON/CSV/Excel with all columns merged

---

## Design Decisions

- **Chunked Processing**: All batch steps use chunking to avoid API/endpoint limits and improve reliability
- **Progress & Logging**: Each batch prints progress and timing for transparency
- **Error Handling**: All steps catch and report errors, and missing/ambiguous results are summarized
- **Flexible I/O**: Utility functions support loading/saving from/to CSV, Excel, and JSON
- **Modularity**: Each batch step is a standalone module, making it easy to swap out or extend
- **Extensibility**: The pipeline can be extended to support new LLMs, knowledge bases, or additional analysis steps

---

## Example Data Flow

| mention   | context                        | canonical_name      | entity_type | confidence | keywords         | description                  | dbpedia_uri                        |
|-----------|-------------------------------|---------------------|-------------|------------|------------------|------------------------------|-------------------------------------|
| Apple     | I work at Apple               | Apple_Inc.          | company     | 0.95       | ["tech", ...]   | Apple Inc. the company       | http://dbpedia.org/resource/Apple_Inc. |
| Apple     | I eat an apple every day      | Apple               | product     | 0.90       | ["fruit", ...]   | The fruit apple              | http://dbpedia.org/resource/Apple   |

---

## Extending the Pipeline
- Add new batch modules for additional analysis (e.g., Wikidata lookup)
- Integrate with other LLMs by swapping out the Gemini-based modules
- Add CLI or API wrappers for production deployment

---

## Requirements
- Python 3.8+
- Gemini API key (for LLM steps)
- DBpedia SPARQL

---
