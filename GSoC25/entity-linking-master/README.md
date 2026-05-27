# Batch Entity Linking Preprocessing Toolkit

A robust, modular, and extensible batch entity linking pipeline for large-scale entity normalization, context disambiguation, and knowledge base linking using LLMs (Gemini) and DBpedia.

## Features

- **Batch Canonical Name Normalization** (via Gemini LLM)
- **Batch Context Analysis** (via Gemini LLM)
- **Batch DBpedia URI Lookup** (via SPARQL)
- **Full Pipeline Integration**: End-to-end workflow
- **Flexible Input/Output**: Supports CSV, Excel, JSON
- **Export/Save Results**: Save to CSV, Excel, or JSON
- **Progress & Logging**: Prints progress and timing for each step
- **Error Reporting**: Summarizes missing/ambiguous results
- **Chunked Processing**: Handles large datasets efficiently

## Usage Example

```python
from batch_preprocessing.full_batch_pipeline import full_batch_entity_linking

entity_contexts = [
    {"mention": "Apple", "context": "I work at Apple"},
    {"mention": "Apple", "context": "I eat an apple every day"},
    # ...
]

# Run the full pipeline and save results
results = full_batch_entity_linking(
    entity_contexts,
    canonical_chunk_size=5,
    context_chunk_size=5,
    dbpedia_chunk_size=5,
    save_path="output.csv",  # or .xlsx, .json
    log=True
)
print(results)
```

| mention   | context                        | canonical_name      | entity_type | confidence | keywords         | description                  | dbpedia_uri                        |
|-----------|-------------------------------|---------------------|-------------|------------|------------------|------------------------------|-------------------------------------|
| Apple     | I work at Apple               | Apple_Inc.          | company     | 0.95       | ["tech", ...]   | Apple Inc. the company       | http://dbpedia.org/resource/Apple_Inc. |
| Apple     | I eat an apple every day      | Apple               | product     | 0.90       | ["fruit", ...]   | The fruit apple              | http://dbpedia.org/resource/Apple   |


## Input/Output
- **Input**: List of dicts with 'mention' and 'context', or load from CSV/Excel/JSON
- **Output**: DataFrame with columns: mention, context, canonical_name, entity_type, confidence, keywords, description, dbpedia_uri

## Architecture

- **batch_preprocessing/**
  - `batch_canonical_name.py`: Batch canonical name normalization (Gemini)
  - `batch_context_analysis.py`: Batch context disambiguation (Gemini)
  - `batch_dbpedia_uri.py`: Batch DBpedia URI lookup (SPARQL)
  - `full_batch_pipeline.py`: Orchestrates the full workflow, merging all results
  - Utility functions for loading/saving and error reporting

- **hybrid_linking/**
  - Core LLM and DBpedia utilities (used by batch modules)

## Technical Documentation
See [TECHNICAL.md](TECHNICAL.md) for a detailed description of the architecture, data flow, and design decisions.

## Requirements
- Python 3.8+
- Gemini API key (for LLM steps)
- DBpedia SPARQL

## License
MIT 