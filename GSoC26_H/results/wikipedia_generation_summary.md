# Wikipedia-Based Synthetic Data Generation — Week 3

**Task:** Generate training examples from real Hindi Wikipedia sentences to reduce
hallucination risk and match the production distribution (model will be evaluated
on real Wikipedia-style content, not just synthetic sentences).

## Rationale

The original 20K dataset uses entirely model-invented sentences. This risks:
1. Model learning artificial phrasing patterns not present in real Hindi text
2. Distribution mismatch at evaluation time (BenchIE sentences are Wikipedia-sourced)

Sourcing sentences from real Wikipedia then using GPT-OSS-120B only for the
extraction/annotation step gives real sentence distribution + model-quality labels.

## Method

### Sentence-length target
BenchIE gold sentences: min=10, median=15.0, mean=15.1, max=23 words.
Wikipedia sentences filtered to the same 10–23 word range.

### Scraper
Reused the existing `GSoC25_H/src/wikipedia/reader.py` → `get_text_of_wiki_page()`
function without modification (as instructed — no rewriting existing scrapers).

**Key fix needed:** Wikimedia API now enforces a User-Agent policy. The `wikipedia`
Python library's default requests are rejected with HTTP 403. Fix:

```python
import requests
def patched_get(self, *args, **kwargs):
    headers = kwargs.get('headers', {}) or {}
    headers['User-Agent'] = "GSoC2026-DBpedia-Hindi-Research/1.0 (research project)"
    kwargs['headers'] = headers
    return original_get(self, *args, **kwargs)
requests.Session.get = patched_get
requests.get = patched_get_direct
```

### Generation
Each scraped sentence → `openai/gpt-oss-120b` with the same prompt schema:

```
System: Extract all subject-relation-object triplets...
User: <real Hindi Wikipedia sentence>
Assistant: {"thought_process": "...", "extracted_triplets": [...]}
```

Output format is identical to the original 20K dataset — directly mergeable.

## Progress

- Scraping: running incrementally with Drive checkpointing
  (auto-resumes on session reset via `wikipedia_scraped_sentences.jsonl`)
- Target: ~15,000–16,000 sentences
- Yield rate: ~7–10 matching sentences per successful article fetch

## Large file handling

Scraped sentences and generated examples are stored on Google Drive.
Output: `wikipedia_scraped_sentences.jsonl` (scraped) →
        `wikipedia_synthetic_data.jsonl` (after GPT-OSS-120B annotation, pending).
