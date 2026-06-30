# Noisy Synthetic Dataset Generation — Week 2-3

**Task:** Generate a noisy training dataset for staged fine-tuning (Phase 1 noisy →
Phase 2 clean refinement), using the same model tier Aditya used for the original
20K dataset.

## Method

Generator: `openai/gpt-oss-120b` via NVIDIA's free hosted API
(`https://integrate.api.nvidia.com/v1`) — same model tier as the original 20K set.

**Why gpt-oss-120b, not a smaller model:**
Using a weaker 3-4B model to generate "noisy" data conflates two variables — the
noise level AND the model capability. By using the same strong model but seeding it
with flawed few-shot examples (score < 8 from the original 20K set), the noise comes
from genuine semantic mistakes being imitated, not from a model's inability to follow
instructions. This produces more realistic, instructive training noise.

**Prompt strategy (reusing Aditya's infrastructure from synthetic_data_gen_2.py):**
- 6 flawed examples (score < 8) as few-shot seeds
- 3 generation strategies with fixed weights:
  - Structure-first: 50% (template-guided sentence generation)
  - Multi-relation: 30% (two relations in one sentence)
  - Targeted-relation: 20% (focus on specific relation type)
- Same `SEMANTIC_CONCEPTS`, `STRUCTURE_TEMPLATES`, prompt builders — unchanged

## Results

| Metric | Value |
|---|---|
| Generation model | openai/gpt-oss-120b |
| JSON-valid success rate | ~86% per batch |
| Duplicate sentences | 0 |
| Empty-triplet rows | 0 |
| Target count | ~15,000–16,000 new examples |

## Staged training plan

```
Phase 1 (noisy training data):
  Original score<8 examples:     ~11,367
  Newly generated noisy:          ~15,000–16,000
  Total Phase 1 pool:             ~26,000–27,000

Phase 2 (clean refinement data):
  Original score≥9 examples:      8,633 (unchanged, verified quality)
```

## Large file handling

The generated dataset JSONL is too large for GitHub (>100MB). It is stored on
Google Drive (`synthetic_bench_hindie_data_gpt_oss_120b-scored.jsonl` for the
original 20K; `noisy_synthetic_data_3b_model.jsonl` for the newly generated set).

The generation script is at `src/finetune/generate_noisy_data.py`.
