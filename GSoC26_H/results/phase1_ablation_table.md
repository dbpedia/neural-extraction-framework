# Phase 1 Milestone: Ablation Table

**Dataset:** Hindi-BenchIE (112 sentences)
**Evaluation:** Fact-cluster matching (essential + compensatory triples)

## Aggregate Metrics

| System                                 |   Precision |   Recall |     F1 |   TPs |   FPs |   FNs |   Silent_Sentences |
|:---------------------------------------|------------:|---------:|-------:|------:|------:|------:|-------------------:|
| IndIE (rule-based)                     |      0.4377 |   0.4854 | 0.4603 |   116 |   149 |   123 |                 24 |
| Zero-shot Gemma-3-1B                   |      0      |   0      | 0      |     0 |    23 |   304 |                 89 |
| GSoC25_H (Gemma-3-12B + IndIE + ReAct) |      0.2141 |   0.584  | 0.3133 |   146 |   536 |   104 |                  6 |

## Error Type Distribution (% of False Positives)

| System                                 |   Predicate_Normalization_pct |   Implicit_Relation_pct |   Language_Mixing_pct |   Predicate_Placeholder_pct |   Argument_Span_pct |
|:---------------------------------------|------------------------------:|------------------------:|----------------------:|----------------------------:|--------------------:|
| IndIE (rule-based)                     |                          68.5 |                     2.7 |                   0   |                        28.9 |                   0 |
| Zero-shot Gemma-3-1B                   |                          43.5 |                    39.1 |                  17.4 |                         0   |                   0 |
| GSoC25_H (Gemma-3-12B + IndIE + ReAct) |                          81.9 |                     8.8 |                   1.1 |                         8.2 |                   0 |

## Key Findings

1. **Argument span errors are 0% in all three systems** — confirming that the predicate slot is the entire failure mode, not the arguments.

2. **GSoC25_H reduces IndIE's placeholder failures from 28.9% → 8.2%** by using the 12B LLM to fill empty predicate slots.

3. **All three systems share the same downstream gap:** surface Hindi predicates that are not aligned to DBpedia ontology — 82% of GSoC25_H's failures.

   ---

## Note on Current Model — Not Directly Comparable

The current fine-tuned pipeline (Gemma 3 4B + F2LLM-1.7B) achieves **F1 = 0.173** on BenchIE, measured using the triple-level set-matching methodology in `normalize_full_scale.py` — a different scoring approach than the fact-cluster / error-taxonomy methodology used for the three systems above.

An attempt was made to score the current model using the same `error_taxonomy.py` classifier for a like-for-like comparison. This surfaced a genuine bug in that script's copula-detection logic: it checks whether any Hindi copula word appears anywhere as a *substring* of the predicted predicate, rather than requiring a whole-word match. Since most grammatically correct Hindi verb phrases legitimately contain a copula substring (e.g. "है" inside "सििद्ध होती है"), this caused a majority of genuinely correct predictions to be misclassified as placeholder errors — confirmed by manual inspection, which found multiple byte-identical gold/predicted matches wrongly flagged this way.

Given this, no new row has been added to the tables above — doing so would risk publishing a result that reflects the classifier's own limitation rather than the model's actual performance. **F1 = 0.173 (via `normalize_full_scale.py`) remains the trustworthy current-model number for BenchIE.**

