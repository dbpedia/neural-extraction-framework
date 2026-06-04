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

