# Embedding Model Comparison — Week 3 Results

**Task:** Replace/compare `paraphrase-multilingual-MiniLM-L12-v2` with newer models
for Hindi predicate → DBpedia property alignment.

**Test setup:**
- Property catalog: full DBpedia ontology TTL (`ontology--DEV_type=parsed.ttl`),
  filtered from 2,891 raw properties to **2,710 clean properties** (removing codes,
  IDs, biological database terms, COVID-tracking properties, underscore-named entries)
- Plus 2 manually re-added properties absent from this TTL snapshot (`dbo:winner`,
  `dbo:lyricist`) with minimal Hindi anchoring
- Test set: 7 predicates with independently verified correct dbo: answers
- Metric: **Recall@15** — does the correct property appear anywhere in the top-15?

## Results

| Model | Size | Recall@15 | Avg rank (when found) | Speed (2,710 props) | Notes |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M | **5/7 = 71.4%** | 9.0 | ~1 sec (GPU) | No instruct prefix needed; symmetric paraphrase model |
| `intfloat/multilingual-e5-large-instruct` | 560M | **7/7 = 100%** | 3.7 | ~2-3 sec (GPU) | Requires `"Instruct: ...\nQuery: "` prefix on queries only |
| `codefuse-ai/F2LLM-v2-1.7B` | 1.7B | **7/7 = 100%** | 3.6 | ~51 sec (GPU) | Requires same instruct prefix; ~20x slower than e5 |

## Per-predicate breakdown

| Predicate | Correct dbo: | MiniLM rank | e5 rank | F2LLM rank |
|---|---|---|---|---|
| जीता | dbo:winner | MISS | 1 | 1 |
| संगीत रचना की | dbo:musicComposer | 6 | 5 | 1 |
| जिसका निर्माण करते थे | dbo:builder | 9 | 8 | 4 |
| देहांत हो गया | dbo:deathPlace | 10 | 5 | 9 |
| लिखे हैं | dbo:author | 7 | 1 | 1 |
| प्रकाशित हुए हैं | dbo:publisher | MISS | 1 | 7 |
| स्थित है | dbo:location | 13 | 5 | 2 |

## Key findings

1. **e5 and F2LLM both reach 100% recall@15**, vs MiniLM's 71.4% — but only when used
   with their documented instruct prefix convention. Using either model without the prefix
   (symmetric encoding) produces compressed, non-discriminative scores identical to
   KaLM-Embedding's failure mode.

2. **e5 is the practical choice over F2LLM**: tied recall, ~20x faster, smaller
   embedding dimension (1024 vs 2048 dims), less storage/memory.

3. **MiniLM remains the correct choice for the curated 73-property fast-path pipeline**
   (`alignment.py`) because it was specifically trained for short-phrase paraphrase
   matching and its 100% precision on the curated set holds — the full-ontology
   setting (2,710 properties) is a harder task where the larger models outperform it.

4. **Trash-bin properties confirmed**: `dbo:winner` and `dbo:lyricist` appear in the
   top-10 for almost any Hindi predicate across all three models, regardless of semantic
   relevance. This is a model-inherent effect (these properties embed near the centroid
   of the vector space) — not fixable by description manipulation. It is exactly why
   the GPT-OSS-120B disambiguation stage is necessary.

## Decision

**Production pipeline going forward:**
```
Stage 1:  f2llm 1.7B fine tuned on gold set created by f2llm 8b 
         → top-15 candidates per predicate (no threshold)
Stage 2: openai/gpt-oss-120b (NVIDIA hosted)
         → disambiguates using full sentence context + 15 diverse few-shot examples
         → outputs final dbo: property or NONE
```

## Previous KaLM-Embedding investigation (Week 2)

`HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2.5` was tested in 4 configurations
and rejected:

| Attempt | Result |
|---|---|
| Symmetric encoding | 98.5% raw alignment, ~0% usable precision (trash-bin matches) |
| Recalibrated threshold (0.78) | 16.2% aligned, ~69% precision |
| Asymmetric (documented correct usage) | Broken scores (0.11–0.30), nonsense matches |
| RoPE patch (`transformers>=5.2.0`) | Crash eliminated but embeddings still broken |
| vLLM fallback | Broke Colab's entire torch/torchvision stack |

Conclusion: KaLM's bidirectional attention mechanism is incompatible with the
`transformers==4.46.3` environment used in this project.
