# Neural Hindi Wiki Triple Extraction Pipeline

This project aims to enhance and evaluate the relation extraction pipeline for Hindi Wikipedia articles, utilizing a combination of state-of-the-art language models and rule-based methods. 
This repo builds on the work done in GSoC24 for the Hindi chapter.

## Components

### LLM_IE
Contains code for the plug-and-play evaluation framework  for measuring how well small-language-models (SLMs) extract `(subject, relation, object)` triplets from Hindi text using the official [*Hindi-BenchIE*](https://github.com/ritwikmishra/hindi-benchie) benchmark.

It also contains the finetuning folder which currently only has the synthetic data generation and filtering scripts. 

Generated data and all extraction results can be found at this [Google Drive link](https://drive.google.com/drive/folders/1rYZbLRgZRwfyVJJvsxhqqODQvIrA1JCs)

TODO: A lot of this code overlaps with the ReAct/ module code. Can be cleaned up.


### Link Prediction
Performed various experiments with link prediction modules using the DBpedia hindi data upto May 2025. Few things we tried:
- Two different KGE models - TransE & ConvE
- Incorporating and comparing MURIL Embeddings as starting points (initial embeddings) dimensionally reduced using PCA

Lot of scope for future work here like hyperparameter tuning, advance entity representation using embeddings of first para of wikipedia text etc. Of course using the english graph in conjunction with the hindi dataset will enhance such a model. 

### IndIE Enhancement
The idea here is simple. 

IndIE relies on a three-stage pipeline to produce its final output: Sentence (input) -> Chunking (P1) -> Creation of MDT (P2) -> Handwritten Rules (P3) -> Triplets (Output)

Let’s look at what each of these phases does at a high level:

1. Chunking: This is the process of breaking down the input sentence into meaningful multi-word units. Each chunk represents:
Noun phrases (entities, objects)
Verb phrases (actions, states)
Prepositional phrases (relationships, locations, times)
Other syntactic units
For example: Sentence: कार्यरूप जगत को देखकर ही शक्तिरूपी माया की सििद्ध होती है . Chunks: [‘कार्यरूप जगत को’, ‘देखकर ही’, ‘शक्तिरूपी माया की’, ‘सििद्ध होती है’, ’.’]

2. Merged Dependency Tree: The MDT shows how chunks relate to each other grammatically, helping identify subjects, objects, and modifiers. For example for the same above sentence, we would derive the following MDT:

```code
Root Phrase: "सििद्ध होती है" (main predicate/action)
Dependency Relations:
  - कार्यरूप जगत को->obj
  - देखकर ही->advcl
  - शक्तिरूपी माया की->nmod
  - सििद्ध होती है->root
  - .->0
```

3. Handwritten Rules for Triplet Extraction: For the final output, indIE uses over 100 handwritten rules to derive the final output. This is the brittle component that we want to enhance.

The idea here was to try replacing the component of hand written rules. Instead of relying on these, we pass the chunks and MDT to a small LM like Gemma 3. We explain to Gemma 3 deeply what a dependency tree is and how it works. Once the model is provided with all this information, it should ideally be able to generate better triplets than before.

We ran multiple experiments here and eventually after multiple iterations we were able to achieve 66% recall up from 50% from the baseline indIE model. 

### Predicate Linking 
This module was an addition to the pipeline from last year to map the extracted relations to the DBpedia ontology.
We map Hindi predicate phrases to English DBpedia ontology properties (`dbo:*`) using a hybrid approach:

- Graph evidence: direct `dbo:*` edges between `dbr:S` and `dbr:O` (both directions)
- Lexical match: SPARQL label search (Hindi/English)
- Type compatibility: `rdfs:domain`/`rdfs:range` vs `rdf:type(S)`/`rdf:type(O)`
- Multilingual sentence embeddings: local index over the ontology built from `GSoC24_H/ontology_input/ontology--DEV_type=parsed.ttl`


Scoring (higher is better): 
```python
w_graph, w_emb, w_lex, w_type = 0.4, 0.3, 0.2, 0.1
# weighted sum -> prioritize graph, then embedding, then lexical, then type
score = w_graph * graph_score + w_emb * emb_score + w_lex * lex_score + w_type * type_score
```

Outputs include `property_uri`, labels, score, direction (`S->O` or `O->S`), and evidence breakdown.


## How to Run
- Install dependencies by running in the GSoC25/ root 

```pip install -r requirements.txt```

### SLM guided IE
- Lives in llm_IE/
- Refer llm_IE/README.md

### Link Prediction
- Lives in link_prediction/
- 2 notebooks - one for analysis, one for actual implementation of the models
- Run the notebooks using jupyter individually

### Predicate Linking (Hindi → DBpedia Ontology)
Implementation lives in `src/predicate_linking.py`.

- CLI (coref → RE → EL → predicate linking):
```bash
python GSoC25_H/src/start.py \
  --input_dir GSoC25_H/input \
  --do_coref --do_rel --do_el --do_prop_link --verbose
```
Logs progress phase-wise (graph, lexical, embedding, typing, scoring) and prints top candidate.

Automated creation of ontology property and embeddings files upon first run. Cached artifacts written under `GSoC25_H/models/ontology/`:
- `dbpedia_property_catalog.json`
- `dbpedia_property_embeddings.npy`
- `dbpedia_property_index.json`

To reset caches, delete the files above and rerun.

### Streamlit Demo
From inside the GSoC25_H/ folder Run
```streamlit run src/demo.py```


### Data
All experiments, results and generated data was uploaded to Google Drive. Find it [here](https://drive.google.com/drive/folders/1fgbZdGAnLhIASQRKEuyOwbBFvFZvJt_R?usp=sharing). 


### What Work was Done
* Streamline the existing pipeline and make it easy to run
* Implement and evaluate new triplet extraction methods using Small Language Models (SLM)
* Enhance the IndIE method by integrating SLMs in the last stage of IndIE
* Implement link prediction and evaluate to predict missing links using the existing hindi KG
* Implement predicate linking to dbpedia ontology in the existing framework
* Deploy the SPARQL endpoint and test its performance


## Progress Overview

### Week 1-2

- Streamlined the existing pipeline from GSoC'24, made it simple to run (steps below)
- Rebased and raised final PR for addings Hindi mappings and extractors - https://github.com/dbpedia/extraction-framework/pull/776

### Week 3 

- Simple framework for evaluating the performance of SLMs w.r.t various prompt strategies on the Hindi BenchIE dataset
- Lives in **llm_IE/** directory
- Read https://advenk.github.io/av-blog/posts/gsoc-25/gsoc-week-3/

### Week 4 

- Used ReAct framework for prompting
- Link prediction setup

### Week 5
- Link Prediction notebook completed
- Integrated gemma into the last stage of the IndIE pipeline

### Week 6
- Experimented and evaluated IndIE + gemma pipeline from previous week
- Achieved ~65% recall
- Designed finetuning plan

### Week 7
- Wrote the synthetic data generation script
- Deployed and performance tested the SPARQL endpoint (lives in https://github.com/advenk/virtuoso-sparql-endpoint-quickstart/tree/gsoc25_hindi_chapter)

### Week 8-9
- Paper Review of "Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning"
- Document the SPARQL endpoint (temporarily here: http://95.217.58.54:8890/sparql)
- Synthetic Data Generation script change

### Week 10
- Evaluated code for entity linking in depth
- Identified Gaps
- EL setup

### Week 11
- Implemented predicate linking to dbpedia ontology
- Integrated into existing pipeline

### Week 12
- EL normalisation to dbpedia resources from wikidata ID
- Translated and scored lexical similarity for predicates
- Clean up and wrap up of code
- Consolidating a

## Known Limitations and Future Work
- Enhance predicate linking to include type linking. It currently only handles relational linking but not type linking. Type linking would require us to identify if a triplet is of the form (subject, "है", object) and then link it as the subject -> type -> object. For other triplets (relational), this is handled. The major difference is the "object" in a type linking is a class in the ontology whereas in regular relational linking this is a property. 
- Enhance link prediction using English knowledge graph, hyperparam tuning, better embeddings for disambiguation etc
- Finetune gemma3 using latest synthetically generated data then gauge performance
- Refine the pipeline for enhanced accuracy.


## Other related links
- Weekly Blog: https://advenk.github.io/av-blog/
- Unmerged PR for Hindi mappings and extractors: https://github.com/dbpedia/extraction-framework/pull/776 - Can be merged only after updating the mappings via UI.

## Archived
- SPARQL performance testing code (not to be merged) - https://github.com/advenk/virtuoso-sparql-endpoint-quickstart/tree/gsoc25_hindi_chapter

- Hindi SPARQL temporary endpoint deployed at http://95.217.58.54:8890/sparql (archived as of 15/09/2025)