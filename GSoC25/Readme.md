# Neural Extraction Framework @DBpedia - GSoC 2025

|   Project Details     |                                                                                                                                                                                               |
|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GSoC Project | [Neural Extraction Framework GSoC'25 @DBpedia](https://summerofcode.withgoogle.com/myprojects/details/uQUHx6jo)                                                                           |
| Contributor | [Gandharva Naveen](https://www.linkedin.com/in/gandharva-naveen)                                                                                                                        |
| Mentors | [Tommaso Soru](), [Abdulsobur Oyewale](), [Diego Moussallem](), [Ronit Banerjee]() |
| Blogs | [GSoC-2025 Gandharva Naveen](https://github.com/Gnav3852/neural-extraction-framework/wiki/)                                                                                                                                   |

### What is Neural Extraction Framework?

The Neural Extraction Framework (NEF) is a system designed to extract structured relational knowledge—called RDF triples—directly from unstructured text. NEF focuses on uncovering the hidden relationships embedded in natural language. It uses large language models (LLMs), embedding-based retrieval, and ontology alignment to identify entities, match predicates to a known ontology (like DBpedia), and produce machine-readable [relational triples](https://en.wikipedia.org/wiki/Semantic_triple) such as (Albert Einstein — award — Nobel Prize in Physics).

The goal of NEF is to move beyond static, infobox-based extraction toward a dynamic, intelligent pipeline that continuously learns and adapts. By integrating embedding search, Redis grounding, and LLM reasoning, NEF can both expand existing knowledge graphs and validate extracted information through similarity scoring and ontology mapping. In essence, NEF bridges text and knowledge representation: transforming open text into structured, queryable data that allows machines to “understand the world, one triple at a time.”


### Code structure
All directories/files have detailed instruction about how to use them in the git wiki posts, but the main folder is NEF.
```
GSoC25/NEF
 ┣ ground_truth
 ┣ test
 ┣ Bench.py
 ┣ Embeddings.py
 ┣ Webscrape.py
 ┣ NEF.py
```
[Example](https://colab.research.google.com/drive/1eeRWKEnMHqCbOMwvC3NAifZVkStjUTSB?usp=sharing)

### Installations 
Run the command below to install all requirements of the project at once (preferably in a virtual environment). And set env variables.
```
!pip install -r requirements.txt

GEMINI_API_KEY = 
NEF_REDIS_HOST = 
NEF_REDIS_PORT = 
NEF_REDIS_PASSWORD = 

```

### Run from command line

You need to precompute the DBpedia embeddings before running the NEF.
```
!python NEF/Emeddings.py
```
Then you can run the pipeline like this:
```
!python NEF/NEF.py -s "Albert Einstein developed the theory of relativity and won the Nobel Prize." \
  --embeddings embeddings.npy --predicates predicates.csv

```


### Project Workflow

<img width="770" height="980" alt="NEF Workflow (1)" src="https://github.com/user-attachments/assets/d959fb05-5258-426c-a912-1a8534200e9e" />


### Future scope

This project introduced a neural framework for entity–relation extraction that combines LLM-based triple generation, Redis-backed entity linking, and embedding-based predicate retrieval. The Enhanced NEF pipeline improves grounding accuracy and contextual understanding over earlier end-to-end methods by enforcing strict validation, predicate thresholding, and disambiguation through Gemini models.

Future work will focus on expanding semantic coverage and adaptability. Planned directions include integrating FAISS-based entity retrieval for broader coverage, refining predicate clustering and ontology alignment to improve generalization, and enabling incremental updates to keep Redis and embedding stores synchronized with evolving knowledge sources.
