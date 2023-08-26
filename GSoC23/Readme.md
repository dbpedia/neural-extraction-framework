# Neural Extraction Framework @DBpedia - GSoC 2023

|   Project Details     | |
|-------------|-------------|
| GSoC Project | [Neural Extraction Framework GSoC'23 @DBpedia](https://summerofcode.withgoogle.com/programs/2023/projects/cKuagkf8)        |
| Contributor | [Aakash Thatte](https://www.linkedin.com/in/aakash-thatte/) |
| Mentors | [Tommaso Soru](https://github.com/mommi84), [Diego Moussallem](https://github.com/DiegoMoussallem), [Ziwei Xu](https://github.com/zoeNantes)|
| Blogs | [GSoC-2023 Aakash Thatte](https://sky-2002.github.io/) |

### Code structure
All directories contain a `notebooks` directory which has notebooks with exploration/experimentation code for the models and methods used. 
```
📦GSoC23
 ┣ 📂CoreferenceResolution
 ┣ 📂Data
 ┣ 📂EntityLinking
 ┣ 📂NER
 ┣ 📂RelationExtraction
 ┣ 📂Validation
```

### Installations 
I have provided the requirements file, but you can go ahead with only the packages below as well.
```
!pip install wikipedia
!pip install transformers
!pip install nltk
!pip install fuzzywuzzy
!pip install rdflib
!pip install SPARQLWrapper
!pip install redis
!pip install thefuzz
!pip install levenshtein
```

For spacy `en_core_web...` models,
use `spacy.download('en_core_trf')`

### Project workflow
```mermaid
graph TD
    wiki_page[Wikipedia Page] --Extract plain text--> pure_text[Pure text]

    pure_text--coreference resolution-->coref_resolved_text[Coreference Resolved Text]
    coref_resolved_text-->sentences[Sentences]
    sentences-->rebel(REBEL)
    rebel--as text-->entities[Entities]
    rebel--as text-->relations[Relations]
    
    relations--get embedding-->vector_similarity(Vector similarity with label embeddings);
    vector_similarity-->predicate_uris[Predicate URIs]

    sentences-->annotation_for_genre(Annotate entities in text)

    entities-->annotation_for_genre

    annotation_for_genre--annotated text-->genre[GENRE]

    genre-->entity_uris[Entity URIs]
    
    entity_uris-->triples[Triples]
    predicate_uris-->triples[Triples]
    triples--Validate-->final_triples[Final triples]
```