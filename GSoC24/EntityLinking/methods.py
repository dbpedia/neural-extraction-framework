import requests
import pandas as pd
import redis


def EL_DBpedia_lookup(query: str, max_results: int):
    """Perform entity linking using DBpedia lookup service on
    the given query.

    Args:
        query (str): The entity mention that you want to link to a resource in dbpedia.
        max_results (int): The maximum candidate entities/resources that you want to fetch.

    Returns:
        List: A list of the candidate entities/resources.
    """
    response = requests.post(
        f"https://lookup.dbpedia.org/api/search?query={query}&format=json&maxResults={max_results}"
    )
    docs = response.json()["docs"]
    resources = [d["resource"][0] for d in docs]
    return resources


def EL_DBpedia_spotlight(query: str, nlp):
    """Perform entity linking using DBpedia spotlight using spacy component.

    Args:
        query (str): The entity mention that you want to link to a resource in dbpedia.
        nlp : Spacy model for english language added with a component
        called `dbpedia_spotlight`.

        The nlp pipeline should output something similar to below.
        ```
        >> nlp.pipeline
        [('dbpedia_spotlight',
        <spacy_dbpedia_spotlight.entity_linker.EntityLinker at --/----/--->)])
        ```

    Returns:
       tuple(List, List): Returns a tuple of lists, the first list contains the entities identified
       by spotlight, and the second list contains the entities it has linked them to.
    """
    doc = nlp(query)
    texts = []
    uris = {}
    labels = []
    for ent in doc.ents:
        texts.append(ent.text)
        labels.append(ent.label_)
        if ent.text + "_" + ent.label_ not in uris:
            uris[ent.text + "_" + ent.label_] = []
        uris[ent.text + "_" + ent.label_].append(ent.kb_id_)
    return texts, uris


def EL_redis_db(
    query: str,
    redis_client_forms: redis.client.Redis,
    redis_client_redir: redis.client.Redis,
):
    """Performs entity linking using a redis database.

    Args:
        query (str): The entity mention that you want to link to a resource in dbpedia.
        redis_client_forms (redis.client.Redis): The redis client object used to query the surface forms.
        redis_client_redir (redis.client.Redis): The redis client object used to handle redirects.

    Returns:
        DataFrame: A pandas dataframe with candidate entity scores and other info. We can then
        select only the candidate URIs.
    """
    from GSoC23.EntityLinking.el_utils import lookup

    return lookup(
        term=query,
        redis_client_forms=redis_client_forms,
        redis_client_redir=redis_client_redir,
    )


def EL_GENRE(entity, model, tokenizer):
    """A method to perform entity linking for entity-mentions annotated
    in sentences using the GENRE model.

    ```
    tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()

    sentences = [
        "[START_ENT] England [END_ENT] won the cricket world cup in 2019",
        "I just finished reading [START_ENT] 'The Jungle Book' [END_ENT]",
        "India is a country in Asia. [START_ENT] It [END_ENT] has a rich cultural heritage"
    ]

    EL_GENRE(annotated_sentences=sentences, model=model, tokenizer=tokenizer)

    ```

    Args:
        annotated_sentences (list): A list of sentences annotated with entity-mentions
        model : GENRE model from huggingface hub
        tokenizer : Appropriate tokenizer for GENRE model
    """
    outputs = model.generate(
        **tokenizer(entity, return_tensors="pt", padding=True),
        num_beams=5,
        num_return_sequences=5,
        # OPTIONAL: use constrained beam search
        # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    )

    entites = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # These entites are in the form of wikipedia page titles. Need to
    # add the https://dbpedia/resource to each of them as postprocessing step
    return entites

def EL_GENRE_pipeline(pipe, annotated_sentences):
    """A method to perform entity linking for entity-mentions annotated
    in sentences using the GENRE model with transformers pipeline. 
    Works faster than the approach without pipeline.

    Args:
        pipe (transformers.pipelines.text2text_generation.Text2TextGenerationPipeline): The transformers pipeline object.
        annotated_sentences (list): List of annotated sentences.

    Returns:
        list: The list of entities.

    Example:
        ans = [annotate_sentence(s,m) for s,m in [
        ("Messi plays for Argentina", "Messi"),
        ("Messi plays for Argentina", "Argentina"),
        ("India is in Asia", "India")
        ]]

        genre_pipe = genre_pipe = pipeline(
            model="facebook/genre-linking-blink", 
            tokenizer="facebook/genre-linking-blink")
        pipe_results = genre_pipe(ans)

        entities = [e['generated_text'] for e in res]
    """
    pipe_results = pipe(annotated_sentences)
    entities = [e['generated_text'] for e in pipe_results]
    return entities
