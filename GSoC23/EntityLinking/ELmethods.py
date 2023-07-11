import requests
import spacy_dbpedia_spotlight
import spacy
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
    docs = response.json()['docs']
    resources = [d['resource'][0] for d in docs]
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
        if ent.text+"_"+ent.label_ not in uris:
            uris[ent.text+"_"+ent.label_]=[]
        uris[ent.text+"_"+ent.label_].append(ent.kb_id_)
    return texts, uris

def EL_redis_db(query: str, 
        redis_client_forms: redis.client.Redis, 
        redis_client_redir: redis.client.Redis):
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
    return lookup(term=query, 
    redis_client_forms = redis_client_forms,
    redis_client_redir = redis_client_redir)
