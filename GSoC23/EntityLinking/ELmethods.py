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

def EL_DBpedia_spotlight(query: str, nlp: spacy.lang.en.English):
    """Perform entity linking using DBpedia spotlight using spacy component.

    Args:
        query (str): The entity mention that you want to link to a resource in dbpedia.
        nlp (spacy.lang.en.English): Spacy model for english language added with a component 
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
    return lookup(term=query, 
    redis_client_forms = redis_client_forms,
    redis_client_redir = redis_client_redir)

def calculate_redirect(source, client):
    result = client.get(source)
    if result is None:
        return source if type(source) is str else source.decode('utf-8')
    return calculate_redirect(result, client)

def query(surface_form, client):
    raw = client.hgetall(surface_form)
    if len(raw) == 0:
        return pd.DataFrame(columns=['entity', 'support', 'score'])
    
    out = []
    for label, score in raw.items():
        out.append({'entity': label.decode('utf-8'), 'support': int(score)})
    df_all = pd.DataFrame(out)
    df_all['score'] = df_all['support'] / df_all['support'].max()
    
    return df_all.sort_values(by='score', ascending=False).reset_index(drop=True)

def lookup(term, top_k=5, thr=0.01, redis_client_forms=None, redis_client_redir=None):
    df_temp = query(term, redis_client_forms)
#     display(df_temp)
    df_temp['entity'] = df_temp['entity'].apply(lambda x: calculate_redirect(x, redis_client_redir))
    if len(df_temp) == 0:
        return pd.DataFrame(columns=['entity', 'support', 'score'])
    df_final = df_temp.groupby('entity').sum()[['support']]
    df_final['score'] = df_final['support'] / df_final['support'].max()
    return df_final[df_final['score'] >= thr].sort_values(by='score', ascending=False)[:top_k]