from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Tuple

DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


def search_dbpedia_entity(label: str, limit: int = 5) -> List[Tuple[str, str]]:
    """
    Search DBpedia for entities with the given label. Returns a list of (URI, label) tuples.
    """
    sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)
    # Use exact string matching with proper language tags
    query = f'''
    SELECT ?uri ?label WHERE {{
      ?uri rdfs:label ?label .
      FILTER (?label = "{label}"@en)
    }} LIMIT {limit}
    '''
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        candidates = []
        for result in results["results"]["bindings"]:
            uri = result["uri"]["value"]
            label = result["label"]["value"]
            candidates.append((uri, label))
        return candidates
    except Exception as e:
        print(f"Error querying DBpedia: {e}")
        return [] 