from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia

# Create a SPARQLWrapper object with the DBpedia SPARQL endpoint URL
# sparql = SPARQLWrapper("http://dbpedia.org/sparql")

def get_abstract(sparql_wrapper, uri):
    query = f"""
    SELECT ?abstract
    WHERE {{
    <http://dbpedia.org/resource/{uri}> <http://dbpedia.org/ontology/abstract> ?abstract .
    FILTER (LANG(?abstract) = 'en')
    }}
    """
    sparql_wrapper.setQuery(query)
    sparql_wrapper.setReturnFormat(JSON)
    results = sparql_wrapper.query().convert()
    return results['results']['bindings'][0]['abstract']['value']

def get_text_of_wiki_page(article_name):
    article_name_result = wikipedia.page(wikipedia.search(article_name)[0], auto_suggest=False)
    article_name_content = article_name_result.content
    article_name_content.replace("\n", "").replace("\t", "")
    return article_name_content