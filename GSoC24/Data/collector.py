from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia
import re

# Create a SPARQLWrapper object with the DBpedia SPARQL endpoint URL
sparql = SPARQLWrapper("http://dbpedia.org/sparql")
print("Sparql wrapper created")


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
    return results["results"]["bindings"][0]["abstract"]["value"]


def get_text_of_wiki_page(article_name: str):
    """Given an article name(need not be exact title of page),
    return the textual content of the wikipedia article.
    We do a search for the articles and select the top-1 result, in case
    where the article name is not the exact title.

    Args:
        article_name (str): Name of a wikipedia article

    Returns:
        str: The text of that article.
    """
    article_name_result = wikipedia.page(
        wikipedia.search(article_name)[0], auto_suggest=False
    )
    article_name_content = article_name_result.content
    article_name_content.replace("\n", "").replace("\t", "")
    return article_name_content


def get_wikiPageWikiLink_entities(entity, sparql_wrapper=sparql):
    """A function to fetch all entities connected to the given entity
    by the dbo:wikiPageWikiLink predicate.

    Args:
        entity (_type_): The source entity, the wiki page we are parsing.
        Example --> <http://dbpedia.org/resource/Berlin_Wall>.
        sparql_wrapper : The SPARQL endpoint. Defaults to sparql.

    Returns: List of entities.
    """

    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT ?connected_entities
    WHERE{{
    {entity} dbo:wikiPageWikiLink ?connected_entities .
    }}
    """
    sparql_wrapper.setQuery(query)
    sparql_wrapper.setReturnFormat(JSON)
    results = sparql_wrapper.query().convert()
    entities = [
        r["connected_entities"]["value"] for r in results["results"]["bindings"]
    ]
    return entities


def get_only_wikiPageWikiLink(entity, sparql_wrapper=sparql):
    """For a given entity(the current wiki page), returns all entities
    that are connected with only the dbo:wikiPageWikiLink predicate and no
    other predicate.
    Args:
        entity : The source entity, the current wiki page.
        Example --> <http://dbpedia.org/resource/Berlin_Wall>.
        sparql_wrapper : The SPARQL endpoint. Defaults to sparql.

    Returns: List of entities.
    """

    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>

    SELECT ?o
    WHERE {{
      {{
        SELECT ?o
        WHERE {{
          {entity} dbo:wikiPageWikiLink ?o .
        }}
      }}
      MINUS
      {{
        {entity} ?p ?o .
        FILTER (?p != dbo:wikiPageWikiLink)
      }}
    }}
    """
    sparql_wrapper.setQuery(query)
    sparql_wrapper.setReturnFormat(JSON)
    results = sparql_wrapper.query().convert()
    return results["results"]["bindings"]


def get_sentences_with_entities(article_text: str, subject: str, object_entity: str):
    """Extract all sentences from the article text where both the subject and object entities are mentioned.
    
    Args:
        article_text (str): The full text of the Wikipedia article
        subject (str): The subject entity name to search for
        object_entity (str): The object entity name to search for
        
    Returns:
        list: List of sentences containing both entities
    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    from nltk.tokenize import sent_tokenize
    
    # Tokenize the article into sentences
    sentences = sent_tokenize(article_text)
    
    # Find sentences containing both entities (case-insensitive)
    relevant_sentences = []
    subject_lower = subject.lower()
    object_lower = object_entity.lower()
    
    for sentence in sentences:
        if subject_lower in sentence.lower() and object_lower in sentence.lower():
            relevant_sentences.append(sentence.strip())
    
    return relevant_sentences


def get_predicates_between(subject_uri: str, object_uri: str, sparql_wrapper=sparql):
    """Fetch all predicates that directly connect a subject and object in DBpedia.
    
    Args:
        subject_uri (str): The subject entity URI
        object_uri (str): The object entity URI
        sparql_wrapper: The SPARQL endpoint. Defaults to sparql.
        
    Returns:
        list: List of predicate URIs that connect the subject and object
    """
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT DISTINCT ?predicate
    WHERE {{
        <{subject_uri}> ?predicate <{object_uri}> .
        FILTER(?predicate != dbo:wikiPageWikiLink)
    }}
    """
    
    sparql_wrapper.setQuery(query)
    sparql_wrapper.setReturnFormat(JSON)
    results = sparql_wrapper.query().convert()
    
    predicates = [r["predicate"]["value"] for r in results["results"]["bindings"]]
    return predicates


def get_entity_types(entity_uri: str, sparql_wrapper=sparql):
    """Fetch all DBpedia ontology classes/types for a given entity.
    
    Args:
        entity_uri (str): The entity URI
        sparql_wrapper: The SPARQL endpoint. Defaults to sparql.
        
    Returns:
        list: List of type/class URIs for the entity
    """
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    
    SELECT DISTINCT ?type
    WHERE {{
        <{entity_uri}> rdf:type ?type .
        FILTER(STRSTARTS(STR(?type), "http://dbpedia.org/ontology/"))
    }}
    """
    
    sparql_wrapper.setQuery(query)
    sparql_wrapper.setReturnFormat(JSON)
    results = sparql_wrapper.query().convert()
    
    types = [r["type"]["value"] for r in results["results"]["bindings"]]
    return types
