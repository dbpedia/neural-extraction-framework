import os
import pandas as pd
from rdflib import Graph
from GSoC25.pipeline2 import EnhancedNEFPipeline

# --- Load DBpedia ontology ---
g = Graph()
g.parse("http://dief.tools.dbpedia.org/server/ontology/dbpedia.owl")

# Pull predicate labels/comments to enrich valid matches
query = """
SELECT ?p ?label ?comment WHERE {
  ?p a rdf:Property ; rdfs:label ?label .
  OPTIONAL { ?p rdfs:comment ?comment . FILTER(lang(?comment) = "en") }
  FILTER(regex(str(?p), "http://dbpedia.org/ontology/"))
  FILTER(lang(?label) = "en")
}
"""
df_props = pd.DataFrame(g.query(query), columns=["p", "label", "comment"])
df_props["p"] = df_props["p"].astype(str)

def predicate_exists(uri: str) -> bool:
    """Check via SPARQL ASK if a predicate URI exists."""
    ask = f"ASK WHERE {{ <{uri}> a rdf:Property . }}"
    return bool(g.query(ask))

def build_predicate_data(uri: str):
    """Return metadata (label/comment) for a validated predicate."""
    match = df_props[df_props["p"] == uri]
    if match.empty:
        return {"uri": uri, "description": "", "domain": "", "range": ""}
    row = match.iloc[0]
    return {
        "uri": uri,
        "description": str(row["comment"] or row["label"]),
        "domain": "",
        "range": ""
    }

# --- Get Gemini candidate URIs and validate each ---
pipeline = EnhancedNEFPipeline(os.environ["GEMINI_API_KEY"])
relation_text = "was born in"
top_candidates = pipeline.get_candidate_predicates(relation_text)[:5]

validated_predicates = [
    build_predicate_data(uri)
    for uri in top_candidates
    if predicate_exists(uri)
]

print("Candidate URIs:", top_candidates)
print("Validated predicates:", validated_predicates)
