from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Dict, Optional, Union
import pandas as pd
import math

def batch_dbpedia_uri_lookup(
    canonical_names: List[str],
    output_format: str = "dataframe",
    chunk_size: int = 5
) -> Union[pd.DataFrame, List[Dict], str]:
    """
    Batch lookup of DBpedia URIs for a list of canonical names using multiple small SPARQL queries.
    Args:
        canonical_names: List of canonical names (e.g., 'Apple_Inc.').
        output_format: 'dataframe', 'json', or 'list'.
        chunk_size: Number of names per SPARQL query (default: 5).
    Returns:
        DataFrame, JSON string, or list of dicts with 'canonical_name' and 'dbpedia_uri'.
    """
    endpoint = "https://dbpedia.org/sparql"
    results = []
    total = len(canonical_names)
    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        batch = canonical_names[i * chunk_size : (i + 1) * chunk_size]
        print(f"[PROGRESS] Processing batch {i+1}/{n_chunks} ({len(batch)} names)...")
        sparql = SPARQLWrapper(endpoint)
        values = " ".join(f'"{name}"@en' for name in batch)
        query = f'''
        SELECT ?canonical_name ?uri WHERE {{
          VALUES ?canonical_name {{ {values} }}
          ?uri rdfs:label ?canonical_name .
          FILTER (lang(?canonical_name) = 'en')
        }}
        '''
        print(f"[DEBUG] SPARQL Query for batch {i+1}/{n_chunks}:\n", query)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            batch_results = sparql.query().convert()
            print(f"[DEBUG] Raw SPARQL results for batch {i+1}:\n", batch_results)
            uri_map = {r["canonical_name"]["value"]: r["uri"]["value"] for r in batch_results["results"]["bindings"]}
        except Exception as e:
            print(f"[ERROR] SPARQL query failed for batch {i+1}: {e}")
            uri_map = {}
        for name in batch:
            results.append({
                "canonical_name": name,
                "dbpedia_uri": uri_map.get(name)
            })
        print(f"[PROGRESS] Completed batch {i+1}/{n_chunks}.")
    if output_format == "dataframe":
        return pd.DataFrame(results)
    elif output_format == "json":
        import json
        return json.dumps(results, indent=2)
    else:
        return results 