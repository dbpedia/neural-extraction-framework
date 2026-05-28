import requests
import os
import re

# REMOVED: urllib3.disable_warnings (It hides real security issues)

class FactChecker:
    def __init__(self):
        # Fix: Read from Docker Environment Variable
        self.endpoint = os.getenv("DBPEDIA_ENDPOINT", "http://dbpedia.org/sparql")

    def _sanitize_uri(self, uri):
        """Prevents SPARQL Injection and formats for DBpedia."""
        # FIX: Replace spaces with underscores BEFORE stripping bad characters
        uri = uri.replace(' ', '_') 
        clean = re.sub(r'[^a-zA-Z0-9_:/?#=\-\.]', '', uri)
        return clean

    def validate_triple(self, subj, pred, obj):
        subj_uri = f"http://dbpedia.org/resource/{self._sanitize_uri(subj)}"
        pred_uri = f"http://dbpedia.org/ontology/{pred.split(':')[-1]}"
        obj_uri = f"http://dbpedia.org/resource/{self._sanitize_uri(obj)}"

        # TIER 1: Strict Match (Is it currently true?)
        strict_query = f"ASK {{ <{subj_uri}> <{pred_uri}> <{obj_uri}> }}"
        
        # TIER 2: The Mentor's Flex (Are they connected via Wiki-Links or any property?)
        relaxed_query = f"""
        ASK {{
            {{ <{subj_uri}> ?p <{obj_uri}> }}
            UNION
            {{ <{obj_uri}> ?p <{subj_uri}> }}
            UNION
            {{ <{subj_uri}> dbo:wikiPageWikiLink <{obj_uri}> }}
        }}
        """

        try:
            # Check Tier 1
            params = {"query": strict_query, "format": "json"}
            r = requests.get(self.endpoint, params=params, timeout=10)
            if r.status_code == 200 and r.json().get("boolean"):
                return "VALIDATED (Strict Ontology Match)"
                
            # Check Tier 2 (The Graph Algorithm)
            params = {"query": relaxed_query, "format": "json"}
            r = requests.get(self.endpoint, params=params, timeout=10)
            if r.status_code == 200 and r.json().get("boolean"):
                return "VALIDATED (Graph/Wiki-Link Proximity)"
                
            # Tier 3 (Hallucination)
            return "REJECTED (Contradicts World Knowledge)"
            
        except Exception as e:
            return f"Error querying DBpedia: {str(e)}"

if __name__ == "__main__":
    checker = FactChecker()
    print("Testing FactChecker: Ronaldo -> Real Madrid...")
    result = checker.validate_triple("Cristiano_Ronaldo", "dbo:team", "Real_Madrid_CF")
    print(f"Result: {result}")
