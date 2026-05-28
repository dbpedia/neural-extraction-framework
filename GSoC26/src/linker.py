import requests
import re
from rapidfuzz import fuzz
import spacy
from redis_client import RedisClient

# Load Spacy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ Warning: Spacy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

class HybridLinker:
    def __init__(self):
        self.redis = RedisClient()
        # Removed hardcoded weights/logic as requested for GSoC prototype

    def resolve_entity(self, term, context=""):
        # 1. Check Redis for Redirects (Fastest - O(1))
        # e.g., "Barca" -> "FC Barcelona"
        resolved_term = self.redis.get_redirect(term)
        
        # 2. Get Candidates from Redis (Simulating DBpedia Lookup)
        # e.g., "Al-Nassr" -> ["Al-Nasr SC", "Al-Nassr FC"]
        candidates = self.redis.get_entities(resolved_term)
        
        if not candidates:
            # Return a list to keep return types consistent
            return resolved_term, ["Unresolved"]

        # 3. Simple Disambiguation (No more hardcoded heuristics)
        # If multiple candidates, pick the one that matches the string best
        best_match = candidates[0] 
        best_score = 0
        
        for cand in candidates:
            score = fuzz.ratio(resolved_term, cand)
            if score > best_score:
                best_score = score
                best_match = cand
        
        # 4. Get Type from Redis
        types = self.redis.get_type(best_match)
        
        return best_match, types

# Wrapper functions to maintain compatibility with main.py
def get_best_entity(term, sentence_context=None):
    linker = HybridLinker()
    ent, types = linker.resolve_entity(term, sentence_context)
    return ent, types 

def extract_entity_spans(sentence):
    if not nlp: return []
    doc = nlp(sentence)
    spans = set([ent.text for ent in doc.ents])
    # Fallback: Merge consecutive PROPNs
    current = []
    for token in doc:
        if token.pos_ == "PROPN": current.append(token.text)
        else:
            if len(current) > 1: spans.add(" ".join(current))
            current = []
    if len(current) > 1: spans.add(" ".join(current))
    
    normalized = set([s.lstrip("the ").strip() for s in spans])
    return list(normalized)
