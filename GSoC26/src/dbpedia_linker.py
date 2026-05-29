import requests
import re
from rapidfuzz import fuzz
# IMPORT YOUR RESOLVER FROM THE OTHER FILE
from src.wiki_resolver import resolve_redirect

DBPEDIA_LOOKUP = "https://lookup.dbpedia.org/api/search"

# Weights Configuration
TYPE_PRIOR = {
    "Person": 1.8, "Athlete": 1.9, "Politician": 1.8, "Artist": 1.7,
    "Company": 1.9, "Organisation": 1.7, "SportsTeam": 2.0,
    "FootballClub": 2.1, "SoccerClub": 2.1, "Place": 1.7, "City": 1.7,
    "Country": 1.7, "Album": 1.5, "Film": 1.4, "Work": 1.2,
    "Event": 0.3, "Season": 0.05, "Tournament": 0.2
}
STOPWORDS = {"fc", "club", "sc", "the", "of", "and", "in", "inc"}

def clean_label(label):
    label = re.sub('<[^<]+?>', '', label)
    label = re.sub(r'\s*\(.*?\)', '', label)
    return label.strip()

def covers_query(query, label):
    q_tokens = [t for t in query.lower().replace("-", " ").split() if t not in STOPWORDS]
    l_tokens = label.lower().replace("-", " ").split()
    for q in q_tokens:
        if not any(l.startswith(q) for l in l_tokens):
            return False
    return True

def get_best_entity(query):
    # 1. Handle Literal Years
    if query.isdigit() and len(query) == 4:
        return query, ["xsd:gYear"]

    # 2. Resolve Slang (Using the imported function!)
    clean_query = resolve_redirect(query)

    # 3. Candidate Generation
    candidates = [clean_query]
    if "fc" not in clean_query.lower():
        candidates.append(clean_query + " FC")

    all_matches = []
    for term in candidates:
        try:
            r = requests.get(DBPEDIA_LOOKUP, params={"query": term, "format": "json", "MaxHits": 20}).json()
            for doc in r.get("docs", []):
                label = clean_label(doc.get("label", [""])[0])
                if label.startswith("List of"): continue

                # Scoring Math
                lex = fuzz.token_set_ratio(clean_query.lower(), label.lower()) / 100.0
                if lex < 0.65 or not covers_query(clean_query, label): continue

                types = doc.get("typeName", [])
                refCount = int(doc.get("refCount", [0])[0])
                
                type_score = max([TYPE_PRIOR.get(t, 1.0) for t in types], default=1.0)
                popularity = 1 + min(refCount / 8000, 1.0)
                score = lex * type_score * popularity

                # Penalties
                if "season" in label.lower() or "season" in types: score *= 0.1
                if clean_query.lower() == label.lower(): score *= 1.5

                all_matches.append({"label": label, "types": types, "score": score})
        except:
            continue

    if not all_matches: return "Unknown", []
    best = max(all_matches, key=lambda x: x["score"])
    return best["label"], best["types"]