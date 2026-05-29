import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "NeuralExtractionBot/1.0 (student_research_project)"
}

def resolve_redirect(term):
    if not term or len(term.split()) > 5 or term.isdigit():
        return term

    # STRATEGY 1: STRICT REDIRECT + DISAMBIGUATION CHECK

    try:
        r = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "titles": term,
                "redirects": 1,
                "prop": "pageprops", # <--- ASK FOR PAGE PROPERTIES
                "format": "json"
            },
            headers=HEADERS, timeout=5
        )
        data = r.json()
        
        # 1. Did we get a redirect? (e.g., UK -> United Kingdom)
        query = data.get("query", {})
        redirects = query.get("redirects", [])
        final_title = redirects[0]["to"] if redirects else term
        
        # 2. Is the final page a DISAMBIGUATION page?
        pages = query.get("pages", {})
        for page_id in pages:
            if page_id == "-1": continue # Page doesn't exist
            
            page = pages[page_id]
            # Check if it has the 'disambiguation' property
            is_disambig = "pageprops" in page and "disambiguation" in page["pageprops"]
            
            if not is_disambig:
                # SUCCESS! It's a real page (e.g., "Physics", "United Kingdom").
                # Return immediately. Do not go to Opensearch.
                return page["title"]
            else:
                # It exists, but it's a Disambiguation (e.g., "Barca").
                # IGNORE it. Fall through to Strategy 2 to find a better suggestion.
                pass
                
    except Exception as e:
        pass


    # STRATEGY 2: OPENSEARCH (Fallback for Slang/Ambiguity)

    try:
        r = requests.get(
            WIKIPEDIA_API,
            params={"action": "opensearch", "search": term, "limit": 5, "namespace": 0, "format": "json"},
            headers=HEADERS, timeout=5
        )
        data = r.json()
        titles = data[1]
        descriptions = data[2]
        
        for title, desc in zip(titles, descriptions):
            # Skip explicit disambiguations here too
            if "disambiguation" in desc.lower(): continue
            
            
            if title.lower() == term.lower() and not desc.strip():
                continue
                
            return title
    except:
        pass

    return term