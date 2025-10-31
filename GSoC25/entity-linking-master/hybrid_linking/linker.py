from hybrid_linking.gemini_api import call_gemini
from hybrid_linking.dbpedia_sparql import search_dbpedia_entity
from typing import Optional, List, Tuple

def normalize_entity_name(entity_mention: str, context: Optional[str] = None) -> str:
    prompt = f"""
Given the following entity mention, return the canonical name as used in DBpedia (just the name, no explanation):
Entity: {entity_mention}
"""
    if context:
        prompt += f"\nContext: {context}"
    return call_gemini(prompt).strip()

def analyze_entity_context(entity_mention: str, context: str) -> dict:
    """
    Use Gemini to analyze the context and determine entity type and characteristics.
    """
    prompt = f"""
Analyze the following entity mention and context to determine the most likely entity type and characteristics.
Return your analysis as a JSON object with the following fields:
- entity_type: "person", "company", "place", "product", "concept", or "other"
- confidence: confidence score (0-1)
- keywords: list of relevant keywords that might help in disambiguation
- description: brief description of what this entity likely refers to

Entity: {entity_mention}
Context: {context}

Return only the JSON object, no additional text.
"""
    
    try:
        response = call_gemini(prompt).strip()
        # Try to extract JSON from the response
        import json
        import re
        
        # Find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            # Fallback if no JSON found
            return {
                "entity_type": "other",
                "confidence": 0.5,
                "keywords": [],
                "description": "Unknown entity type"
            }
    except Exception as e:
        print(f"Error analyzing context: {e}")
        return {
            "entity_type": "other",
            "confidence": 0.5,
            "keywords": [],
            "description": "Error in analysis"
        }

def search_dbpedia_with_context(label: str, context_analysis: dict, limit: int = 10) -> List[Tuple[str, str, float]]:
    """
    Search DBpedia with context-aware filtering and scoring.
    Returns list of (URI, label, score) tuples.
    """
    from SPARQLWrapper import SPARQLWrapper, JSON
    
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    
    # Build context-aware query with more entity information
    query = f'''
    SELECT DISTINCT ?uri ?label ?type ?abstract WHERE {{
      ?uri rdfs:label ?label .
      FILTER (?label = "{label}"@en)
      OPTIONAL {{
        ?uri rdf:type ?type .
      }}
      OPTIONAL {{
        ?uri dbo:abstract ?abstract .
        FILTER (lang(?abstract) = 'en')
      }}
    }} LIMIT {limit}
    '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        candidates = []
        seen_uris = set()
        
        for result in results["results"]["bindings"]:
            uri = result["uri"]["value"]
            
            # Skip duplicates
            if uri in seen_uris:
                continue
            seen_uris.add(uri)
            
            label = result["label"]["value"]
            entity_type_uri = result.get("type", {}).get("value", "")
            abstract = result.get("abstract", {}).get("value", "")
            
            # Score based on context analysis
            score = calculate_context_score(uri, entity_type_uri, abstract, context_analysis)
            candidates.append((uri, label, score))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates
        
    except Exception as e:
        print(f"Error in context-aware search: {e}")
        return []

def calculate_context_score(uri: str, entity_type_uri: str, abstract: str, context_analysis: dict) -> float:
    """
    Calculate a score for how well an entity matches the context.
    """
    score = 0.5  # Base score
    
    # Check entity type match
    expected_type = context_analysis.get("entity_type", "other")
    
    if expected_type == "company":
        if any(term in entity_type_uri.lower() for term in ["company", "corporation", "organisation", "organization"]):
            score += 0.4
        elif "product" in entity_type_uri.lower() or "brand" in entity_type_uri.lower():
            score += 0.2
    elif expected_type == "person":
        if any(term in entity_type_uri.lower() for term in ["person", "human", "agent"]):
            score += 0.4
    elif expected_type == "place":
        if any(term in entity_type_uri.lower() for term in ["place", "location", "city", "country", "region"]):
            score += 0.4
    elif expected_type == "product":
        if any(term in entity_type_uri.lower() for term in ["product", "good", "device", "software"]):
            score += 0.4
    
    # Check for keywords in URI and abstract
    keywords = context_analysis.get("keywords", [])
    for keyword in keywords:
        if keyword.lower() in uri.lower():
            score += 0.1
        if keyword.lower() in abstract.lower():
            score += 0.15
    
    # Check for work-related terms in abstract for company context
    if expected_type == "company" and abstract:
        work_terms = ["company", "corporation", "business", "technology", "software", "hardware", "employees"]
        if any(term in abstract.lower() for term in work_terms):
            score += 0.2
    
    # Boost score for high confidence analysis
    confidence = context_analysis.get("confidence", 0.5)
    score += confidence * 0.2
    
    return min(score, 1.0)  # Cap at 1.0

def link_entity_to_dbpedia(entity_mention: str, context: Optional[str] = None, limit: int = 5):
    # Step 1: Normalize entity name using Gemini
    canonical_name = normalize_entity_name(entity_mention, context)
    
    # Step 2: Analyze context if provided
    context_analysis = {}
    if context:
        context_analysis = analyze_entity_context(entity_mention, context)
        print(f"Context Analysis: {context_analysis}")
    
    # Step 3: Search DBpedia with context-aware filtering
    if context_analysis:
        candidates = search_dbpedia_with_context(canonical_name, context_analysis, limit=limit)
        # Convert to the expected format
        candidate_list = [(uri, label) for uri, label, score in candidates]
    else:
        # Fallback to simple search
        candidates = search_dbpedia_entity(canonical_name, limit=limit)
        candidate_list = candidates
    
    # Step 4: Return results
    return {
        "mention": entity_mention,
        "canonical_name": canonical_name,
        "context_analysis": context_analysis if context else None,
        "candidates": candidate_list
    }

if __name__ == "__main__":
    import sys
    entity = sys.argv[1] if len(sys.argv) > 1 else "apple"
    context = sys.argv[2] if len(sys.argv) > 2 else "I work at apple"
    
    print(f"Entity: {entity}")
    print(f"Context: {context}")
    print("-" * 50)
    
    result = link_entity_to_dbpedia(entity, context)
    print(f"Entity Mention: {result['mention']}")
    print(f"Canonical Name: {result['canonical_name']}")
    if result['context_analysis']:
        print(f"Entity Type: {result['context_analysis'].get('entity_type', 'unknown')}")
        print(f"Confidence: {result['context_analysis'].get('confidence', 0):.2f}")
    print("Candidates:")
    for uri, label in result['candidates']:
        print(f"  {label}: {uri}") 