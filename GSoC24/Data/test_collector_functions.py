# Test file for the new collector.py functions

from GSoC24.Data.collector import (
    get_sentences_with_entities,
    get_predicates_between,
    get_entity_types,
    get_text_of_wiki_page
)

def test_sentences_with_entities():
    """Test the get_sentences_with_entities function"""
    print("=== Testing get_sentences_with_entities ===")
    
    # Get Wikipedia article text for Albert Einstein
    article_text = get_text_of_wiki_page("Albert Einstein")
    
    # Test with subject and object that should appear together
    subject = "Albert Einstein"
    object_entity = "Ulm"
    
    relevant_sentences = get_sentences_with_entities(article_text, subject, object_entity)
    
    print(f"Found {len(relevant_sentences)} sentences containing both '{subject}' and '{object_entity}':")
    for i, sentence in enumerate(relevant_sentences[:3], 1):  # Show first 3 sentences
        print(f"{i}. {sentence[:200]}...")
    
    print()

def test_predicates_between():
    """Test the get_predicates_between function"""
    print("=== Testing get_predicates_between ===")
    
    # Test with Albert Einstein and Ulm
    subject_uri = "http://dbpedia.org/resource/Albert_Einstein"
    object_uri = "http://dbpedia.org/resource/Ulm"
    
    predicates = get_predicates_between(subject_uri, object_uri)
    
    print(f"Predicates connecting {subject_uri} and {object_uri}:")
    if predicates:
        for predicate in predicates:
            print(f"- {predicate}")
    else:
        print("No direct predicates found (they might only be connected via wikiPageWikiLink)")
    
    print()

def test_entity_types():
    """Test the get_entity_types function"""
    print("=== Testing get_entity_types ===")
    
    # Test with Albert Einstein
    entity_uri = "http://dbpedia.org/resource/Albert_Einstein"
    
    types = get_entity_types(entity_uri)
    
    print(f"Types for {entity_uri}:")
    for entity_type in types:
        print(f"- {entity_type}")
    
    print()

def test_entity_types_place():
    """Test the get_entity_types function with a place"""
    print("=== Testing get_entity_types (Place) ===")
    
    # Test with Ulm (a place)
    entity_uri = "http://dbpedia.org/resource/Ulm"
    
    types = get_entity_types(entity_uri)
    
    print(f"Types for {entity_uri}:")
    for entity_type in types:
        print(f"- {entity_type}")
    
    print()

if __name__ == "__main__":
    print("Testing new collector.py functions...\n")
    
    try:
        test_sentences_with_entities()
    except Exception as e:
        print(f"Error in test_sentences_with_entities: {e}\n")
    
    try:
        test_predicates_between()
    except Exception as e:
        print(f"Error in test_predicates_between: {e}\n")
    
    try:
        test_entity_types()
    except Exception as e:
        print(f"Error in test_entity_types: {e}\n")
    
    try:
        test_entity_types_place()
    except Exception as e:
        print(f"Error in test_entity_types_place: {e}\n")
    
    print("Testing completed!") 