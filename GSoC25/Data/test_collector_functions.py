# Test file for the new collector.py functions

import sys
from pathlib import Path
import argparse

# Add project root to Python path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent.parent  # Go up 3 levels to reach project root
except NameError:
    PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GSoC25.Data.collector import (
    get_sentences_with_entities,
    get_predicates_between,
    get_entity_types,
    get_text_of_wiki_page
)

def test_sentences_with_entities(article_name, subject, object_entity):
    """Test the get_sentences_with_entities function"""
    print("=== Testing get_sentences_with_entities ===")
    
    # Get Wikipedia article text
    article_text = get_text_of_wiki_page(article_name)
    
    relevant_sentences = get_sentences_with_entities(article_text, subject, object_entity)
    
    print(f"Found {len(relevant_sentences)} sentences containing both '{subject}' and '{object_entity}':")
    for i, sentence in enumerate(relevant_sentences[:3], 1):  # Show first 3 sentences
        print(f"{i}. {sentence[:200]}...")
    
    print()

def test_predicates_between(subject_uri, object_uri):
    """Test the get_predicates_between function"""
    print("=== Testing get_predicates_between ===")
    
    predicates = get_predicates_between(subject_uri, object_uri)
    
    print(f"Predicates connecting {subject_uri} and {object_uri}:")
    if predicates:
        for predicate in predicates:
            print(f"- {predicate}")
    else:
        print("No direct predicates found (they might only be connected via wikiPageWikiLink)")
    
    print()

def test_entity_types(entity_uri, entity_name=""):
    """Test the get_entity_types function"""
    print(f"=== Testing get_entity_types ({entity_name}) ===")
    
    types = get_entity_types(entity_uri)
    
    print(f"Types for {entity_uri}:")
    for entity_type in types:
        print(f"- {entity_type}")
    
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test new collector.py functions")
    parser.add_argument("--article", type=str, default="Albert Einstein", 
                       help="Wikipedia article name to test with")
    parser.add_argument("--subject", type=str, default="Albert Einstein", 
                       help="Subject entity name to search for")
    parser.add_argument("--object", type=str, default="Ulm", 
                       help="Object entity name to search for")
    parser.add_argument("--subject_uri", type=str, 
                       default="http://dbpedia.org/resource/Albert_Einstein",
                       help="Subject entity URI for predicate/type tests")
    parser.add_argument("--object_uri", type=str, 
                       default="http://dbpedia.org/resource/Ulm",
                       help="Object entity URI for predicate/type tests")
    
    args = parser.parse_args()
    
    print(f"Testing new collector.py functions with:")
    print(f"Article: {args.article}")
    print(f"Subject: {args.subject} ({args.subject_uri})")
    print(f"Object: {args.object} ({args.object_uri})")
    print()
    
    try:
        test_sentences_with_entities(args.article, args.subject, args.object)
    except Exception as e:
        print(f"Error in test_sentences_with_entities: {e}\n")
    
    try:
        test_predicates_between(args.subject_uri, args.object_uri)
    except Exception as e:
        print(f"Error in test_predicates_between: {e}\n")
    
    try:
        test_entity_types(args.subject_uri, "Subject")
    except Exception as e:
        print(f"Error in test_entity_types (Subject): {e}\n")
    
    try:
        test_entity_types(args.object_uri, "Object")
    except Exception as e:
        print(f"Error in test_entity_types (Object): {e}\n")
    
    print("Testing completed!") 