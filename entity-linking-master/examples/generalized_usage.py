"""
Examples of using the generalized entity linking system.
"""

from hybrid_linking import (
    GeneralizedEntityLinker, 
    GeminiProvider, 
    DBpediaKnowledgeBase,
    create_default_linker
)

def example_basic_usage():
    """Basic usage with default configuration."""
    print("=== Basic Usage Example ===")
    
    # Create a default linker (Gemini + DBpedia)
    linker = create_default_linker()
    
    # Link an entity
    result = linker.link_entity("Donald Trump")
    
    print(f"Entity: {result.entity_mention}")
    print(f"Canonical Name: {result.canonical_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print("Top Candidates:")
    for candidate in result.candidates[:3]:
        print(f"  {candidate.label}: {candidate.uri} (score: {candidate.score:.2f})")
    print()

def example_context_aware():
    """Context-aware disambiguation example."""
    print("=== Context-Aware Example ===")
    
    linker = create_default_linker()
    
    # Company context
    result1 = linker.link_entity("apple", context="I work at apple")
    print(f"Company context: {result1.canonical_name}")
    if result1.context_analysis:
        print(f"Entity Type: {result1.context_analysis.get('entity_type')}")
    
    # Fruit context
    result2 = linker.link_entity("apple", context="I eat an apple every day")
    print(f"Fruit context: {result2.canonical_name}")
    if result2.context_analysis:
        print(f"Entity Type: {result2.context_analysis.get('entity_type')}")
    print()

def example_custom_configuration():
    """Example with custom configuration."""
    print("=== Custom Configuration Example ===")
    
    # Create custom providers
    gemini = GeminiProvider()
    dbpedia = DBpediaKnowledgeBase()
    
    # Create linker with custom configuration
    linker = GeneralizedEntityLinker(
        llm_provider=gemini,
        knowledge_bases=[dbpedia]
    )
    
    # Add additional LLM providers (if available)
    # linker.add_llm_provider("openai", OpenAIProvider())
    
    # Link with specific settings
    result = linker.link_entity(
        entity_mention="Washington",
        context="Washington was the first president",
        knowledge_bases=["DBpedia"],
        llm_provider="Gemini",
        limit=3
    )
    
    print(f"Entity: {result.entity_mention}")
    print(f"Canonical Name: {result.canonical_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"LLM Provider: {result.metadata['llm_provider']}")
    print(f"Knowledge Bases: {result.metadata['knowledge_bases_searched']}")
    print()

def example_batch_processing():
    """Example of batch processing multiple entities."""
    print("=== Batch Processing Example ===")
    
    linker = create_default_linker()
    
    # Define entities to link
    entities = [
        {"mention": "Apple", "context": "I work at Apple"},
        {"mention": "Microsoft", "context": "Microsoft released a new product"},
        {"mention": "Paris", "context": "I visited Paris last summer"}
    ]
    
    # Process all entities
    results = linker.batch_link(entities)
    
    for result in results:
        print(f"{result.entity_mention} -> {result.canonical_name} (confidence: {result.confidence:.2f})")
        if result.candidates:
            print(f"  Top result: {result.candidates[0].uri}")
    print()

def example_extensibility():
    """Example showing how to extend the system."""
    print("=== Extensibility Example ===")
    
    # This shows how you could add new knowledge bases or LLM providers
    print("Available LLM Providers:")
    linker = create_default_linker()
    print(f"  {linker.llm_registry.list_available()}")
    
    print("Available Knowledge Bases:")
    print(f"  {linker.kb_registry.list_available()}")
    
    print("\nTo add new providers:")
    print("1. Implement KnowledgeBase or LLMProvider interface")
    print("2. Register with the appropriate registry")
    print("3. Use in your linking operations")
    print()

if __name__ == "__main__":
    example_basic_usage()
    example_context_aware()
    example_custom_configuration()
    example_batch_processing()
    example_extensibility() 