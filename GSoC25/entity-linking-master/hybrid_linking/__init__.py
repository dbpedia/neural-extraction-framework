"""
Hybrid LLM + Symbolic Entity Linking System

A flexible and extensible entity linking system that combines LLM-based 
entity normalization and context analysis with knowledge base queries.
"""

from .generalized_linker import GeneralizedEntityLinker, LinkingResult
from .knowledge_base import KnowledgeBase, KnowledgeBaseRegistry, EntityCandidate, DBpediaKnowledgeBase
from .llm_provider import LLMProvider, LLMRegistry, GeminiProvider
from .linker import link_entity_to_dbpedia

# Convenience function for quick usage
def create_default_linker():
    """Create a default entity linker with Gemini and DBpedia."""
    
    gemini = GeminiProvider()
    dbpedia = DBpediaKnowledgeBase()
    
    return GeneralizedEntityLinker(
        llm_provider=gemini,
        knowledge_bases=[dbpedia]
    )

__version__ = "1.0.0"
__all__ = [
    "GeneralizedEntityLinker",
    "LinkingResult", 
    "KnowledgeBase",
    "KnowledgeBaseRegistry",
    "EntityCandidate",
    "DBpediaKnowledgeBase",
    "LLMProvider",
    "LLMRegistry",
    "GeminiProvider",
    "link_entity_to_dbpedia",
    "create_default_linker"
] 