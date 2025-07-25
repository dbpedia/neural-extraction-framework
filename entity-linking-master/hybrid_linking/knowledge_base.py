from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EntityCandidate:
    """Represents a candidate entity from a knowledge base."""
    uri: str
    label: str
    score: float
    entity_type: Optional[str] = None
    description: Optional[str] = None
    confidence: float = 0.0

class KnowledgeBase(ABC):
    """Abstract interface for knowledge bases."""
    
    @abstractmethod
    def search_entities(self, label: str, context: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[EntityCandidate]:
        """Search for entities by label with optional context."""
        pass
    
    @abstractmethod
    def get_entity_info(self, uri: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about an entity."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this knowledge base."""
        pass

class DBpediaKnowledgeBase(KnowledgeBase):
    """DBpedia implementation of the knowledge base interface."""
    
    def __init__(self, endpoint: str = "https://dbpedia.org/sparql"):
        self.endpoint = endpoint
    
    def search_entities(self, label: str, context: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[EntityCandidate]:
        from .dbpedia_sparql import search_dbpedia_entity
        
        # Use existing DBpedia search logic
        candidates = search_dbpedia_entity(label, limit)
        
        # Convert to EntityCandidate objects
        entity_candidates = []
        for uri, label_text in candidates:
            score = 0.5  # Base score
            if context:
                # Apply context-aware scoring
                score = self._calculate_context_score(uri, context)
            
            entity_candidates.append(EntityCandidate(
                uri=uri,
                label=label_text,
                score=score
            ))
        
        # Sort by score
        entity_candidates.sort(key=lambda x: x.score, reverse=True)
        return entity_candidates
    
    def get_entity_info(self, uri: str) -> Optional[Dict[str, Any]]:
        # Implementation for getting detailed entity info
        pass
    
    def get_name(self) -> str:
        return "DBpedia"
    
    def _calculate_context_score(self, uri: str, context: Dict[str, Any]) -> float:
        # Context scoring logic (simplified version)
        score = 0.5
        expected_type = context.get("entity_type", "other")
        
        if expected_type == "company" and "company" in uri.lower():
            score += 0.3
        elif expected_type == "person" and "person" in uri.lower():
            score += 0.3
        
        return score

class WikidataKnowledgeBase(KnowledgeBase):
    """Wikidata implementation (placeholder for future extension)."""
    
    def search_entities(self, label: str, context: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[EntityCandidate]:
        # Placeholder for Wikidata implementation
        return []
    
    def get_entity_info(self, uri: str) -> Optional[Dict[str, Any]]:
        pass
    
    def get_name(self) -> str:
        return "Wikidata"

class KnowledgeBaseRegistry:
    """Registry for managing multiple knowledge bases."""
    
    def __init__(self):
        self._knowledge_bases: Dict[str, KnowledgeBase] = {}
    
    def register(self, name: str, kb: KnowledgeBase):
        """Register a knowledge base."""
        self._knowledge_bases[name] = kb
    
    def get(self, name: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by name."""
        return self._knowledge_bases.get(name)
    
    def list_available(self) -> List[str]:
        """List all available knowledge bases."""
        return list(self._knowledge_bases.keys())
    
    def search_all(self, label: str, context: Optional[Dict[str, Any]] = None, limit: int = 10) -> Dict[str, List[EntityCandidate]]:
        """Search across all registered knowledge bases."""
        results = {}
        for name, kb in self._knowledge_bases.items():
            try:
                results[name] = kb.search_entities(label, context, limit)
            except Exception as e:
                print(f"Error searching {name}: {e}")
                results[name] = []
        return results 