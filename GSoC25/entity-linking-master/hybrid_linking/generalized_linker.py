from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from .knowledge_base import KnowledgeBase, KnowledgeBaseRegistry, EntityCandidate
from .llm_provider import LLMProvider, LLMRegistry

@dataclass
class LinkingResult:
    """Result of entity linking operation."""
    entity_mention: str
    canonical_name: str
    context_analysis: Optional[Dict[str, Any]] = None
    candidates: List[EntityCandidate] = None
    knowledge_base: str = "unknown"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

class GeneralizedEntityLinker:
    """
    A generalized entity linker that can work with multiple knowledge bases and LLM providers.
    """
    
    def __init__(self, 
                 llm_provider: Optional[LLMProvider] = None,
                 knowledge_bases: Optional[List[KnowledgeBase]] = None):
        
        # Initialize LLM registry
        self.llm_registry = LLMRegistry()
        if llm_provider:
            self.llm_registry.register(llm_provider.get_name(), llm_provider)
        
        # Initialize knowledge base registry
        self.kb_registry = KnowledgeBaseRegistry()
        if knowledge_bases:
            for kb in knowledge_bases:
                self.kb_registry.register(kb.get_name(), kb)
    
    def add_llm_provider(self, name: str, provider: LLMProvider):
        """Add an LLM provider to the registry."""
        self.llm_registry.register(name, provider)
    
    def add_knowledge_base(self, name: str, kb: KnowledgeBase):
        """Add a knowledge base to the registry."""
        self.kb_registry.register(name, kb)
    
    def link_entity(self, 
                   entity_mention: str, 
                   context: Optional[str] = None,
                   knowledge_bases: Optional[List[str]] = None,
                   llm_provider: Optional[str] = None,
                   limit: int = 5) -> LinkingResult:
        """
        Link an entity mention to knowledge base URIs.
        
        Args:
            entity_mention: The entity to link
            context: Optional context text
            knowledge_bases: List of knowledge base names to search (None = all)
            llm_provider: Name of LLM provider to use (None = first available)
            limit: Maximum number of candidates per knowledge base
        """
        
        # Get LLM provider
        if llm_provider:
            provider = self.llm_registry.get(llm_provider)
        else:
            available = self.llm_registry.list_available()
            if not available:
                raise ValueError("No LLM providers available")
            provider = self.llm_registry.get(available[0])
        
        # Step 1: Normalize entity name
        canonical_name = self._normalize_entity_name(entity_mention, context, provider)
        
        # Step 2: Analyze context if provided
        context_analysis = None
        if context:
            context_analysis = self._analyze_entity_context(entity_mention, context, provider)
        
        # Step 3: Search knowledge bases
        if knowledge_bases:
            # Search specific knowledge bases
            all_candidates = []
            for kb_name in knowledge_bases:
                kb = self.kb_registry.get(kb_name)
                if kb:
                    candidates = kb.search_entities(canonical_name, context_analysis, limit)
                    all_candidates.extend(candidates)
        else:
            # Search all knowledge bases
            results = self.kb_registry.search_all(canonical_name, context_analysis, limit)
            all_candidates = []
            for kb_name, candidates in results.items():
                all_candidates.extend(candidates)
        
        # Step 4: Rank and select best candidates
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        top_candidates = all_candidates[:limit]
        
        # Step 5: Calculate overall confidence
        confidence = self._calculate_overall_confidence(top_candidates, context_analysis)
        
        return LinkingResult(
            entity_mention=entity_mention,
            canonical_name=canonical_name,
            context_analysis=context_analysis,
            candidates=top_candidates,
            confidence=confidence,
            metadata={
                "llm_provider": provider.get_name(),
                "knowledge_bases_searched": knowledge_bases or self.kb_registry.list_available()
            }
        )
    
    def _normalize_entity_name(self, entity_mention: str, context: Optional[str], provider: LLMProvider) -> str:
        """Normalize entity name using the specified LLM provider."""
        prompt = f"""
Given the following entity mention, return the canonical name as used in knowledge bases (just the name, no explanation):
Entity: {entity_mention}
"""
        if context:
            prompt += f"\nContext: {context}"
        
        return provider.generate_text(prompt).strip()
    
    def _analyze_entity_context(self, entity_mention: str, context: str, provider: LLMProvider) -> Dict[str, Any]:
        """Analyze entity context using the specified LLM provider."""
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
            response = provider.generate_text(prompt).strip()
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
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
    
    def _calculate_overall_confidence(self, candidates: List[EntityCandidate], context_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate overall confidence for the linking result."""
        if not candidates:
            return 0.0
        
        # Average of top candidate scores
        avg_score = sum(c.score for c in candidates) / len(candidates)
        
        # Boost if context analysis is confident
        if context_analysis:
            context_confidence = context_analysis.get("confidence", 0.5)
            return (avg_score + context_confidence) / 2
        
        return avg_score
    
    def batch_link(self, entities: List[Dict[str, Any]]) -> List[LinkingResult]:
        """Link multiple entities in batch."""
        results = []
        for entity_data in entities:
            result = self.link_entity(
                entity_mention=entity_data["mention"],
                context=entity_data.get("context"),
                knowledge_bases=entity_data.get("knowledge_bases"),
                llm_provider=entity_data.get("llm_provider"),
                limit=entity_data.get("limit", 5)
            )
            results.append(result)
        return results 