#!/usr/bin/env python3
"""
Enhanced Neural Extraction Framework Pipeline

This pipeline combines:
1. Gemini-based triple extraction
2. Redis for fast entity lookup and resolution
3. Predicate scoring fusion for deterministic predicate selection
4. Final triple resolution with URIs

"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to Python path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent
except NameError:
    PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import NEF components
from GSoC24.Data.collector import get_text_of_wiki_page

# Entity-linking-master removed - using Redis-only approach


class PredicateScoringFusion:
    """Predicate scoring fusion system for deterministic predicate selection"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.predicate_descriptions = {}
        self.domain_range_mapping = {}
        self._fitted_vectorizer = None
        
        # Load default predicate data
        self._load_default_predicates()
    
    def _load_default_predicates(self):
        """Load default DBpedia predicate data"""
        default_predicates = [
            {
                'uri': 'http://dbpedia.org/ontology/birthPlace',
                'description': 'The place where a person was born',
                'domain': 'Person',
                'range': 'Place'
            },
            {
                'uri': 'http://dbpedia.org/ontology/employer',
                'description': 'The organization that employs a person',
                'domain': 'Person',
                'range': 'Organization'
            },
            {
                'uri': 'http://dbpedia.org/ontology/location',
                'description': 'The location of something',
                'domain': 'Thing',
                'range': 'Place'
            },
            {
                'uri': 'http://dbpedia.org/ontology/occupation',
                'description': 'The job or profession of a person',
                'domain': 'Person',
                'range': 'Thing'
            },
            {
                'uri': 'http://dbpedia.org/ontology/foundedBy',
                'description': 'The person who founded an organization',
                'domain': 'Organization',
                'range': 'Person'
            },
            {
                'uri': 'http://dbpedia.org/ontology/worksFor',
                'description': 'The organization a person works for',
                'domain': 'Person',
                'range': 'Organization'
            },
            {
                'uri': 'http://dbpedia.org/ontology/livesIn',
                'description': 'The place where a person lives',
                'domain': 'Person',
                'range': 'Place'
            }
        ]
        self.load_predicate_data(default_predicates)
    
    def load_predicate_data(self, predicate_data: List[Dict]):
        """Load predicate descriptions and domain/range mappings"""
        for pred in predicate_data:
            uri = pred['uri']
            self.predicate_descriptions[uri] = pred.get('description', '')
            self.domain_range_mapping[uri] = {
                'domain': pred.get('domain', ''),
                'range': pred.get('range', '')
            }
    
    def compute_type_compatibility(self, subject_uri: str, object_uri: str, predicate_uri: str) -> float:
        """Compute type compatibility score (1/0) based on domain/range constraints"""
        if predicate_uri not in self.domain_range_mapping:
            return 0.0
            
        domain = self.domain_range_mapping[predicate_uri]['domain']
        range_type = self.domain_range_mapping[predicate_uri]['range']
        
        # Get entity types
        subject_type = self._get_entity_type(subject_uri)
        object_type = self._get_entity_type(object_uri)
        
        # Check if subject matches domain and object matches range
        domain_match = self._type_matches(subject_type, domain) if domain else True
        range_match = self._type_matches(object_type, range_type) if range_type else True
        
        return 1.0 if (domain_match and range_match) else 0.0
    
    def _get_entity_type(self, uri: str) -> str:
        """Get entity type from URI (improved)"""
        entity_name = uri.split('/')[-1].lower()
        
        # Better type detection based on entity names
        if any(word in entity_name for word in ['einstein', 'jobs', 'person', 'human', 'scientist', 'inventor']):
            return 'Person'
        elif any(word in entity_name for word in ['ulm', 'germany', 'place', 'city', 'country', 'location']):
            return 'Place'
        elif any(word in entity_name for word in ['apple', 'inc', 'company', 'organization', 'corporation']):
            return 'Organization'
        else:
            return 'Thing'
    
    def _type_matches(self, entity_type: str, constraint_type: str) -> bool:
        """Check if entity type matches constraint type"""
        if not constraint_type:
            return True
        return entity_type.lower() in constraint_type.lower() or constraint_type.lower() in entity_type.lower()
    
    def compute_predicate_similarity(self, evidence_text: str, predicate_uri: str) -> float:
        """Compute max cosine similarity between evidence and predicate description"""
        if predicate_uri not in self.predicate_descriptions:
            return 0.0
            
        predicate_desc = self.predicate_descriptions[predicate_uri]
        
        # Use pre-fitted vectorizer to avoid refitting
        if self._fitted_vectorizer is None:
            all_texts = [evidence_text, predicate_desc] + list(self.predicate_descriptions.values())
            self._fitted_vectorizer = TfidfVectorizer(stop_words='english').fit(all_texts)
        
        # Transform texts
        texts = [evidence_text, predicate_desc]
        tfidf_matrix = self._fitted_vectorizer.transform(texts)
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    
    def compute_bm25_overlap(self, evidence_text: str, predicate_desc: str) -> float:
        """Compute BM25-style overlap score"""
        evidence_words = set(evidence_text.lower().split())
        predicate_words = set(predicate_desc.lower().split())
        
        if not predicate_words:
            return 0.0
            
        overlap = len(evidence_words.intersection(predicate_words))
        return overlap / len(predicate_words)
    
    def compute_desc_similarity(self, evidence_text: str, subject_uri: str, object_uri: str) -> float:
        """Compute cosine similarity between evidence and entity descriptions"""
        # Simplified - in practice you'd fetch actual abstracts from DBpedia
        subject_name = subject_uri.split('/')[-1].replace('_', ' ')
        object_name = object_uri.split('/')[-1].replace('_', ' ')
        
        combined_desc = f"{subject_name} {object_name}"
        
        if self._fitted_vectorizer is None:
            all_texts = [evidence_text, combined_desc] + list(self.predicate_descriptions.values())
            self._fitted_vectorizer = TfidfVectorizer(stop_words='english').fit(all_texts)
        
        texts = [evidence_text, combined_desc]
        tfidf_matrix = self._fitted_vectorizer.transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def select_best_predicate(self, evidence_text: str, subject_uri: str, object_uri: str, 
                            candidate_predicates: List[str], weights: Dict[str, float] = None) -> Tuple[str, Dict[str, float]]:
        """Select the best predicate using scoring fusion"""
        if weights is None:
            weights = {
                'type_compat': 0.4,  # Higher weight for type compatibility
                'pred_sim': 0.3,
                'bm25': 0.2,
                'desc_sim': 0.1
            }
        
        best_predicate = None
        best_score = -1
        best_scores = {}
        
        for predicate_uri in candidate_predicates:
            # Compute individual scores
            type_compat = self.compute_type_compatibility(subject_uri, object_uri, predicate_uri)
            pred_sim = self.compute_predicate_similarity(evidence_text, predicate_uri)
            bm25 = self.compute_bm25_overlap(evidence_text, self.predicate_descriptions.get(predicate_uri, ''))
            desc_sim = self.compute_desc_similarity(evidence_text, subject_uri, object_uri)
            
            # Compute weighted combination
            combined_score = (
                weights['type_compat'] * type_compat +
                weights['pred_sim'] * pred_sim +
                weights['bm25'] * bm25 +
                weights['desc_sim'] * desc_sim
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_predicate = predicate_uri
                best_scores = {
                    'combined_score': combined_score,
                    'type_compat': type_compat,
                    'pred_sim': pred_sim,
                    'bm25': bm25,
                    'desc_sim': desc_sim
                }
        
        return best_predicate, best_scores


class RedisEntityLinking:
    """Redis-based entity linking"""
    
    def __init__(self, host='91.99.92.217', port=6379, password='NEF!gsoc2025'):
        try:
            import redis
            self.redis_forms = redis.Redis(host=host, port=port, password=password, db=0, decode_responses=True)
            self.redis_redir = redis.Redis(host=host, port=port, password=password, db=1, decode_responses=True)
            
            # Test connection
            if self.redis_forms.ping() and self.redis_redir.ping():
                print("✓ Connected to Redis server successfully!")
                self.available = True
            else:
                print("✗ Could not connect to Redis")
                self.available = False
        except Exception as e:
            print(f"✗ Redis connection error: {e}")
            self.available = False
    
    def calculate_redirect(self, source):
        """Calculate redirects recursively"""
        result = self.redis_redir.get(source)
        if result is None:
            return source if isinstance(source, str) else source.decode('utf-8')
        return self.calculate_redirect(result)
    
    def query(self, surface_form):
        """Query surface forms from Redis"""
        if not self.available:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
            
        raw = self.redis_forms.hgetall(surface_form)
        if len(raw) == 0:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
        
        out = []
        for label, score in raw.items():
            out.append({'entity': label, 'support': int(score)})
        df_all = pd.DataFrame(out)
        df_all['score'] = df_all['support'] / df_all['support'].max()
        
        return df_all.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    def lookup(self, term, top_k=5, thr=0.01):
        """Lookup entity with redirects"""
        if not self.available:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
            
        df_temp = self.query(term)
        if len(df_temp) == 0:
            return pd.DataFrame(columns=['entity', 'support', 'score'])
        
        df_temp['entity'] = df_temp['entity'].apply(lambda x: self.calculate_redirect(x))
        df_final = df_temp.groupby('entity').sum()[['support']]
        df_final['score'] = df_final['support'] / df_final['support'].max()
        
        return df_final[df_final['score'] >= thr].sort_values(by='score', ascending=False)[:top_k]


class PredicateEmbeddingRetriever:
    """Retrieve candidate predicates using Gemini embeddings"""
    
    def __init__(self, api_key: str, embeddings_path: str = "embeddings.npy", 
                 predicates_path: str = "predicates.csv"):
        self.api_key = api_key
        
        # Load precomputed embeddings
        self.embeddings = np.load(embeddings_path)
        self.predicates = pd.read_csv(predicates_path)["predicate"].tolist()
        
        # Normalize embeddings once for faster cosine similarity
        self.embeddings_norm = self.embeddings / np.linalg.norm(
            self.embeddings, axis=-1, keepdims=True
        )
        
        print(f"✓ Loaded {len(self.predicates)} predicate embeddings")
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a query text using Gemini API"""
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent",
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
            data=json.dumps({
                "model": "models/embedding-001",
                "content": {"parts": [{"text": query_text}]}
            })
        )
        query_vec = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
        # Normalize
        query_vec = query_vec / np.linalg.norm(query_vec)
        return query_vec
    
    def get_top_k_predicates(self, relation_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k most similar predicates for a relation text"""
        # Embed the query
        query_vec = self.embed_query(relation_text)
        
        # Cosine similarity (already normalized)
        similarities = np.dot(self.embeddings_norm, query_vec)
        
        # Get top-k
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.predicates[idx], float(similarities[idx])))
        
        return results


class LLMDisambiguator:
    """LLM-based final disambiguation of candidates"""
    
    def __init__(self, gemini_model):
        self.model = gemini_model
    
    def disambiguate_triple(self, 
                           context: str,
                           subject_candidates: List[Tuple[str, float]],
                           predicate_candidates: List[Tuple[str, float]],
                           object_candidates: List[Tuple[str, float]]) -> Tuple[str, str, str]:
        """Use LLM to select the best triple from candidates"""
        
        # Format candidates nicely
        subj_text = "\n".join([f"  - {uri} (score: {score:.3f})" 
                               for uri, score in subject_candidates])
        pred_text = "\n".join([f"  - {uri} (similarity: {score:.3f})" 
                               for uri, score in predicate_candidates])
        obj_text = "\n".join([f"  - {uri} (score: {score:.3f})" 
                              for uri, score in object_candidates])
        
        prompt = f"""Given the following context and candidate URIs, select the most coherent and accurate RDF triple.

Context: "{context}"

Subject candidates:
{subj_text}

Predicate candidates:
{pred_text}

Object candidates:
{obj_text}

Instructions:
1. Consider the context meaning
2. Consider the candidate scores (higher = more confident)
3. Ensure semantic coherence between subject-predicate-object
4. Return ONLY the selected triple in JSON format

Return format:
{{"subject": "full_uri", "predicate": "full_uri", "object": "full_uri", "reasoning": "brief explanation"}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            return (result["subject"], result["predicate"], result["object"])
            
        except Exception as e:
            print(f"✗ LLM disambiguation error: {e}")
            # Fallback: take top candidates
            return (subject_candidates[0][0], 
                   predicate_candidates[0][0], 
                   object_candidates[0][0])


class EnhancedNEFPipeline:
    """Enhanced pipeline with embeddings + LLM disambiguation"""
    
    def __init__(self, gemini_api_key: str, embeddings_path: str = "embeddings.npy"):
        self.gemini_api_key = gemini_api_key
        
        # Initialize components
        self.redis_el = RedisEntityLinking()
        self.predicate_retriever = PredicateEmbeddingRetriever(
            gemini_api_key, embeddings_path
        )
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        self.llm_disambiguator = LLMDisambiguator(self.gemini_model)
        
        print("✓ Enhanced NEF Pipeline with embeddings initialized!")
    
    def resolve_entity_candidates(self, entity_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k entity candidates from Redis"""
        candidates = self.redis_el.lookup(entity_text, top_k=top_k)
        
        if candidates.empty:
            # Fallback
            fallback_uri = f"http://dbpedia.org/resource/{entity_text.replace(' ', '_')}"
            return [(fallback_uri, 0.5)]
        
        results = []
        for entity in candidates.index:
            score = candidates.loc[entity, 'score']
            # Format URI
            if entity.startswith('http://'):
                uri = entity
            else:
                uri = f"http://dbpedia.org/resource/{entity}"
            results.append((uri, score))
        
        return results
    
    def run_pipeline(self, sentence: str) -> List[Tuple[str, str, str]]:
        """Run pipeline on a single sentence"""
        print(f"\n📝 Processing: '{sentence}'")
        
        # Step 1: Extract raw triple with LLM
        raw_triples = self.extract_triples_with_gemini(sentence)
        if not raw_triples:
            return []
        
        resolved_triples = []
        
        for triple in raw_triples:
            print(f"\n🔍 Triple: {triple['subject']} - {triple['predicate']} - {triple['object']}")
            
            # Step 2: Get candidates
            print("   📍 Getting entity candidates from Redis...")
            subject_candidates = self.resolve_entity_candidates(triple['subject'], top_k=5)
            object_candidates = self.resolve_entity_candidates(triple['object'], top_k=5)
            
            print("   🔗 Getting predicate candidates from embeddings...")
            predicate_candidates = self.predicate_retriever.get_top_k_predicates(
                triple['predicate'], top_k=5
            )
            
            # Step 3: LLM disambiguation
            print("   🧠 LLM disambiguation...")
            final_triple = self.llm_disambiguator.disambiguate_triple(
                sentence,
                subject_candidates,
                predicate_candidates,
                object_candidates
            )
            
            resolved_triples.append(final_triple)
            print(f"   ✅ Final: {final_triple}")
        
        return resolved_triples

if __name__ == "__main__":
    exit(main())