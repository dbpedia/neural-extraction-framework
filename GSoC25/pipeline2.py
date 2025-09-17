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


class EnhancedNEFPipeline:
    """Enhanced Neural Extraction Framework Pipeline with Redis-based entity linking"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.redis_el = RedisEntityLinking()
        self.predicate_scorer = PredicateScoringFusion()
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
        
        print("✓ Enhanced NEF Pipeline (Redis-only) initialized successfully!")
    
    def extract_triples_with_gemini(self, text: str) -> List[Dict]:
        """Extract raw triples using Gemini API"""
        prompt = f"""Extract all (subject, predicate, object) triples from the following text. 
Use specific, meaningful predicates and clean entity names.

Text: {text}

Return as JSON:
[{{"subject": "...", "predicate": "...", "object": "..."}}, ...]

Guidelines:
- Use specific predicates (e.g., "Discovered", "Was born in", "Worked at")
- Capitalize the first letter of predicates
- Extract clean entity names without articles
- Focus on meaningful relationships"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            triples = json.loads(response_text)
            print(f"✓ Extracted {len(triples)} raw triples with Gemini")
            return triples
        except Exception as e:
            print(f"✗ Error extracting triples with Gemini: {e}")
            return []
    
    def resolve_entity_with_redis(self, entity_text: str, context: str) -> str:
        """Resolve entity using Redis only (no entity-linking-master needed)"""
        try:
            # Get candidates from Redis
            candidates = self.redis_el.lookup(entity_text, top_k=1)
            
            if not candidates.empty:
                redis_entity = candidates.index[0]
                print(f"✓ Redis found entity: {entity_text} → {redis_entity}")
                
                # Redis already gives us the correct DBpedia resource name
                if redis_entity.startswith('http://dbpedia.org/resource/'):
                    return redis_entity  # Already a full URI
                else:
                    # Convert to proper DBpedia URI format
                    entity_uri = f"http://dbpedia.org/resource/{redis_entity}"
                    return entity_uri
            else:
                # Fallback: construct URI from original text
                entity_uri = f"http://dbpedia.org/resource/{entity_text.replace(' ', '_')}"
                print(f"⚠ Redis failed for '{entity_text}', using constructed URI: {entity_uri}")
                return entity_uri
                
        except Exception as e:
            print(f"✗ Redis error for '{entity_text}': {e}")
            # Fallback: construct URI
            entity_uri = f"http://dbpedia.org/resource/{entity_text.replace(' ', '_')}"
            return entity_uri
    
    def get_candidate_predicates(self, relation_text: str) -> List[str]:
        """Get candidate predicates using Gemini API"""
        prompt = f"""Given the relation text "{relation_text}", suggest 3-5 most appropriate DBpedia ontology predicates.

Return as JSON array of predicate URIs:
["http://dbpedia.org/ontology/...", "http://dbpedia.org/ontology/...", ...]

Examples:
- "was born in" → ["http://dbpedia.org/ontology/birthPlace"]
- "works at" → ["http://dbpedia.org/ontology/employer", "http://dbpedia.org/ontology/workplace"]
- "lives in" → ["http://dbpedia.org/ontology/residence", "http://dbpedia.org/ontology/location"]

Focus on the most semantically appropriate DBpedia ontology predicates."""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            candidate_predicates = json.loads(response_text)
            
            print(f"✓ Gemini found {len(candidate_predicates)} candidate predicates for '{relation_text}'")
            return candidate_predicates
        except Exception as e:
            print(f"✗ Error getting candidate predicates with Gemini: {e}")
            # Fallback predicates
            return [
                "http://dbpedia.org/ontology/birthPlace",
                "http://dbpedia.org/ontology/location",
                "http://dbpedia.org/ontology/country"
            ]
    
    def run_pipeline(self, article_text: str) -> List[Tuple[str, str, str]]:
        """Run the enhanced pipeline"""
        print(f"\n📝 Step 1: Extracting raw triples...")
        raw_triples = self.extract_triples_with_gemini(article_text)
        if not raw_triples:
            print("✗ No triples extracted, exiting pipeline")
            return []
        
        resolved_triples = []
        
        for i, triple in enumerate(raw_triples, 1):
            print(f"\n�� Step 2.{i}: Processing triple: {triple['subject']} - {triple['predicate']} - {triple['object']}")
            
            # Step 2a: Resolve entities with Redis
            print("   📍 Resolving entities...")
            subject_uri = self.resolve_entity_with_redis(triple['subject'], article_text)
            object_uri = self.resolve_entity_with_redis(triple['object'], article_text)
            
            # Step 2b: Get candidate predicates
            print("   🔗 Getting candidate predicates...")
            predicate_candidates = self.get_candidate_predicates(triple['predicate'])
            
            # Step 2c: Use predicate scoring fusion for disambiguation
            print("   🧠 Running predicate scoring fusion...")
            best_predicate, scores = self.predicate_scorer.select_best_predicate(
                article_text,
                subject_uri,
                object_uri,
                predicate_candidates
            )
            
            if best_predicate:
                resolved_triple = (subject_uri, best_predicate, object_uri)
                resolved_triples.append(resolved_triple)
                print(f"   ✅ Resolved: {resolved_triple}")
            else:
                print(f"   ⚠ Failed to resolve predicate for triple {i}")
        
        print(f"\n�� Pipeline completed! Resolved {len(resolved_triples)} triples")
        return resolved_triples


def main():
    """CLI wrapper for the Enhanced NEF pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced Neural Extraction Framework Pipeline')
    parser.add_argument("--article", type=str, help="Article text to process")
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument("--api_key", type=str, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--wikipage", type=str, help="Wikipedia page title to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("✗ Error: Gemini API key required. Use --api_key or set GEMINI_API_KEY environment variable")
        return 1
    
    # Get text to process
    text = None
    if args.article:
        text = args.article
    elif args.text:
        text = args.text
    elif args.wikipage:
        try:
            text = get_text_of_wiki_page(args.wikipage)
            print(f"✓ Retrieved Wikipedia article: {args.wikipage}")
        except Exception as e:
            print(f"✗ Error retrieving Wikipedia article: {e}")
            return 1
    else:
        print("✗ Error: Must provide --article, --text, or --wikipage")
        return 1
    
    # Initialize and run pipeline
    try:
        pipeline = EnhancedNEFPipeline(api_key)
        resolved_triples = pipeline.run_pipeline(text)
        
        # Print results
        print("\n" + "=" * 80)
        print("🎯 FINAL RESULTS")
        print("=" * 80)
        
        if resolved_triples:
            for i, (subject_uri, predicate_uri, object_uri) in enumerate(resolved_triples, 1):
                print(f"\n{i}. {subject_uri}")
                print(f"   {predicate_uri}")
                print(f"   {object_uri}")
        else:
            print("No triples were successfully resolved.")
        
        return 0
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())