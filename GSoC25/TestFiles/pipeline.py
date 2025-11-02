#!/usr/bin/env python3
"""
Gemini + Redis Neural Extraction Framework Pipeline

This pipeline combines:
1. Gemini-based triple extraction
2. Redis entity linking (with Gemini fallback)
3. Gemini-based context collection
4. Gemini-based predicate candidate generation
5. Gemini-based disambiguation for final triple selection

Usage:
    python pipeline.py --article "Albert Einstein was born in Ulm."
    python pipeline.py --text "Your text here" --api_key YOUR_GEMINI_API_KEY
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import google.generativeai as genai

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


class RedisEntityLinking:
    """Redis-based entity linking with fallback to LLM"""
    
    def __init__(self, host=None, port=None, password=None):
         self.host = host or os.environ.get('NEF_REDIS_HOST')
         self.port = port or os.environ.get('NEF_REDIS_PORT')
         self.password = password or os.environ.get('NEF_REDIS_PASSWORD')
         try:
            import redis
            self.redis_forms = redis.Redis(host=self.host, port=self.port, password=self.password, db=0, decode_responses=True)
            self.redis_redir = redis.Redis(host=self.host, port=self.port, password=self.password, db=1, decode_responses=True)
            try:
            # Test connection
                if self.redis_forms.ping() and self.redis_redir.ping():
                    print("✓ Connected to Redis server successfully!")
                    self.available = True
                else:
                    print("✗ Could not connect to Redis")
                    raise ConnectionError("Redis connection failed")
            except Exception as e:
                raise ConnectionError(f"Redis connection error: {e}")
    
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


class NEFPipeline:
    """Gemini + Redis Neural Extraction Framework Pipeline"""
    
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key
        self.redis_el = RedisEntityLinking()
        
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
        print("✓ NEF Pipeline initialized successfully!")
    
    
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
    
    def resolve_entity_with_redis(self, entity_text: str, top_k: int = 3) -> List[str]:
        """Resolve entity using Redis first, with fallback to LLM"""
        # Try Redis first
        if self.redis_el.available:
            redis_results = self.redis_el.lookup(entity_text, top_k=top_k, thr=0.01)
            if len(redis_results) > 0:
                candidate_uris = []
                for uri in redis_results.index:
                    # Ensure proper DBpedia URI format
                    if uri.startswith('http://dbpedia.org/resource/'):
                        candidate_uris.append(uri)
                    else:
                        # Convert to proper DBpedia URI format
                        formatted_uri = f"http://dbpedia.org/resource/{uri.replace(' ', '_')}"
                        candidate_uris.append(formatted_uri)
                print(f"✓ Redis found {len(candidate_uris)} candidates for '{entity_text}'")
                return candidate_uris
        
        # Fallback to LLM-based entity linking
        print(f"⚠ Redis failed for '{entity_text}', using LLM fallback")
        return self._resolve_entity_with_llm(entity_text)
    
    def _resolve_entity_with_llm(self, entity_text: str) -> List[str]:
        """Fallback entity resolution using LLM"""
        try:
            # Simple heuristic: convert to DBpedia resource URI format
            entity_uri = f"http://dbpedia.org/resource/{entity_text.replace(' ', '_')}"
            return [entity_uri]
        except Exception as e:
            print(f"✗ LLM entity resolution failed for '{entity_text}': {e}")
            return []
    
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
    
    def llm_disambiguate_triple(self, original_text: str, candidate_subject_uris: List[str], 
                               candidate_predicate_uris: List[str], candidate_object_uris: List[str]) -> Optional[Dict]:
        """Use LLM to disambiguate and select the best triple"""
        prompt = f'''
Given the following sentence:
"""
{original_text[:500]}
"""

And these candidate subject URIs:
{json.dumps(candidate_subject_uris, indent=2)}

And these candidate predicate URIs:
{json.dumps(candidate_predicate_uris, indent=2)}

And these candidate object URIs:
{json.dumps(candidate_object_uris, indent=2)}

Select the best subject, predicate, and object URIs that together form the most accurate RDF triple for the sentence.

CRITICAL REQUIREMENTS:
1. You MUST return the EXACT URIs from the candidate lists above
2. Do NOT modify, shorten, or change them in any way
3. All URIs must be complete with full prefixes:
   - Subject/Object URIs: http://dbpedia.org/resource/...
   - Predicate URIs: http://dbpedia.org/ontology/...
4. Copy the URIs exactly as they appear in the candidate lists

Respond in JSON as:
{{
  "subject_uri": "http://dbpedia.org/resource/...",
  "predicate_uri": "http://dbpedia.org/ontology/...",
  "object_uri": "http://dbpedia.org/resource/..."
}}
'''
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            triple = json.loads(response_text)
            
            # Validate and fix URI formats if needed
            # Validate and fix URI formats if needed
            if 'subject_uri' in triple and not triple['subject_uri'].startswith('http://dbpedia.org/resource/'):
                # Clean the URI and add proper prefix
                clean_uri = triple['subject_uri'].replace(' ', '_').replace('http://dbpedia.org/resource/', '')
                triple['subject_uri'] = f"http://dbpedia.org/resource/{clean_uri}"
            
            if 'object_uri' in triple and not triple['object_uri'].startswith('http://dbpedia.org/resource/'):
                # Clean the URI and add proper prefix
                clean_uri = triple['object_uri'].replace(' ', '_').replace('http://dbpedia.org/resource/', '')
                triple['object_uri'] = f"http://dbpedia.org/resource/{clean_uri}"
            
            if 'predicate_uri' in triple and not triple['predicate_uri'].startswith('http://dbpedia.org/ontology/'):
                # Clean the URI and add proper prefix
                clean_uri = triple['predicate_uri'].replace(' ', '_').replace('http://dbpedia.org/ontology/', '')
                triple['predicate_uri'] = f"http://dbpedia.org/ontology/{clean_uri}"
            
            print("✓ LLM disambiguation completed")
            return triple
        except Exception as e:
            print(f"✗ Error in LLM disambiguation: {e}")
            return None
    
    def collect_context(self, subject: str, object_entity: str, article_text: str = None) -> Dict:
        """Collect context using Gemini API"""
        context = {
            'sentences': [],
            'subject_types': [],
            'object_types': [],
            'predicates_between': []
        }
        
        try:
            if article_text:
                # Use Gemini to find relevant sentences
                prompt = f"""Given this text: "{article_text[:1000]}"

Find sentences that mention both "{subject}" and "{object_entity}".

Return as JSON array of sentences:
["sentence 1", "sentence 2", ...]"""
                
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
                context['sentences'] = json.loads(response_text)
                print(f"✓ Gemini found {len(context['sentences'])} relevant sentences")
        except Exception as e:
            print(f"⚠ Error collecting context with Gemini: {e}")
        
        return context
    
    def run_pipeline(self, article_text: str) -> List[Tuple[str, str, str]]:
        """Main pipeline function that processes text and returns resolved triples"""
        print(f"\n�� Starting NEF Pipeline for text: '{article_text[:100]}...'")
        print("=" * 80)
        
        # Step 1: Extract raw triples with Gemini
        print("\n📝 Step 1: Extracting raw triples...")
        raw_triples = self.extract_triples_with_gemini(article_text)
        if not raw_triples:
            print("✗ No triples extracted, exiting pipeline")
            return []
        
        resolved_triples = []
        
        # Step 2: Process each triple
        for i, triple in enumerate(raw_triples, 1):
            print(f"\n�� Step 2.{i}: Processing triple: {triple['subject']} - {triple['predicate']} - {triple['object']}")
            
            # Step 2a: Resolve entities with Redis (with LLM fallback)
            print("   📍 Resolving entities...")
            subject_candidates = self.resolve_entity_with_redis(triple['subject'])
            object_candidates = self.resolve_entity_with_redis(triple['object'])
            
            if not subject_candidates or not object_candidates:
                print(f"   ⚠ Skipping triple {i} - entity resolution failed")
                continue
            
            # Step 2b: Get candidate predicates
            print("   🔗 Getting candidate predicates...")
            predicate_candidates = self.get_candidate_predicates(triple['predicate'])
            
            # Step 2c: Collect context (optional)
            print("   📚 Collecting context...")
            context = self.collect_context(triple['subject'], triple['object'], article_text)
            
            # Step 2d: LLM disambiguation
            print("   🧠 Running LLM disambiguation...")
            disambiguated = self.llm_disambiguate_triple(
                article_text, 
                subject_candidates, 
                predicate_candidates, 
                object_candidates
            )
            
            if disambiguated:
                resolved_triple = (
                    disambiguated['subject_uri'],
                    disambiguated['predicate_uri'], 
                    disambiguated['object_uri']
                )
                resolved_triples.append(resolved_triple)
                print(f"   ✓ Resolved: {resolved_triple}")
            else:
                print(f"   ✗ Disambiguation failed for triple {i}")
        
        print(f"\n�� Pipeline completed! Resolved {len(resolved_triples)} triples")
        return resolved_triples


def main():
    """CLI wrapper for the NEF pipeline"""
    parser = argparse.ArgumentParser(description='Neural Extraction Framework Pipeline')
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
        pipeline = NEFPipeline(api_key)
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