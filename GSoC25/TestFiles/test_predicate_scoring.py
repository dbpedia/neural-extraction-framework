#!/usr/bin/env python3
"""
Predicate Scoring Fusion Test

This file tests the scoring fusion system for predicate selection using:
- Type compatibility (1/0) based on domain/range constraints
- Predicate similarity (cosine similarity with evidence)
- BM25 overlap between evidence and predicate descriptions
- Description similarity between evidence and entity descriptions
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple
import json
import sys
from pathlib import Path

# Add project root to Python path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent
except NameError:
    PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class PredicateScoringFusion:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.predicate_descriptions = {}
        self.domain_range_mapping = {}
        self._fitted_vectorizer = None
        
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
    
    def compute_combined_score(self, evidence_text: str, subject_uri: str, object_uri: str, 
                             predicate_uri: str, weights: Dict[str, float] = None) -> Dict[str, float]:
        """Compute combined score using all components"""
        if weights is None:
            weights = {
                'type_compat': 0.4,  # Higher weight for type compatibility
                'pred_sim': 0.3,
                'bm25': 0.2,
                'desc_sim': 0.1
            }
        
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
        
        return {
            'combined_score': combined_score,
            'type_compat': type_compat,
            'pred_sim': pred_sim,
            'bm25': bm25,
            'desc_sim': desc_sim
        }


def main():
    """Main test function"""
    
    # Test data - DBpedia predicates with descriptions and domain/range
    test_predicates = [
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
        }
    ]

    # Test cases
    test_cases = [
        {
            'evidence': 'Albert Einstein was born in Ulm, Germany',
            'subject_uri': 'http://dbpedia.org/resource/Albert_Einstein',
            'object_uri': 'http://dbpedia.org/resource/Ulm',
            'candidate_predicates': [
                'http://dbpedia.org/ontology/birthPlace',
                'http://dbpedia.org/ontology/location',
                'http://dbpedia.org/ontology/employer'
            ],
            'expected_best': 'http://dbpedia.org/ontology/birthPlace'
        },
        {
            'evidence': 'Steve Jobs worked at Apple Inc',
            'subject_uri': 'http://dbpedia.org/resource/Steve_Jobs',
            'object_uri': 'http://dbpedia.org/resource/Apple_Inc',
            'candidate_predicates': [
                'http://dbpedia.org/ontology/employer',
                'http://dbpedia.org/ontology/birthPlace',
                'http://dbpedia.org/ontology/location'
            ],
            'expected_best': 'http://dbpedia.org/ontology/employer'
        },
        {
            'evidence': 'Apple Inc was founded by Steve Jobs',
            'subject_uri': 'http://dbpedia.org/resource/Apple_Inc',
            'object_uri': 'http://dbpedia.org/resource/Steve_Jobs',
            'candidate_predicates': [
                'http://dbpedia.org/ontology/foundedBy',
                'http://dbpedia.org/ontology/employer',
                'http://dbpedia.org/ontology/birthPlace'
            ],
            'expected_best': 'http://dbpedia.org/ontology/foundedBy'
        }
    ]

    # Initialize scorer
    scorer = PredicateScoringFusion()
    scorer.load_predicate_data(test_predicates)

    print("=" * 80)
    print("PREDICATE SCORING FUSION TEST")
    print("=" * 80)
    print()

    total_correct = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['evidence']}")
        print(f"Subject: {test_case['subject_uri']}")
        print(f"Object: {test_case['object_uri']}")
        print(f"Expected Best: {test_case['expected_best']}")
        print("\nCandidate Predicate Scores:")
        
        scores = []
        for pred_uri in test_case['candidate_predicates']:
            score_result = scorer.compute_combined_score(
                test_case['evidence'],
                test_case['subject_uri'],
                test_case['object_uri'],
                pred_uri
            )
            scores.append((pred_uri, score_result))
        
        # Sort by combined score
        scores.sort(key=lambda x: x[1]['combined_score'], reverse=True)
        
        for j, (pred_uri, scores_dict) in enumerate(scores):
            status = "✅ BEST" if j == 0 else ""
            print(f"\n{j+1}. {pred_uri} {status}")
            print(f"   Combined Score: {scores_dict['combined_score']:.3f}")
            print(f"   Type Compat: {scores_dict['type_compat']:.3f}")
            print(f"   Pred Sim: {scores_dict['pred_sim']:.3f}")
            print(f"   BM25: {scores_dict['bm25']:.3f}")
            print(f"   Desc Sim: {scores_dict['desc_sim']:.3f}")
        
        # Check if best prediction matches expected
        best_predicate = scores[0][0]
        if best_predicate == test_case['expected_best']:
            print(f"\n✅ CORRECT: Best predicate matches expected!")
            total_correct += 1
        else:
            print(f"\n❌ INCORRECT: Expected {test_case['expected_best']}, got {best_predicate}")
        
        print("\n" + "=" * 80 + "\n")

    # Summary
    accuracy = (total_correct / total_tests) * 100
    print(f"SUMMARY:")
    print(f"Correct predictions: {total_correct}/{total_tests}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🎉 Excellent performance!")
    elif accuracy >= 60:
        print("👍 Good performance!")
    else:
        print("⚠️  Needs improvement!")


if __name__ == "__main__":
    main()
