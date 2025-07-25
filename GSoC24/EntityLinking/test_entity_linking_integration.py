#!/usr/bin/env python3
"""
Test file for integrating entity-linking-master with GSoC24 framework.
This file tests the entity-linking functionality dynamically and shows how to integrate
it with the existing pipeline without Redis dependency.
"""

import sys
import os
import pandas as pd
import time
import argparse
from typing import List, Dict, Optional

# Add entity-linking-master to path
sys.path.append('../entity-linking-master')

# Import entity-linking-master modules
try:
    from batch_preprocessing.full_batch_pipeline import full_batch_entity_linking
    from batch_preprocessing.batch_canonical_name import batch_canonical_name_normalization
    from batch_preprocessing.batch_context_analysis import batch_context_analysis
    from batch_preprocessing.batch_dbpedia_uri import batch_dbpedia_uri_lookup
    print("✅ Successfully imported entity-linking-master modules")
except ImportError as e:
    print(f"❌ Failed to import entity-linking-master: {e}")
    print("Make sure entity-linking-master is in the correct location")
    sys.exit(1)

# Import existing GSoC24 modules
try:
    from .methods import EL_lookup
    from .el_utils import query, lookup
    print("✅ Successfully imported GSoC24 EntityLinking modules")
except ImportError as e:
    print(f"⚠️  Could not import GSoC24 modules: {e}")
    print("Continuing with entity-linking-master only")


class EntityLinkingIntegrationTest:
    """Test class for entity-linking-master integration"""
    
    def __init__(self, api_key: str = None):
        self.test_results = {}
        self.api_key = api_key
        
        # Set API key if provided
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
            print(f"✅ API key set from command line")
        else:
            print("⚠️  No API key provided - using environment variable GOOGLE_API_KEY")
        
        self.sample_data = [
            {"mention": "Apple", "context": "I work at Apple as a software engineer"},
            {"mention": "Apple", "context": "I eat an apple every day for health"},
            {"mention": "Paris", "context": "I visited Paris last summer for vacation"},
            {"mention": "Barack Obama", "context": "Barack Obama was the 44th president of the United States"},
            {"mention": "Tesla", "context": "Tesla is a leading electric car company"},
            {"mention": "Amazon", "context": "Amazon is the world's largest online retailer"},
            {"mention": "Python", "context": "Python is a popular programming language"},
            {"mention": "Python", "context": "A python is a type of snake found in Asia"},
            {"mention": "Meta", "context": "Meta Platforms is the parent company of Facebook"},
            {"mention": "Meta", "context": "The concept of meta-analysis is important in statistics"},
            {"mention": "Cambridge", "context": "Cambridge is a university town in England"},
            {"mention": "Cambridge", "context": "Cambridge Analytica was a political consulting firm"}
        ]
    
    def test_basic_functionality(self) -> bool:
        """Test basic entity-linking-master functionality"""
        print("\n🧪 Testing Basic Functionality...")
        
        try:
            # Test canonical name normalization
            mentions = [e['mention'] for e in self.sample_data[:3]]
            canonical_df = batch_canonical_name_normalization(
                mentions, 
                chunk_size=3, 
                output_format="dataframe"
            )
            print(f"✅ Canonical name normalization: {len(canonical_df)} results")
            
            # Test context analysis
            context_df = batch_context_analysis(
                self.sample_data[:3], 
                chunk_size=3, 
                output_format="dataframe"
            )
            print(f"✅ Context analysis: {len(context_df)} results")
            
            # Test DBpedia URI lookup
            canonical_names = list(canonical_df['canonical_name'])
            dbpedia_df = batch_dbpedia_uri_lookup(
                canonical_names, 
                output_format="dataframe", 
                chunk_size=3
            )
            print(f"✅ DBpedia URI lookup: {len(dbpedia_df)} results")
            
            self.test_results['basic_functionality'] = True
            return True
            
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            self.test_results['basic_functionality'] = False
            return False
    
    def test_full_pipeline(self) -> bool:
        """Test the full entity linking pipeline"""
        print("\n🧪 Testing Full Pipeline...")
        
        try:
            start_time = time.time()
            
            # Run full pipeline
            results_df = full_batch_entity_linking(
                self.sample_data,
                canonical_chunk_size=5,
                context_chunk_size=5,
                dbpedia_chunk_size=5,
                log=True
            )
            
            end_time = time.time()
            
            print(f"✅ Full pipeline completed in {end_time - start_time:.2f} seconds")
            print(f"📊 Results shape: {results_df.shape}")
            print(f"📋 Columns: {list(results_df.columns)}")
            
            # Display sample results
            print("\n📋 Sample Results:")
            print(results_df.head(3).to_string())
            
            # Check for successful entity links
            successful_links = results_df[results_df['dbpedia_uri'].notna()]
            print(f"🔗 Successful entity links: {len(successful_links)}/{len(results_df)}")
            
            self.test_results['full_pipeline'] = True
            return True
            
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            self.test_results['full_pipeline'] = False
            return False
    
    def test_integration_with_csv_context(self, csv_path: Optional[str] = None) -> bool:
        """Test integration with CSV context data (when available)"""
        print("\n🧪 Testing Integration with CSV Context...")
        
        if csv_path and os.path.exists(csv_path):
            try:
                # Load CSV context data
                context_df = pd.read_csv(csv_path)
                print(f"✅ Loaded CSV context: {len(context_df)} entries")
                
                # Convert to entity contexts format
                entity_contexts = []
                for _, row in context_df.head(5).iterrows():  # Test with first 5
                    entity_contexts.append({
                        "mention": row.get('mention', row.get('entity', 'Unknown')),
                        "context": row.get('context', row.get('sentence', 'No context'))
                    })
                
                # Test with CSV context
                results_df = full_batch_entity_linking(
                    entity_contexts,
                    canonical_chunk_size=3,
                    context_chunk_size=3,
                    dbpedia_chunk_size=3,
                    log=False
                )
                
                print(f"✅ CSV context integration: {len(results_df)} results")
                self.test_results['csv_integration'] = True
                return True
                
            except Exception as e:
                print(f"❌ CSV integration test failed: {e}")
                self.test_results['csv_integration'] = False
                return False
        else:
            print("⚠️  No CSV context file provided, skipping CSV integration test")
            self.test_results['csv_integration'] = None
            return True
    
    def test_performance(self) -> bool:
        """Test performance with different chunk sizes"""
        print("\n🧪 Testing Performance...")
        
        try:
            # Test different chunk sizes
            chunk_sizes = [3, 5, 10]
            
            for chunk_size in chunk_sizes:
                start_time = time.time()
                
                results_df = full_batch_entity_linking(
                    self.sample_data[:6],  # Use subset for performance test
                    canonical_chunk_size=chunk_size,
                    context_chunk_size=chunk_size,
                    dbpedia_chunk_size=chunk_size,
                    log=False
                )
                
                end_time = time.time()
                print(f"✅ Chunk size {chunk_size}: {end_time - start_time:.2f}s for {len(results_df)} results")
            
            self.test_results['performance'] = True
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid inputs"""
        print("\n🧪 Testing Error Handling...")
        
        try:
            # Test with empty input
            empty_results = full_batch_entity_linking([], log=False)
            print(f"✅ Empty input handling: {len(empty_results)} results")
            
            # Test with invalid mentions
            invalid_data = [
                {"mention": "", "context": "Empty mention"},
                {"mention": "   ", "context": "Whitespace mention"},
                {"mention": "Valid", "context": "Valid context"}
            ]
            
            invalid_results = full_batch_entity_linking(invalid_data, log=False)
            print(f"✅ Invalid input handling: {len(invalid_results)} results")
            
            self.test_results['error_handling'] = True
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.test_results['error_handling'] = False
            return False
    
    def run_all_tests(self, csv_path: Optional[str] = None) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 Starting Entity-Linking Integration Tests...")
        print("=" * 60)
        
        tests = [
            ("Basic Functionality", self.test_basic_functionality),
            ("Full Pipeline", self.test_full_pipeline),
            ("Performance", self.test_performance),
            ("Error Handling", self.test_error_handling),
            ("CSV Integration", lambda: self.test_integration_with_csv_context(csv_path))
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                self.test_results[test_name.lower().replace(' ', '_')] = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL" if result is False else "⚠️  SKIP"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        return self.test_results
    
    def save_test_results(self, output_path: str = "entity_linking_test_results.csv"):
        """Save test results to CSV"""
        try:
            results_df = pd.DataFrame([self.test_results])
            results_df.to_csv(output_path, index=False)
            print(f"✅ Test results saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save test results: {e}")


def create_integration_wrapper():
    """Create a wrapper function that integrates entity-linking-master with GSoC24"""
    
    def integrated_entity_linking(
        text: str,
        max_results: int = 5,
        use_csv_context: bool = False,
        csv_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Integrated entity linking that combines entity-linking-master with GSoC24
        
        Args:
            text: Input text to extract entities from
            max_results: Maximum number of entities to return
            use_csv_context: Whether to use CSV context data
            csv_path: Path to CSV context file
            
        Returns:
            List of entity dictionaries
        """
        
        # Simple entity extraction (you can enhance this)
        # For now, we'll use a simple approach
        entities = []
        
        # Extract potential entities (simple approach)
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 2 and word[0].isupper():  # Simple entity detection
                context_start = max(0, i-3)
                context_end = min(len(words), i+4)
                context = " ".join(words[context_start:context_end])
                
                entities.append({
                    "mention": word,
                    "context": context
                })
        
        if not entities:
            return []
        
        # Use entity-linking-master for linking
        try:
            results_df = full_batch_entity_linking(
                entities[:max_results],
                canonical_chunk_size=5,
                context_chunk_size=5,
                dbpedia_chunk_size=5,
                log=False
            )
            
            # Convert to list of dictionaries
            results = []
            for _, row in results_df.iterrows():
                if pd.notna(row['dbpedia_uri']):  # Only return successful links
                    results.append({
                        'mention': row['mention'],
                        'canonical_name': row['canonical_name'],
                        'entity_type': row['entity_type'],
                        'confidence': row['confidence'],
                        'dbpedia_uri': row['dbpedia_uri'],
                        'description': row['description']
                    })
            
            return results[:max_results]
            
        except Exception as e:
            print(f"Entity linking failed: {e}")
            return []
    
    return integrated_entity_linking


def main():
    """Main function with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for entity-linking-master integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_entity_linking_integration.py --api-key YOUR_API_KEY
  python test_entity_linking_integration.py -k YOUR_API_KEY
  python test_entity_linking_integration.py  # Uses GOOGLE_API_KEY environment variable
        """
    )
    
    parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='Google API key for Gemini LLM (or set GOOGLE_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        help='Path to CSV context file (optional)'
    )
    
    args = parser.parse_args()
    
    print("🔧 Entity-Linking Integration Test Suite")
    print("=" * 60)
    
    # Create test instance with API key
    tester = EntityLinkingIntegrationTest(args.api_key)
    
    # Check for CSV context file
    csv_path = args.csv_path
    if not csv_path:
        possible_csv_paths = [
            "Data/surface_forms.csv",
            "Data/entity_redirects.csv",
            "surface_forms.csv",
            "entity_redirects.csv"
        ]
        
        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_path = path
                print(f"📁 Found CSV context file: {path}")
                break
    
    # Run all tests
    results = tester.run_all_tests(csv_path)
    
    # Save results
    tester.save_test_results()
    
    # Create integration wrapper
    integrated_el = create_integration_wrapper()
    
    # Test the integration wrapper
    print("\n🧪 Testing Integration Wrapper...")
    test_text = "Apple is a technology company based in Cupertino. Steve Jobs founded Apple."
    results = integrated_el(test_text, max_results=3)
    
    print(f"✅ Integration wrapper found {len(results)} entities:")
    for entity in results:
        print(f"  - {entity['mention']} → {entity['canonical_name']} ({entity['entity_type']})")
    
    print("\n🎉 Integration test completed!")


if __name__ == "__main__":
    main() 