#!/usr/bin/env python3
"""
Quick test script for entity-linking-master integration.
Run this to quickly test if the integration works.
"""

import sys
import os
import argparse
import pandas as pd

# Add entity-linking-master to path
sys.path.append('../../entity-linking-master')

def quick_test(api_key: str = None):
    """Quick test of entity-linking-master functionality"""
    
    print("🚀 Quick Entity-Linking Test")
    print("=" * 40)
    
    # Set API key if provided
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
        print(f"✅ API key set from command line")
    else:
        print("⚠️  No API key provided - using environment variable GOOGLE_API_KEY")
    
    # Test imports
    try:
        from batch_preprocessing.full_batch_pipeline import full_batch_entity_linking
        print("✅ Successfully imported entity-linking-master")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Make sure entity-linking-master is in the correct location")
        return False
    
    # Test data
    test_data = [
        {"mention": "Apple", "context": "I work at Apple as a software engineer"},
        {"mention": "Tesla", "context": "Tesla is a leading electric car company"},
        {"mention": "Python", "context": "Python is a popular programming language"}
    ]
    
    print(f"📝 Testing with {len(test_data)} entities...")
    
    try:
        # Run the pipeline
        results = full_batch_entity_linking(
            test_data,
            canonical_chunk_size=3,
            context_chunk_size=3,
            dbpedia_chunk_size=3,
            log=True
        )
        
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Results shape: {results.shape}")
        print(f"📋 Columns: {list(results.columns)}")
        
        # Show results
        print("\n📋 Results:")
        for _, row in results.iterrows():
            status = "✅" if pd.notna(row['dbpedia_uri']) else "❌"
            print(f"{status} {row['mention']} → {row['canonical_name']} ({row['entity_type']})")
            if pd.notna(row['dbpedia_uri']):
                print(f"   URI: {row['dbpedia_uri']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_integration_wrapper(api_key: str = None):
    """Test the integration wrapper function"""
    
    print("\n🔧 Testing Integration Wrapper...")
    
    # Set API key if provided
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
    
    try:
        from batch_preprocessing.full_batch_pipeline import full_batch_entity_linking
        import pandas as pd
        
        def simple_entity_linking(text: str, max_results: int = 3):
            """Simple entity linking function"""
            
            # Simple entity extraction
            words = text.split()
            entities = []
            
            for i, word in enumerate(words):
                if len(word) > 2 and word[0].isupper():
                    context_start = max(0, i-2)
                    context_end = min(len(words), i+3)
                    context = " ".join(words[context_start:context_end])
                    
                    entities.append({
                        "mention": word,
                        "context": context
                    })
            
            if not entities:
                return []
            
            # Use entity-linking-master
            results_df = full_batch_entity_linking(
                entities[:max_results],
                canonical_chunk_size=3,
                context_chunk_size=3,
                dbpedia_chunk_size=3,
                log=False
            )
            
            # Convert to simple format
            results = []
            for _, row in results_df.iterrows():
                if pd.notna(row['dbpedia_uri']):
                    results.append({
                        'mention': row['mention'],
                        'canonical_name': row['canonical_name'],
                        'entity_type': row['entity_type'],
                        'dbpedia_uri': row['dbpedia_uri']
                    })
            
            return results
        
        # Test the wrapper
        test_text = "Apple is a technology company. Steve Jobs founded Apple in Cupertino."
        results = simple_entity_linking(test_text, max_results=3)
        
        print(f"✅ Integration wrapper found {len(results)} entities:")
        for entity in results:
            print(f"  - {entity['mention']} → {entity['canonical_name']} ({entity['entity_type']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration wrapper test failed: {e}")
        return False


def main():
    """Main function with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Quick test for entity-linking-master integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_test.py --api-key YOUR_API_KEY
  python quick_test.py -k YOUR_API_KEY
  python quick_test.py  # Uses GOOGLE_API_KEY environment variable
        """
    )
    
    parser.add_argument(
        '-k', '--api-key',
        type=str,
        help='Google API key for Gemini LLM (or set GOOGLE_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    
    # Run quick test
    success1 = quick_test(args.api_key)
    
    # Run integration wrapper test
    success2 = test_integration_wrapper(args.api_key)
    
    print("\n" + "=" * 40)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 40)
    print(f"Basic Functionality: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Integration Wrapper: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Entity-linking-master integration is working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main() 