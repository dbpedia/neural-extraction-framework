import sys
from pathlib import Path
import json
import pandas as pd
import google.generativeai as genai

# Add project root to path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent 
except NameError:
    PROJECT_ROOT = Path().resolve() / "neural-extraction-framework"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import existing methods
from GSoC24.RelationExtraction.methods import get_triples_from_sentence
from GSoC24.RelationExtraction.llm_utils import response_to_triples
import llama_cpp
from llama_cpp import Llama
from outlines import generate, models

def extract_triples_with_gemini(text, api_key):
    """Extract triples using Gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
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
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.startswith('```'):
            response_text = response_text[3:]  # Remove ```
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        
        response_text = response_text.strip()
        triples = json.loads(response_text)
        return triples
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.text}")
        return []
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []

def extract_triples_with_llama(text, llama_model):
    """Extract triples using the current Llama-based approach"""
    try:
        triples_df = response_to_triples(text, llama_model)
        # Convert DataFrame to list of dictionaries
        triples = []
        for _, row in triples_df.iterrows():
            triples.append({
                "subject": row['Subject'],
                "predicate": row['Predicate'], 
                "object": row['Object']
            })
        return triples
    except Exception as e:
        print(f"Error with Llama extraction: {e}")
        return []

def load_llama_model():
    """Load the Llama model for comparison"""
    MODEL_PATH = PROJECT_ROOT / "GSoC24" / "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf"
    llm = Llama(
        str(MODEL_PATH),
        tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            "NousResearch/Hermes-2-Pro-Llama-3-8B"
        ),
        n_gpu_layers=-1,
        flash_attn=True,
        n_ctx=8192,
        verbose=False
    )
    return models.LlamaCpp(llm)

def compare_extraction_methods(text, gemini_api_key, use_llama=True):
    """Compare different relation extraction methods"""
    results = {}
    
    # Method 1: Gemini API
    print("Extracting triples with Gemini API...")
    gemini_triples = extract_triples_with_gemini(text, gemini_api_key)
    results['Gemini'] = gemini_triples
    print(f"Gemini found {len(gemini_triples)} triples")
    
    # Method 2: Llama (if requested)
    if use_llama:
        print("Extracting triples with Llama...")
        try:
            llama_model = load_llama_model()
            llama_triples = extract_triples_with_llama(text, llama_model)
            results['Llama'] = llama_triples
            print(f"Llama found {len(llama_triples)} triples")
        except Exception as e:
            print(f"Could not load Llama model: {e}")
            results['Llama'] = []
    
    return results

def print_comparison(results):
    """Print comparison results in a readable format"""
    print("\n" + "="*60)
    print("RELATION EXTRACTION COMPARISON")
    print("="*60)
    
    for method, triples in results.items():
        print(f"\n{method.upper()} RESULTS:")
        print("-" * 30)
        if triples:
            for i, triple in enumerate(triples, 1):
                print(f"{i}. Subject: {triple['subject']}")
                print(f"   Predicate: {triple['predicate']}")
                print(f"   Object: {triple['object']}")
                print()
        else:
            print("No triples extracted")

def save_comparison_results(results, filename="extraction_comparison.csv"):
    """Save comparison results to CSV"""
    all_triples = []
    for method, triples in results.items():
        for triple in triples:
            all_triples.append({
                'method': method,
                'subject': triple['subject'],
                'predicate': triple['predicate'],
                'object': triple['object']
            })
    
    df = pd.DataFrame(all_triples)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare relation extraction methods')
    parser.add_argument("--text", required=True, help="Text to extract triples from")
    parser.add_argument("--api_key", required=True, help="Gemini API key")
    parser.add_argument("--no_llama", action="store_true", help="Skip Llama comparison")
    parser.add_argument("--save", default="extraction_comparison.csv", help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_extraction_methods(
        args.text, 
        args.api_key, 
        use_llama=not args.no_llama
    )
    
    # Print results
    print_comparison(results)
    
    # Save results
    save_comparison_results(results, args.save) 