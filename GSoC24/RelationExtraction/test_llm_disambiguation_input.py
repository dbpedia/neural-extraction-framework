import os
import json
import google.generativeai as genai
import sys
from pathlib import Path

# Add project root to Python path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent.parent  # Go up 3 levels to reach project root
except NameError:
    PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GSoC24.Data.collector import get_text_of_wiki_page
from GSoC24.RelationExtraction.relation_similarity import ontosim_search
from GSoC24.RelationExtraction.text_encoding_models import get_sentence_transformer_model
import pickle
import argparse

# 1. Fetch original text (context) from Wikipedia using real function
article_name = "Albert Einstein"
original_text = get_text_of_wiki_page(article_name)
print("Original text (from Wikipedia):", original_text[:300], "...")  # Print first 300 chars

# 2. Simulate candidate entity URIs (mocked Redis output)
candidate_subject_uris = [
    "http://dbpedia.org/resource/Albert_Einstein",
    "http://dbpedia.org/resource/Einstein_(surname)",
    "http://dbpedia.org/resource/Albert_Einstein_Hospital"
]
candidate_object_uris = [
    "http://dbpedia.org/resource/Ulm",
    "http://dbpedia.org/resource/Germany",
    "http://dbpedia.org/resource/Ulm_Minster"
]
print("Candidate subject URIs (from Redis):", candidate_subject_uris)
print("Candidate object URIs (from Redis):", candidate_object_uris)

# 3. Generate candidate predicate URIs using embedding search (real function)
# Load models and tbox as in the main pipeline
# You may need to adjust these paths based on your setup
config = {
    "model_names": {
        "encoder_model": "all-MiniLM-L6-v2"
    }
}

def get_pickle_object(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Load tbox (DBpedia ontology labels to URIs)
tbox_file = os.path.join(os.path.dirname(__file__), "../RelationExtraction/tbox.pkl")
tbox = get_pickle_object(tbox_file)

# Load a sentence transformer model
encoder_model = get_sentence_transformer_model(model_name=config["model_names"]["encoder_model"])

# For demonstration, we use a mock gensim model (not loaded here)
class DummyGensimModel:
    def most_similar(self, positive, topn=5):
        # Return dummy results with proper URI format
        return [
            ("birthPlace", 0.95),
            ("location", 0.90),
            ("birth", 0.85),
            ("country", 0.80),
        ]

gensim_model = DummyGensimModel()

relation_text = "was born in"
candidate_predicate_df = ontosim_search(relation_text, gensim_model, encoder_model, tbox)
# Clean up the candidate predicate URIs to ensure they're in proper format
candidate_predicate_uris = []
for uri_list in candidate_predicate_df["URIs"].tolist():
    if isinstance(uri_list, list):
        # If it's a list, take the first valid URI
        for uri in uri_list:
            if uri and uri.startswith("http://"):
                candidate_predicate_uris.append(uri)
                break
    elif isinstance(uri_list, str) and uri_list.startswith("http://"):
        candidate_predicate_uris.append(uri_list)

# If no valid URIs found, use some default ones
if not candidate_predicate_uris:
    candidate_predicate_uris = [
        "http://dbpedia.org/ontology/birthPlace",
        "http://dbpedia.org/ontology/location",
        "http://dbpedia.org/ontology/country"
    ]

print("Candidate predicate URIs (from embedding search):", candidate_predicate_uris)

# 4. LLM Disambiguation using Gemini

def llm_disambiguate_with_gemini(original_text, candidate_subject_uris, candidate_predicate_uris, candidate_object_uris, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
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

CRITICAL: You MUST return the EXACT URIs from the candidate lists above. Do NOT modify, shorten, or change them in any way.
- Copy the subject_uri exactly from the candidate_subject_uris list
- Copy the predicate_uri exactly from the candidate_predicate_uris list  
- Copy the object_uri exactly from the candidate_object_uris list

Respond in JSON as:
{{
  "subject_uri": "http://dbpedia.org/resource/...",
  "predicate_uri": "http://dbpedia.org/ontology/...",
  "object_uri": "http://dbpedia.org/resource/..."
}}
'''
    response = model.generate_content(prompt)
    response_text = response.text.strip()
    # Remove markdown code blocks if present
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    response_text = response_text.strip()
    try:
        triple = json.loads(response_text)
    except Exception as e:
        print("Error parsing Gemini response:", e)
        print("Raw response:", response_text)
        triple = None
    return triple

# Get Gemini API key from environment variable or set here
parser = argparse.ArgumentParser(description="Test LLM Disambiguation Input with Gemini")
parser.add_argument("--api_key", type=str, default=None, help="Gemini API key")
args = parser.parse_args()

GEMINI_API_KEY = args.api_key or os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

print("\nCalling Gemini for LLM disambiguation...")
triple = llm_disambiguate_with_gemini(
    original_text,
    candidate_subject_uris,
    candidate_predicate_uris,
    candidate_object_uris,
    GEMINI_API_KEY
)

print("\nLLM Disambiguation Output:")
print(json.dumps(triple, indent=2)) 