import os
import json
import google.generativeai as genai
import sys
from pathlib import Path
import pickle
import argparse

# Add project root to Python path
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent.parent 
    PROJECT_ROOT = Path().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GSoC24.Data.collector import get_text_of_wiki_page
from GSoC24.RelationExtraction.relation_similarity import ontosim_search, load_key_vector_model_from_file
from GSoC24.RelationExtraction.text_encoding_models import get_sentence_transformer_model

def get_pickle_object(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


article_name = "Albert Einstein"
original_text = get_text_of_wiki_page(article_name)
print("Original text (from Wikipedia):", original_text[:300], "...") 

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


CONFIG_PATH = PROJECT_ROOT / "GSoC24" / "RelationExtraction" / "config.json"
with open(CONFIG_PATH, "r") as config_file:
    config = json.load(config_file)

file_paths = config.get("file_paths", {})
for key, filename in file_paths.items():
    file_paths[key] = str(PROJECT_ROOT / "GSoC24" / "RelationExtraction" / filename)

label_embeddings_file = config["file_paths"]["label_embeddings_file"]
gensim_model = load_key_vector_model_from_file(label_embeddings_file)

tbox_pickle_file = config["file_paths"]["tbox_pickle_file"]
tbox = get_pickle_object(tbox_pickle_file)

encoder_model = get_sentence_transformer_model(
    model_name=config["model_names"]["encoder_model"]
)

print("Real models loaded successfully!")

relation_text = "was born in"
candidate_predicate_df = ontosim_search(relation_text, gensim_model, encoder_model, tbox)
candidate_predicate_uris = []
for uri_list in candidate_predicate_df["URIs"].tolist():
    if isinstance(uri_list, list):
        for uri in uri_list:
            if uri and uri.startswith("http://"):
                candidate_predicate_uris.append(uri)
                break
    elif isinstance(uri_list, str) and uri_list.startswith("http://"):
        candidate_predicate_uris.append(uri_list)

if not candidate_predicate_uris:
    candidate_predicate_uris = [
        "http://dbpedia.org/ontology/birthPlace",
        "http://dbpedia.org/ontology/location",
        "http://dbpedia.org/ontology/country"
    ]

print("Candidate predicate URIs (from real embedding search):", candidate_predicate_uris)


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