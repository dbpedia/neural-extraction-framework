# ----------------------------------------------------------------------------------
# This file is the initial version of using the end-2-end framework.
# This is subject to a lot of changes and optimizations.
# WARNING - Running this file on large texts may result in lot of RAM consumption.
# We are working to make this more efficient and fast.
# ----------------------------------------------------------------------------------
import sys
from pathlib import Path

try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent 
except NameError:
    PROJECT_ROOT = Path().resolve() / "neural-extraction-framework"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import tqdm
import pandas as pd
import llama_cpp
from llama_cpp import Llama
from outlines import generate, models
from nltk import sent_tokenize
from GSoC24.Data.collector import get_text_of_wiki_page
from GSoC24.RelationExtraction.methods import get_triples_from_sentence
import google.generativeai as genai
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import argparse

parser = argparse.ArgumentParser(description='An end-2-end utility program')
parser.add_argument("--sentence", default=None, 
                    help="The sentence on which the user wants to run triple extraction")
parser.add_argument("--text", default="", 
                    help="The text on which the user wants to run triple extraction")
parser.add_argument("--wikipage", default=None, 
                    help="The title of wikipedia page on which to perform relation extraction")
parser.add_argument("--save_filename", default=None, 
                    help="The file name of the csv of triples, if this is specified, the file will be saved, else not")
parser.add_argument("--v", default=0, help="If set to 1, print the triples dataframe")
parser.add_argument("--text_filepath", default="", 
                    help="The text file on which the user wants to run triple extraction")
parser.add_argument("--method", default="llama", choices=["llama", "gemini"], 
                    help="Choose extraction method: llama or gemini")
parser.add_argument("--api_key", default=None, 
                    help="Gemini API key (required if method is gemini)")
args = parser.parse_args()

def extract_triples_with_gemini(text, api_key):
    """Extract triples using Gemini API with entity linking and predicate mapping"""
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
        
        # Import the function for entity linking and predicate mapping
        from GSoC24.RelationExtraction.re_utils import get_triple_from_triple
        
        # Convert to the format expected by the pipeline with entity linking and predicate mapping
        formatted_triples = []
        for triple in triples:
            try:
                # Get URIs through entity linking and predicate mapping
                subject_uri, predicate_uri, object_uri = get_triple_from_triple(
                    triple["subject"], 
                    triple["predicate"], 
                    triple["object"]
                )
                
                formatted_triples.append({
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                    "subject_URI": subject_uri,
                    "predicate_URI": predicate_uri,
                    "object_URI": object_uri
                })
            except Exception as e:
                print(f"Error processing triple {triple}: {e}")
                # Add triple without URIs if processing fails
                formatted_triples.append({
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"],
                    "subject_URI": "",
                    "predicate_URI": "",
                    "object_URI": ""
                })
        
        return formatted_triples
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response.text}")
        return []
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return []

# Load Llama model only if needed
llama_model = None
if args.method == "llama":
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
    llama_model = models.LlamaCpp(llm)
    print("llama model loaded")
elif args.method == "gemini":
    if not args.api_key:
        print("Error: --api_key is required when using gemini method")
        exit(1)
    print("Using Gemini API for triple extraction")

sentences = None
if args.sentence:
    sentences = args.sentence
elif args.text:
    sentences = args.text
elif args.wikipage:
    article_text = get_text_of_wiki_page(args.wikipage)
    sentences = article_text
elif args.text_filepath:
    with open(args.text_filepath, "r") as f:
        print("Reading text from file...")
        text = f.read()
        sentences = text

triples = []

print("Extracting triples...")
if args.method == "llama":
    sentence_triples = get_triples_from_sentence(user_prompt=sentences, model=llama_model)
    for sent_trip in sentence_triples:
        triples.append(sent_trip)
elif args.method == "gemini":
    triples = extract_triples_with_gemini(sentences, args.api_key)
print("Done")

triples_dataframe = pd.DataFrame(data=triples)

if args.save_filename:
    triples_dataframe.to_csv(args.save_filename)
    print("Triples saved to file")

if int(args.v)==1:
    print(triples_dataframe)
