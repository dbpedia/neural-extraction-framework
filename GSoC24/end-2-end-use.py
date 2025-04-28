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
args = parser.parse_args()

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
sentence_triples = get_triples_from_sentence(user_prompt=sentences, model=llama_model)
for sent_trip in sentence_triples:
    triples.append(sent_trip)
print("Done")

triples_dataframe = pd.DataFrame(data=triples)

if args.save_filename:
    triples_dataframe.to_csv(args.save_filename)
    print("Triples saved to file")

if int(args.v)==1:
    print(triples_dataframe)
