# ----------------------------------------------------------------------------------
# This file is the initial version of using the end-2-end framework.
# This is subject to a lot of changes and optimizations.
# WARNING - Running this file on large texts may result in lot of RAM consumption.
# We are working to make this more efficient and fast.
# ----------------------------------------------------------------------------------
import sys
sys.path.append("/content/neural-extraction-framework/")
import tqdm
import pandas as pd
from nltk import sent_tokenize
from GSoC23.Data.collector import get_text_of_wiki_page
from GSoC23.RelationExtraction.methods import get_triples_from_sentence_using_rebel

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

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
print("Tokenizer and model loaded")

sentences = None
if args.sentence:
    sentences = sent_tokenize(args.sentence)
elif args.text:
    sentences = sent_tokenize(args.text)
elif args.wikipage:
    article_text = get_text_of_wiki_page(args.wikipage)
    sentences = sent_tokenize(article_text)
elif args.text_filepath:
    with open(args.text_filepath, "r") as f:
        print("Reading text from file...")
        text = f.read()
        sentences = sent_tokenize(text)

triples = []

print("Extracting triples...")
for sentence in tqdm.tqdm(sentences):
    sentence_triples = get_triples_from_sentence_using_rebel(sentence, model, tokenizer)
    for sent_trip in sentence_triples:
        triples.append(sent_trip)
print("Done")

triples_dataframe = pd.DataFrame(data=triples)

if args.save_filename:
    triples_dataframe.to_csv(args.save_filename)
    print("Triples saved to file")

if int(args.v)==1:
    print(triples_dataframe)
