#----------------------------------------------------------------------------------
# This file is the initial version of using the end-2-end framework.
# This is subject to a lot of changes and optimizations.
# WARNING - Running this file on large texts may result in lot of RAM and time
# consumption. Though I have shown with wikipedia text, please consider
# changing to some smaller text(around 5-10 lines) to see quick results.
# We are working to make this more efficient and fast.
#----------------------------------------------------------------------------------
import pandas as pd
from nltk import sent_tokenize
from GSoC23.Data.collector import get_text_of_wiki_page
from GSoC23.RelationExtraction.methods import get_triples_from_sentence_using_rebel

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

wiki_page = "Pacific ocean"
article_text = get_text_of_wiki_page(wiki_page)

sentences = sent_tokenize(article_text)


triples = []

for sentence in sentences:
    sentence_triples = get_triples_from_sentence_using_rebel(sentence, model, tokenizer)
    for sent_trip in sentence_triples:
        triples.append(sent_trip)

triples_dataframe = pd.DataFrame(data=triples)
