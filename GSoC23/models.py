# Using this file to keep these models donloaded on the local system.

# Transformer models for relation extraction 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
rebel_tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
rebel_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
print("Downloaded REBEL")

# Transformer models for entity linking
genre_tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
genre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink")
print("Downloaded GENRE")

# NLTK models
import nltk
nltk.download('punkt')
print("nltk punkt downloaded")

# Spacy models
import spacy
spacy.download("en_core_web_sm") # Or specify other model if needed
