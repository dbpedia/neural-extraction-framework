# Using this file to keep these models downloaded on the local system.

# Transformer models for relation extraction 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Transformer models for entity linking
genre_tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
genre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink")
print("Downloaded GENRE")

# NLTK models
import nltk
nltk.download('punkt')
print("nltk punkt downloaded")

