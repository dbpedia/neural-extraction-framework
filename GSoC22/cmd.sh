#!/bin/bash

pip install wikipedia
pip install transformers
pip install nltk
pip install fuzzywuzzy
pip install rdflib
pip install SPARQLWrapper
pip install -U spacy==2.1.0 
python -m spacy download en
pip uninstall -y neuralcoref 
pip install neuralcoref --no-binary neuralcoref