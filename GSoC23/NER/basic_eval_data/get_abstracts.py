from SPARQLWrapper import SPARQLWrapper, JSON

# These 2 lines help import from the Data dir
import sys
sys.path.append("/home/aakash/College/GSoC/neural-extraction-framework/GSoC23")

from Data.collector import get_abstract

import pickle

# Create a SPARQLWrapper object with the DBpedia SPARQL endpoint URL
sparql = SPARQLWrapper("http://dbpedia.org/sparql")

five_uris = ['Lionel_Messi', 'Rio_de_Janeiro',' Wood', 'Ice_cream', 'Car']
abstracts_and_entities = {}

for uri in five_uris:
    try:
        print(f"Fetching abstract for {uri}...")
        abst = get_abstract(sparql_wrapper=sparql, uri=uri)
        abstracts_and_entities[uri]={}
        abstracts_and_entities[uri]['abstract']=abst
        abstracts_and_entities[uri]['entities']=[]
    except:
        continue

with open("/home/aakash/College/GSoC/neural-extraction-framework/GSoC23/NER/basic_eval_data/asbtracts_and_entities.pkl","wb") as file:
    pickle.dump(abstracts_and_entities, file)