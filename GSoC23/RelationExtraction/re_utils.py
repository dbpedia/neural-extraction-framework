import pickle
import logging
from GSoC23.EntityLinking.methods import EL_GENRE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from GSoC23.RelationExtraction.relation_similarity import load_key_vector_model_from_file
from GSoC23.RelationExtraction.text_encoding_models import get_sentence_transformer_model
from GSoC23.RelationExtraction.relation_similarity import ontosim_search
from GSoC23.EntityLinking.el_utils import annotate_sentence

label_embeddings_file = "/home/aakash/D/College/GSoC/neural-extraction-framework/GSoC23/RelationExtraction/dbpedia-ontology.vectors"
gensim_model = load_key_vector_model_from_file(label_embeddings_file)

tbox_pickle_file = "/home/aakash/D/College/GSoC/neural-extraction-framework/GSoC23/RelationExtraction/tbox.pkl"
labels_pickle_file = "/home/aakash/D/College/GSoC/neural-extraction-framework/GSoC23/RelationExtraction/labels.pkl"

def get_pickle_object(file):
    """This function can be used to load the labels.pkl
    and tbox.pkl files.
    """
    with open(file,"rb") as f:
        return pickle.load(f)
    
labels=get_pickle_object(labels_pickle_file)
tbox=get_pickle_object(tbox_pickle_file)

encoder_model = get_sentence_transformer_model(model_name="paraphrase-MiniLM-L6-v2")
logging.info("Encoder model loaded")

genre_tokenizer = AutoTokenizer.from_pretrained("facebook/genre-linking-blink")
genre_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/genre-linking-blink").eval()
logging.info("GENRE model loaded")

def get_triple_from_triple(sub, relation, obj, sentence):
    """Give subject, relation, object in natural language form,
    return the corressponding Dbpedia URIs for them.
    Args:
        sub (str): Subject
        relation (str): Relation
        obj (str): Object
        sentence (str): The sentence in which they appear.

    Returns:
        tuple: A tuple of entity and predicate URIs
    """
    
    subject_entity = EL_GENRE(
        annotate_sentence(sentence, sub), genre_model, genre_tokenizer)[0]
    subject_entity = "https://dbpedia.org/resource/"+"_".join(subject_entity.split())
    
    object_entity = EL_GENRE(
        annotate_sentence(sentence, obj), genre_model, genre_tokenizer)[0]
    object_entity = "https://dbpedia.org/resource/"+"_".join(object_entity.split())
    
    predicates, label, score = ontosim_search(
        relation, gensim_model, encoder_model, tbox).iloc[0].values
    
    predicate = None
    for p in predicates:
        if p.split("/")[-1][0].islower():
            predicate = p
            break
    # can also return this - (subject_entity, (predicate, score), object_entity)
    return (subject_entity, predicate, object_entity)

