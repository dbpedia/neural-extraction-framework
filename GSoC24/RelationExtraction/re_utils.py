import json
import pickle
import logging
from GSoC24.EntityLinking.methods import EL_GENRE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from GSoC24.RelationExtraction.relation_similarity import (
    load_key_vector_model_from_file,
)
from GSoC24.RelationExtraction.text_encoding_models import (
    get_sentence_transformer_model,
)
from GSoC24.RelationExtraction.relation_similarity import ontosim_search
from GSoC23.EntityLinking.el_utils import annotate_sentence

with open("/content/neural-extraction-framework/GSoC24/RelationExtraction/config.json", "r") as config_file:
    config = json.load(config_file)

label_embeddings_file = config["file_paths"]["label_embeddings_file"]
gensim_model = load_key_vector_model_from_file(label_embeddings_file)

tbox_pickle_file = config["file_paths"]["tbox_pickle_file"]
labels_pickle_file = config["file_paths"]["labels_pickle_file"]


def get_pickle_object(file):
    """This function can be used to load the labels.pkl
    and tbox.pkl files.
    """
    with open(file, "rb") as f:
        return pickle.load(f)


labels = get_pickle_object(labels_pickle_file)
tbox = get_pickle_object(tbox_pickle_file)

encoder_model = get_sentence_transformer_model(
    model_name=config["model_names"]["encoder_model"]
)
logging.info("Encoder model loaded")

genre_tokenizer = AutoTokenizer.from_pretrained(
    config["model_names"]["genre_tokenizer"]
)
genre_model = AutoModelForSeq2SeqLM.from_pretrained(
    config["model_names"]["genre_model"]
).eval()
logging.info("GENRE model loaded")


def get_triple_from_triple(sub, relation, obj):
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
        sub, genre_model, genre_tokenizer
    )[0]
    subject_entity = "https://dbpedia.org/resource/" + "_".join(subject_entity.split())

    object_entity = EL_GENRE(
        obj, genre_model, genre_tokenizer
    )[0]
    object_entity = "https://dbpedia.org/resource/" + "_".join(object_entity.split())

    predicates, label, score = (
        ontosim_search(relation, gensim_model, encoder_model, tbox).iloc[0].values
    )

    predicate = None
    for p in predicates:
        if p.split("/")[-1][0].islower():
            predicate = p
            break
    # can also return this - (subject_entity, (predicate, score), object_entity)
    return (subject_entity, predicate, object_entity)
