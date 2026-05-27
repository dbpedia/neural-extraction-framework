import pandas as pd
from gensim.models import KeyedVectors
from GSoC24.RelationExtraction.encoding_utils import to_uri


def ontosim_search(term, key_vector_model, encoder_model, tbox):
    """Given a natural language relation, return top 5 most
    appropriate dbpedia predicates for it.

    Args:
        term : The relation, in natural language.
        key_vector_model : The Gensim KeyedVectors model.
        It is like an embedding table for words.
        encoder_model : The encoder model used to encode text to vectors.
        tbox : A dictionary of labels (of predicates) with a set of
        predicates with that label as values.

    Returns:
    A dataframe.
    """
    result = key_vector_model.most_similar(
        positive=encoder_model.encode([term]), topn=5
    )
    out = []
    for label, score in result:
        out.append({"label": label.replace("_", " "), "score": score})
    df = pd.DataFrame(out)
    df.insert(0, "URIs", df["label"].map(lambda x: to_uri(x, tbox=tbox)))
    return df


def load_key_vector_model_from_file(filepath):
    model = KeyedVectors.load_word2vec_format(filepath, binary=False)
    return model
