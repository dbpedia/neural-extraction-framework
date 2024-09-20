from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def get_sentence_transformer_model(model_name):
    model = SentenceTransformer(model_name_or_path=model_name)
    return model


def get_embeddings(labels, sent_tran_model):
    embeddings = sent_tran_model.encode(labels)
    return embeddings
