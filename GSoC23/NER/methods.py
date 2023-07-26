import spacy
from transformers import pipeline

stop = [
    "the", "is", "in",
    "for", "where", "when",
    "to", "at"
]

def spacy_ner(language: spacy.lang.en.English, text: str):
    """Performs NER using the spacy model(language) specified. 

    Args:
        language (spacy.lang.en.English): The spacy English model to use.
        text (str): The text on which to perform the NER.

    Returns:
        List[Dict]: A list of entities each represented by a dictionary
        containing the entity-mention/word and its start and end positions.
    """
    result = language(text)
    entities = []
    for ent in result.ents:
        d = {}
        d['entity_group']=ent.label_

        # check for stop-words
        text = ent.text
        for st in stop:
            if st in ent.text:
                # print(type(ent.text))
                # ent.text = ent.text.replace(st,"")
                text = text.replace(st, "")
                
        d['word']=text
        d['start']=ent.start_char
        d['end']=ent.end_char
        entities.append(d)
    return entities

def hf_transformer_ner(model, tokenizer, text):
    """Given a text, model and its tokenizer(from huggingface), perform
    NER using huggingface ner pipeline using that model and tokenizer.

    Args:
        model : The HF model
        tokenizer : Respective tokenizer for the model
        text : The text on which to perform NER.

    Returns:
        List[Dict]: A list of entities each represented by a dictionary
        containing the entity-mention/word and its start and end positions.
    """
    text = text.lower()
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp(text)