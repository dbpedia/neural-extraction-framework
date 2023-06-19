import spacy
from transformers import pipeline

stop = [
    "the", "is", "in",
    "for", "where", "when",
    "to", "at"
]

def spacy_ner(language, text):
    result = language(text)
    entities = []
    for ent in result.ents:
        d = {}
        d['entity_group']=ent.label_

        # check for stop-words
        for st in stop:
            if st in ent.text:
                ent.text = ent.text.replace(st,"")
                
        d['word']=ent.text
        d['start']=ent.start_char
        d['end']=ent.end_char
        entities.append(d)
    return entities

def hf_transformer_ner(model, tokenizer, text):
    text = text.lower()
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    return nlp(text)