import spacy
from fastcoref import spacy_component

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
    "fastcoref",
    config={
        'model_architecture': 'LingMessCoref', 
        'model_path': 'biu-nlp/lingmess-coref', 
        'device': 'cpu'}
    )

def get_coref_resolved_text(text):
    global nlp
    doc = nlp(
    text,
    component_cfg={"fastcoref": {'resolve_text': True}}
    )

    return doc._.resolved_text