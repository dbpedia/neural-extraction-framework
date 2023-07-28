from GSoC23.RelationExtraction.rebel import extract_relations_rebel
from GSoC23.RelationExtraction.re_utils import get_triple_from_triple

def get_triples_from_sentence_using_rebel(sentence, rebel_model, rebel_tokenizer):
    sent_triples =  extract_relations_rebel(rebel_model, rebel_tokenizer, text=sentence)
    triples = []
    
    for i in range(len(sent_triples)):
        subject, relation, objct, score = sent_triples.iloc[i].values
        triple = get_triple_from_triple(subject, relation, objct, sentence)
        triples.append(
        {
            "subject":subject,
            "predicate":relation,
            "object":objct,
            "subject_URI": triple[0],
            "predicate_URI":triple[1],
            "object_URI":triple[2],
            "relation_confidence_score":score,
        }
        )
    return triples