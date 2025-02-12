from GSoC24.RelationExtraction.re_utils import get_triple_from_triple
from GSoC24.RelationExtraction.llm_utils import response_to_triples


def get_triples_from_sentence(user_prompt, model):
    sent_triples = response_to_triples(user_prompt, model)
    triples = []

    for i in range(len(sent_triples)):
        subject, relation, objct = sent_triples.iloc[i].values
        triple = get_triple_from_triple(subject, relation, objct)
        triples.append(
            {
                "subject": subject,
                "predicate": relation,
                "object": objct ,
                "subject_URI": triple[0],
                "predicate_URI": triple[1],
                "object_URI": triple[2]
            }
        )
    return triples
