
def get_majority_vote(candidate_list):
    candidate_entity = max(set(candidate_list), key=candidate_list.count)
    return candidate_entity

def convert_sentence_for_genre_model(sentence, entity):
    """A function to provide annotated form of the sentence.
    The sentence we provide here should first be passed through
    the NER system to get all entities in it. Then, with each entity
    individually, we pass it to this function. And we pass the result of 
    this function to the GENRE entity disambiguation model.

    Args:
        sentence (_type_): A given sentence
        entity (_type_): A dictionary containing the entity text
        entity span and other such info
    """
    start = entity['start']
    end = entity['end']

    # We need to add [START_ENT] tag before the entity and [END_ENT] tag 
    # after the entity we want to link and disambiguate
    result_sentence = sentence[:start]
    result_sentence+= '[START_ENT] '
    result_sentence+= sentence[start:end]
    result_sentence+= ' [END_ENT] '
    result_sentence+= sentence[end:]
    result_sentence = result_sentence.strip()
    return result_sentence