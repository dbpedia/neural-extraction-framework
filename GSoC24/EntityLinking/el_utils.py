import re
import pandas as pd


def annotate_sentence(sentence, mention):
    match = re.search(mention.lower(), sentence.lower())
    start, end = match.span()
    sentence = (
        sentence[:start]
        + " [START_ENT] "
        + sentence[start:end]
        + " [END_ENT] "
        + sentence[end:]
    )
    return sentence


def get_majority_vote(candidate_list):
    """Perform majority vote over a list of candidate entities
    based on frequency.

    Initially, we are using frequency to do the voting.
    But later on, we can also do a weighted voting,
    by weighing each candidate based on the method which returned it,
    thereby giving more importance to more accurate methods.

    Args:
        candidate_list : List of candidate entities/resources.

    Returns:
        Returns a resource/entity URI
    """
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
    start = entity["start"]
    end = entity["end"]

    # We need to add [START_ENT] tag before the entity and [END_ENT] tag
    # after the entity we want to link and disambiguate
    result_sentence = sentence[:start]
    result_sentence += "[START_ENT] "
    result_sentence += sentence[start:end]
    result_sentence += " [END_ENT] "
    result_sentence += sentence[end:]
    result_sentence = result_sentence.strip()
    return result_sentence


def calculate_redirect(source, client):
    result = client.get(source)
    if result is None:
        return source if type(source) is str else source.decode("utf-8")
    return calculate_redirect(result, client)


def query(surface_form, client):
    raw = client.hgetall(surface_form)
    if len(raw) == 0:
        return pd.DataFrame(columns=["entity", "support", "score"])

    out = []
    for label, score in raw.items():
        out.append({"entity": label.decode("utf-8"), "support": int(score)})
    df_all = pd.DataFrame(out)
    df_all["score"] = df_all["support"] / df_all["support"].max()

    return df_all.sort_values(by="score", ascending=False).reset_index(drop=True)


def lookup(term, top_k=5, thr=0.01, redis_client_forms=None, redis_client_redir=None):
    df_temp = query(term, redis_client_forms)
    #     display(df_temp)
    df_temp["entity"] = df_temp["entity"].apply(
        lambda x: calculate_redirect(x, redis_client_redir)
    )
    if len(df_temp) == 0:
        return pd.DataFrame(columns=["entity", "support", "score"])
    df_final = df_temp.groupby("entity").sum()[["support"]]
    df_final["score"] = df_final["support"] / df_final["support"].max()
    return df_final[df_final["score"] >= thr].sort_values(by="score", ascending=False)[
        :top_k
    ]
