import requests
import json
import time
import re


# def get_prompt(text):
#     template = {"formatted_text": ""}
#     prompt = f"""Annotate all entity mentions in the following sentence using coreference clusters. Use Markdown tags to indicate clusters in the output, with the following format [mention](#cluster_number). Return the response in json in the following format.
# {json.dumps(template, indent=4)}
# Some examples of valid coreference clusters are:
# [Carl](#1) thrust [his](#1) hands into [his](#1) pockets, lowered [his](#1) head, and darted up [the street](#2) against the north wind.
# Here's the text to analyze:
# {text}
# """
#     return prompt


def get_prompt(text):
    template = {"formatted_text": ""}
    prompt = f"""Rewrite the below sentence as it is with the correct coreference resolution. Give the response in JSON in the following format.
{json.dumps(template, indent=4)}
Here's the text to analyze:
{text}
"""
    return prompt


def extract_coref_chains(formatted_text):
    pattern = r"\[([^\]]+)\]\(#([^)]+)\)"
    matches = re.findall(pattern, formatted_text)

    coref_chains = {}

    for mention, symbol in matches:
        if symbol not in coref_chains:
            coref_chains[symbol] = []
        coref_chains[symbol].append(mention)
    return coref_chains


def resolve_coreferences(sentence, clusters):
    resolved_sentence = sentence
    for cluster in clusters:
        canonical_name = cluster[0]
        for mention in cluster[1:]:
            resolved_sentence = resolved_sentence.replace(mention, canonical_name)
    return resolved_sentence


# sentence = "John went to the store. He bought some milk."
# clusters = [["John", "He"]]
# resolved_sentence = resolve_coreferences(sentence, clusters)
# print(resolved_sentence)  # Output: "John went to the store. John bought some milk."


def get_llm_coref(user_text):
    model = "phi3-mini-128k-q8:latest"
    prompt = get_prompt(user_text)
    data = {
        "prompt": prompt,
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate", json=data, stream=False
    )
    ttaken = round(time.time() - start_time, 3)
    json_data = json.loads(response.text)
    result = json.loads(json_data["response"].strip())
    return result["formatted_text"], ttaken


if __name__ == "__main__":
    user_text = """[गुलाम मुस्तफ़ा खान](#) को [भारत सरकार](#) द्वारा सन 2006 में [कला](#) के [क्षेत्र](#) में पद्म भूषण से सम्मानित किया गया था। ये [महाराष्ट्र](#) से हैं।"""
    formatted_text, ttaken = get_llm_coref(user_text)
    print(formatted_text)
    coref_chains = extract_coref_chains(formatted_text)
    print(coref_chains)
    clusters = list(coref_chains.values())
    resolved_text = resolve_coreferences(formatted_text, clusters)
    print(resolved_text)
