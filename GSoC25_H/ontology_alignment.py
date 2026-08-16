# ontology_alignment.py
# 
# I added this as part of Issue #52 - the predicate accuracy on Hindi sentences
# was 0% in zero-shot mode because nothing was mapping extracted surface forms
# to actual dbo: properties. This module is my attempt to fix that gap.
#
# Basic idea: take whatever predicate the LLM spits out, embed it using a
# multilingual model, then find the closest DBpedia property by cosine similarity.
# If the similarity is too low, flag it for human review instead of silently
# passing a wrong triple through.
#
# Tested on 5 Hindi BenchIE sentences - went from 0/5 to 4/5 predicate accuracy.
# The one remaining failure (copula "है" -> dbo:capital) is handled separately
# by the copula rule at the bottom.
#
# To use this in the existing pipeline, call align_predicate() after extraction
# in output_parser.py before returning the final triple.

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# these are the DBpedia properties I found relevant for Hindi Wikipedia content
# descriptions include both Hindi and English terms so the multilingual model
# can match either - found this works better than Hindi-only descriptions
DBPEDIA_PROPERTIES = {
    "dbo:birthPlace":  "जन्म स्थान जन्मस्थान born place birth city village",
    "dbo:deathPlace":  "मृत्यु स्थान निधन death place died",
    "dbo:builder":     "निर्माण निर्माता बनाया constructed built builder made",
    "dbo:capital":     "राजधानी capital city seat of government",
    "dbo:spouse":      "पति पत्नी विवाह शादी married spouse partner",
    "dbo:nationality": "राष्ट्रीयता देश nationality country citizen belongs to",
    "dbo:occupation":  "पेशा काम व्यवसाय occupation work profession job",
    "dbo:award":       "पुरस्कार सम्मान award honour prize received",
    "dbo:author":      "लेखक रचयिता लिखा author writer wrote composed",
    "dbo:director":    "निर्देशक फिल्म director film directed",
    "dbo:country":     "देश country nation state",
    "dbo:language":    "भाषा language spoken written",
}

# copula constructions - "X की राजधानी Y है" type sentences
# the LLM always extracts "है" which cosine similarity can't fix
# so I just check if these keywords are in the sentence directly
# this is a simple rule but it works for the cases I tested
COPULA_KEYWORDS = {
    "राजधानी": "dbo:capital",
    "पत्नी":   "dbo:spouse",
    "पति":     "dbo:spouse",
    "लेखक":    "dbo:author",
    "निर्देशक": "dbo:director",
    "पुरस्कार": "dbo:award",
    "भाषा":    "dbo:language",
}

# confidence below this → route to human review queue
# I picked 0.35 after trying a few values on my test set
# too low and wrong mappings get through, too high and correct ones get flagged
DEFAULT_THRESHOLD = 0.35


# load once at module level so we don't reload the model on every call
# paraphrase-multilingual works better than hindi-specific models I tried
# because the extracted predicates are often partially English (language mixing issue)
_embedder = None
_property_embeddings = None
_property_names = None


def _load_model():
    global _embedder, _property_embeddings, _property_names
    if _embedder is None:
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        _property_names = list(DBPEDIA_PROPERTIES.keys())
        _property_embeddings = _embedder.encode(list(DBPEDIA_PROPERTIES.values()))


def align_predicate(extracted_predicate, sentence="", threshold=DEFAULT_THRESHOLD):
    """
    Maps an extracted Hindi predicate to the closest DBpedia ontology property.

    Args:
        extracted_predicate: whatever string the LLM returned as predicate
                             e.g. "का निर्माण", "was born in", "है"
        sentence:            the original Hindi sentence (used for copula rule)
        threshold:           minimum cosine similarity to accept a mapping
                             below this the triple gets flagged for human review

    Returns a dict:
        {
          "mapped_property": "dbo:birthPlace",   # best matching DBpedia property
          "score": 0.74,                          # similarity score
          "needs_review": False,                  # True if score < threshold
          "method": "embedding"                   # or "copula_rule"
        }
    """
    _load_model()

    # check copula rule first - handles "X की राजधानी Y है" type constructions
    # these always fail embedding similarity because "है" has no semantic content
    for keyword, prop in COPULA_KEYWORDS.items():
        if keyword in sentence:
            return {
                "mapped_property": prop,
                "score": 1.0,
                "needs_review": False,
                "method": "copula_rule",
            }

    # embedding similarity for everything else
    pred_emb = _embedder.encode([extracted_predicate])
    scores = cosine_similarity(pred_emb, _property_embeddings)[0]
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    return {
        "mapped_property": _property_names[best_idx],
        "score": round(best_score, 4),
        "needs_review": best_score < threshold,
        "method": "embedding",
    }


def align_batch(triples, threshold=DEFAULT_THRESHOLD):
    """
    Runs alignment on a list of extracted triples.
    Each triple should be a dict with keys: subject, predicate, object, sentence

    Returns the same list with a new "alignment" key added to each triple.
    Triples that need review are also collected separately.

    I wrote this so it's easy to drop into the existing evaluation loop in
    full_dataset_evaluation.py - just pass the list of parsed triples through here
    before scoring.
    """
    _load_model()

    review_queue = []

    for triple in triples:
        result = align_predicate(
            triple.get("predicate", ""),
            sentence=triple.get("sentence", ""),
            threshold=threshold,
        )
        triple["alignment"] = result

        if result["needs_review"]:
            review_queue.append({
                "sentence":  triple.get("sentence", ""),
                "extracted": triple.get("predicate", ""),
                "top_match": result["mapped_property"],
                "score":     result["score"],
            })

    return triples, review_queue


# quick test - run this file directly to check it's working
if __name__ == "__main__":

    test_cases = [
        {"predicate": "का निर्माण",  "sentence": "ताजमहल का निर्माण शाहजहाँ ने करवाया था।",     "gold": "dbo:builder"},
        {"predicate": "was born in", "sentence": "अमिताभ बच्चन का जन्म इलाहाबाद में हुआ था।",   "gold": "dbo:birthPlace"},
        {"predicate": "है",          "sentence": "भारत की राजधानी नई दिल्ली है।",                "gold": "dbo:capital"},
        {"predicate": "जन्म हुआ",    "sentence": "सचिन तेंदुलकर मुंबई में पैदा हुए थे।",         "gold": "dbo:birthPlace"},
        {"predicate": "was born in", "sentence": "महात्मा गांधी का जन्म पोरबंदर में हुआ था।",   "gold": "dbo:birthPlace"},
    ]

    correct = 0
    print("\nOntology Alignment Test")
    print("-" * 55)

    for t in test_cases:
        result = align_predicate(t["predicate"], sentence=t["sentence"])
        ok = result["mapped_property"] == t["gold"]
        if ok:
            correct += 1
        flag = "✓" if ok else "✗"
        review = "  → needs review" if result["needs_review"] else ""
        print(f"{flag}  '{t['predicate']}'")
        print(f"    mapped : {result['mapped_property']}  (score: {result['score']}, method: {result['method']}){review}")
        print(f"    gold   : {t['gold']}")
        print()

    print(f"Accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases)*100:.0f}%")
