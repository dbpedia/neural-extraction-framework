import json
import pickle
import time

from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie

from el_normalize import normalize_to_dbpedia_title_from_genre_text
from predicate_linking import link_predicate


def load_el():
    with open("models/EL_model/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
        trie = pickle.load(f)
    el_model = mGENRE.from_pretrained(
        "models/EL_model/fairseq_multilingual_entity_disambiguation"
    ).eval()
    return trie, el_model


def run_entity_linking(triples):
    trie, el_model = load_el()
    el_sents = {}
    for (s, p, o) in triples:
        el_sents[s] = f"[START] {s} [END] {p} {o}"
        el_sents[o] = f"{s} {p} [START] {o} [END]"

    ans = el_model.sample(
        list(el_sents.values()),
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist()) if e < len(el_model.task.target_dictionary)
        ],
        marginalize=True,
    )

    el_maps = {}
    for surface_l, annot in zip(el_sents.keys(), ans):
        annot = sorted(annot, key=lambda x: x["score"], reverse=True)
        top_text = annot[0]["text"]  # e.g., "दिल्ली >> hi"
        en_title, _dbr = normalize_to_dbpedia_title_from_genre_text(top_text)
        print(f" mapped surface title: {surface_l} -> dbpedia title: {en_title}")
        el_maps[surface_l] = en_title if en_title else top_text.split(" >> ")[0]
    return el_maps


def main():
    triples = [
        ("सचिन तेंदुलकर", "जन्म स्थान", "मुंबई"),
        ("माइक्रोसॉफ़्ट", "द्वारा स्थापित", "बिल गेट्स"),
        ("भारत", "राजधानी", "नई दिल्ली"),
    ]

    print(" Testing Entity Linking (mGENRE → English DBpedia titles)")
    t0 = time.time()
    el_maps = run_entity_linking(triples)
    print(f"EL done in {time.time() - t0:.2f}s")
    for k, v in el_maps.items():
        print(f"  {k} -> {v}")

    print("\n Testing Predicate Linking (DBpedia)")
    for (s, p, o) in triples:
        s_norm = el_maps.get(s, s)
        o_norm = el_maps.get(o, o)
        print(f"\nInput triple: ({s_norm}, {p}, {o_norm})")
        res = link_predicate(p, s_norm, o_norm, lang="hi")
        print("Result:")
        print(f"  property_uri: {res.get('property_uri')}")
        print(f"  property_label(en): {res.get('property_label',{}).get('en')}")
        print(f"  score: {res.get('score'):.3f}")
        print(f"  direction: {res.get('direction')}")
        print(f"  evidence: {json.dumps(res.get('evidence'))}")


if __name__ == "__main__":
    main()


