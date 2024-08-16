import argparse
from glob import glob
import os
import stanza
import torch
from coref import CorefModel
from coref.tokenizer_customization import *
from coref.mcoref import resolve_pronouns_hindi
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sent_features import get_pos_ner_tags
from indIE import get_triples
from llm_triplets import get_llm_triplets
from llm_coreference import get_llm_coref
import pickle
from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--do_coref", action="store_true", required=False)
    parser.add_argument("--do_rel", action="store_true", required=False)
    parser.add_argument("--do_el", action="store_true", required=False)
    parser.add_argument("--verbose", action="store_true", required=False)
    return parser.parse_args()


def format_mentions(sentence, mentions):
    sorted_mentions = sorted(mentions, key=len, reverse=True)
    escaped_mentions = [re.escape(mention) for mention in sorted_mentions]
    pattern = "|".join(escaped_mentions)

    def replace_mention(match):
        return f"[{match.group(0)}](#)"

    formatted_sentence = re.sub(pattern, replace_mention, sentence)
    return formatted_sentence


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    coref_model = CorefModel("models/coref_model/config.toml", "xlmr")
    coref_model.config.device = device
    coref_model.load_weights(
        path="models/coref_model/xlmr_multi_plus_hi2.pt",
        map_location=device,
        ignore={
            "bert_optimizer",
            "general_optimizer",
            "bert_scheduler",
            "general_scheduler",
        },
    )
    coref_model.training = False
    with open("input/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
        trie = pickle.load(f)
    el_model = mGENRE.from_pretrained(
        "models/fairseq_multilingual_entity_disambiguation"
    ).eval()
    nlp = stanza.Pipeline(lang="hi", processors="tokenize,pos")
    # tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicNER")
    # ner_model = AutoModelForTokenClassification.from_pretrained(
    #     "ai4bharat/IndicNER"
    # ).eval()

    wiki_pages = glob(os.path.join(args.input_dir, "*.txt"))

    for id, articles in enumerate(wiki_pages):
        with open(articles, "r") as f:
            wiki_article = f.read()
            if args.do_coref:
                _, out_sentences = resolve_pronouns_hindi(
                    [wiki_article], coref_model, nlp
                )
                wiki_article = out_sentences[0]
                if args.verbose:
                    print(wiki_article)
                #############################
                # NER/POS/Mention Detection #
                #############################
                # pos_tags, mentions = get_pos_ner_tags(
                #     wiki_article, nlp, tokenizer, ner_model
                # )
                # if args.verbose:
                #     print(*pos_tags, sep="\n")
                #     print(*mentions, sep="\n")

                ##########################
                # Coreference Resolution #
                ##########################
                # formated_text = format_mentions(wiki_article, mentions)
                # coref_resolved_text, ttaken_coref = get_llm_coref(formated_text)
                # if args.verbose:
                #     print(coref_resolved_text, ttaken_coref)

            if args.do_rel:
                #######################
                # Relation extrcation #
                #######################
                doc: stanza.Document = nlp(wiki_article)
                for sentence in doc.sentences:
                    sent = sentence.text
                    print(sent)
                    exts_rule, ttaken_rule = get_triples(sent)
                    # exts_llm, ttaken_llm = get_llm_triplets(sent)
                    if args.verbose:
                        print("Rule based triplets in the article: ", ttaken_rule)
                        # print("LLM based triplets in the article: ", ttaken_llm)
                        print(*exts_rule, sep="\n")
                        # print(*exts_llm, sep="\n")
                        print("\n")

                    el_sents = {}
                    for relation in exts_rule[0]:
                        s, p, o = relation
                        el_sents[s] = f"[START] {s} [END] {p} {o}"
                        el_sents[o] = f"{s} {p} [START] {o} [END]"

                    if args.do_el:
                        ####################
                        # Entity Linking #
                        ####################
                        ans = el_model.sample(
                            list(el_sents.values()),
                            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                e
                                for e in trie.get(sent.tolist())
                                if e < len(el_model.task.target_dictionary)
                            ],
                        )
                        # print(ans)
                        el_maps = {}
                        for surface_l, annot in zip(el_sents.keys(), ans):
                            annot = sorted(
                                annot, key=lambda x: x["score"], reverse=True
                            )
                            el_maps[surface_l] = annot[0]["text"].split(" >> ")[0]
                        new_exts_rule = []
                        for relation in exts_rule[0]:
                            s, p, o = relation
                            new_exts_rule.append(
                                (
                                    el_maps[s],
                                    p,
                                    el_maps[o],
                                )
                            )
                        exts_rule[0] = new_exts_rule
                    print(exts_rule)
