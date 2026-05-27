# CUDA_VISIBLE_DEVICES=1 python hi_predict.py --weight data/xlmr2.pt --output_file out_xlmr_2.jsonlines --experiment xlmr
import argparse

import jsonlines
import torch, time
from tqdm import tqdm

from coref import CorefModel
from coref.tokenizer_customization import *

import stanza
import re

# stanza_path = importlib.util.find_spec("stanza").submodule_search_locations[0]
# https://stackoverflow.com/questions/269795/how-do-i-find-the-location-of-python-module-sources
# https://stackoverflow.com/questions/35288021/what-is-the-equivalent-of-imp-find-module-in-importlib
# stanza_version = stanza.__version__

# if not os.path.exists(stanza_path + "/" + stanza_version + "/"):
#     os.makedirs(stanza_path + "/" + stanza_version + "/")


def build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(model.config.bert_model, lambda _: True)
    token_map = TOKENIZER_MAPS.get(model.config.bert_model, {})

    word2subword = []
    subwords = []
    word_id = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (
            token_map[word] if word in token_map else model.tokenizer.tokenize(word)
        )
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id

    doc["head2span"] = []
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []

    return doc


def resolve_pronouns_hindi(text_list, model, hinlp):
    # json_list = []
    # # for sent in text_list:
    # #     json_object = {"document_id": "fu", "cased_words": [], "sent_id":[], "pos":[]}
    # #     doc = hinlp(sent)
    # #     snum = -1
    # #     for s in doc.sentences:
    # #         snum+=1
    # #         for w in s.words:
    # #             json_object['cased_words'].append(w.text)
    # #             json_object['pos'].append(w.upos)
    # #             json_object['sent_id'].append(snum)
    # #     json_list.append(json_object)
    # text = '. '.join(text_list)
    # doc = hinlp(text)
    # snum = -1
    # json_object = {"document_id": "fu", "cased_words": [], "sent_id":[], "pos":[]}
    # for s in doc.sentences:
    #     snum+=1
    #     for w in s.words:
    #         json_object['cased_words'].append(w.text)
    #         json_object['pos'].append(w.upos)
    #         json_object['sent_id'].append(snum)
    # json_list.append(json_object)

    # docs = [build_doc(json_object, model) for json_object in json_list]

    # with torch.no_grad():
    #     for doc in tqdm(docs, unit="docs", disable=True):
    #         a = time.time()
    #         result = model.run(doc)
    #         doc["span_clusters"] = result.span_clusters
    #         doc["word_clusters"] = result.word_clusters

    #         for key in ("word2subword", "subwords", "word_id", "head2span","speaker"):
    #             del doc[key]

    #         doc['cased_words2'] = [str(i)+'_'+x for i,x in enumerate(doc['cased_words'])]
    #         doc['pos2'] = [str(i)+'_'+x for i,x in enumerate(doc['pos'])]

    docs = resolve_coreferences(text_list, model, hinlp)
    # print(docs)

    text_list = []
    for doc in docs:
        replacements_needed = []
        for chain in doc["span_clusters"]:
            # print('chain', chain)
            for i in range(len(chain) - 1, 0, -1):
                replacee = chain[i]  # the one that should be replaced i.e. replaceee
                if "PRON" in doc["pos"][replacee[0] : replacee[1]]:
                    candidate_ants = []
                    for j in range(i - 1, -1, -1):
                        replacer = chain[j]  # replacer
                        if "PROPN" in doc["pos"][replacer[0] : replacer[1]]:
                            # replacements_needed.append([replacee, replacer])
                            candidate_ants.append(replacer)
                    if not candidate_ants:
                        for j in range(i - 1, -1, -1):
                            replacer = chain[j]
                            if "NOUN" in doc["pos"][replacer[0] : replacer[1]]:
                                replacements_needed.append([replacee, replacer])
                                break
                    else:
                        span_len = [c[1] - c[0] for c in candidate_ants]
                        replacer = candidate_ants[span_len.index(min(span_len))]
                        replacements_needed.append([replacee, replacer])

                else:
                    # print('not pronoun', doc['pos'][replacee[0]:replacee[1]], replacee)
                    pass
        # print(doc)
        # print(replacements_needed)
        # doc['cased_words'] = [ '_'.join(x.split('_')[1:]) for x in doc['cased_words']]
        # if len(replacements_needed) > 0:
        #     print(re.sub(r'\s+',' ',' '.join(doc['cased_words'])).strip())
        for rn in replacements_needed[::-1]:
            replacee, replacer = rn[0], rn[1]
            replacee_string = re.sub(
                r"\s+", " ", " ".join(doc["cased_words"][replacee[0] : replacee[1]])
            ).strip()
            replacer_string = re.sub(
                r"\s+", " ", " ".join(doc["cased_words"][replacer[0] : replacer[1]])
            ).strip()
            replacer_string = add_karak(remove_karak(replacer_string), replacee_string)
            replacer_string = " ".join(
                [x for x in replacer_string.split() if x != "एक"]
            )
            if replacee[0] >= replacer[1]:  # there should be no overlap
                doc["cased_words"] = (
                    doc["cased_words"][: replacee[0]]
                    + [replacer_string]
                    + [""] * (replacee[1] - replacee[0] - 1)
                    + doc["cased_words"][replacee[1] :]
                )
        text_list.append(re.sub(r"\s+", " ", " ".join(doc["cased_words"])).strip())
        # print(text_list[-1])
        # input('wait')
        # if len(replacements_needed) > 0:
        #     print(re.sub(r'\s+',' ',' '.join(doc['cased_words'])).strip())
        #     print(doc)
        #     input('wait')

    return docs, text_list


def resolve_coreferences(text_list, model, hinlp):
    json_list = []
    # for sent in text_list:
    #     json_object = {"document_id": "fu", "cased_words": [], "sent_id":[], "pos":[]}
    #     doc = hinlp(sent)
    #     snum = -1
    #     for s in doc.sentences:
    #         snum+=1
    #         for w in s.words:
    #             json_object['cased_words'].append(w.text)
    #             json_object['pos'].append(w.upos)
    #             json_object['sent_id'].append(snum)
    #     json_list.append(json_object)
    text = ".".join(text_list)
    doc = hinlp(text)
    snum = -1
    json_object = {"document_id": "mu", "cased_words": [], "sent_id": [], "pos": []}
    for s in doc.sentences:
        snum += 1
        for w in s.words:
            json_object["cased_words"].append(w.text)
            json_object["pos"].append(w.upos)
            json_object["sent_id"].append(snum)
    json_list.append(json_object)

    docs = [build_doc(json_object, model) for json_object in json_list]

    with torch.no_grad():
        for doc in tqdm(docs, unit="docs", disable=True):
            a = time.time()
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span", "speaker"):
                del doc[key]

            doc["cased_words2"] = [
                str(i) + "_" + x for i, x in enumerate(doc["cased_words"])
            ]
            doc["pos2"] = [str(i) + "_" + x for i, x in enumerate(doc["pos"])]
    return docs


def remove_karak(text):
    text = text.strip().split()
    for k in ["ने", "को", "से", "का", "की", "के", "में", "मे", "पर", "केलिए"]:
        if text[-1] == k:
            text[-1] = ""
    if text[-1] == "लिए" and len(text) > 1 and text[-2] == "के":
        text[-1], text[-2] = "", ""
    if text[0] == "हे":
        text[0] = ""
    text = (" ".join(text)).strip()
    return text


def add_karak(text1, replacee_string):
    for k in ["ने", "को", "से", "का", "की", "के", "में", "मे", "पर", "केलिए"]:
        if (
            re.search(k + r"$", replacee_string.strip())
            and not (re.search(r"^(आ|अ)प", replacee_string.strip()))
            and replacee_string != "इसे"
        ):
            return text1 + " " + k
    if replacee_string == "उन्हें" or replacee_string == "उन्हे" or replacee_string == "इसे":
        return text1 + " को"
    if re.search(
        r"^आप", replacee_string
    ):  # sentences written in second person are generally not resolved correctly
        return replacee_string
    return text1


if __name__ == "__main__":
    # span clusters means these word-spans are clustered according to their predicted coreference chains
    # word_clusters means those head_words which belong to one chain, similar to span_clusters but at word level
    # remember both are 0 indexed
    # and in span clusters [a,b] means a is included whereas b is not included

    # check if gpu is available
    device = None
    if torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda")
    else:
        use_gpu = False
        device = torch.device("cpu")

    try:
        hinlp = stanza.Pipeline(
            "hi",
            # dir=stanza_path + "/" + stanza_version + "/",
            # download_method=2,
            use_gpu=use_gpu,
        )
    except:
        stanza.download(
            "hi", dir=stanza_path + "/" + stanza_version + "/"
        )  # model_dir or dir depending on the stanza version
        hinlp = stanza.Pipeline(
            "hi",
            dir=stanza_path + "/" + stanza_version + "/",
            download_method=2,
            use_gpu=use_gpu,
        )
    print("Stanza loaded")

    model = CorefModel("coref/config.toml", "xlmr")
    model.config.device = device
    model.load_weights(
        path="model/xlmr_multi_plus_hi2.pt",
        map_location=device,
        ignore={
            "bert_optimizer",
            "general_optimizer",
            "bert_scheduler",
            "general_scheduler",
        },
    )
    model.training = False

    raw_sentences = [
        "हमने कोहरे में एक आदमी को भागते देखा।  उसने काले रंग की शाल पहनी थी।  उसका कद लम्बा था। ",
        "मार्च 2001 में एक बार फिर अमरीका के अट्ठाइसवें रक्षा सचिव के रूप में उनकी वापसी हुई।",
    ]
    # raw_sentences = [
    #     "Ramchandra was a good king. His brother's name was Laxman.",
    #     "رام چندر جی اچھے بادشاہ تھے۔ اس کے بھائی کا نام لکشمن تھا۔",
    #     "ராமச்சந்திரா ஒரு நல்ல அரசர். அவரது சகோதரரின் பெயர் லக்ஷ்மன்.",
    #     "రామచంద్ర మంచి రాజు. అతని సోదరుడి పేరు లక్ష్మణ్.",
    #     "രാമചന്ദ്രജി ഒരു നല്ല രാജാവായിരുന്നു. സഹോദരന്റെ പേര് ലക്ഷ്മണൻ.",
    #     "রামচন্দ্র জি একজন ভালো রাজা ছিলেন। তার ভাইয়ের নাম ছিল লক্ষ্মণ।",
    #     "રામચંદ્રજી સારા રાજા હતા. તેમના ભાઈનું નામ લક્ષ્મણ હતું.",
    #     "ৰামচন্দ্ৰ জী এজন ভাল ৰজা আছিল। ভায়েকৰ নাম লক্ষ্মণ।",
    #     "ਰਾਮਚੰਦਰ ਜੀ ਚੰਗੇ ਰਾਜੇ ਸਨ। ਉਸ ਦੇ ਭਰਾ ਦਾ ਨਾਂ ਲਕਸ਼ਮਣ ਸੀ।",
    #     "रामचंद्र जी इक अच्छे राजा थे। उंदे भ्राऽ दा नां लक्ष्मण हा।",
    # ]
    # raw_sentences = [
    #     "हमने कोहरे में एक आदमी को भागते देखा।  उसने काले रंग की शाल पहनी थी।  उसका कद लम्बा था।"
    # ]

    docs, out_sentences = resolve_pronouns_hindi(raw_sentences, model, hinlp)
    out_sentences = out_sentences[0].split(".")
    assert len(raw_sentences) == len(out_sentences)
    for x, y in zip(raw_sentences, out_sentences):
        print("In ", x)
        print("Out", y, end="\n\n")


# In  हमने कोहरे में एक आदमी को भागते देखा।  उसने काले रंग की शाल पहनी थी।  उसका कद लम्बा था।
# Out हमने कोहरे में एक आदमी को भागते देखा । आदमी ने काले रंग की शाल पहनी थी । आदमी का कद लम्बा था ।
