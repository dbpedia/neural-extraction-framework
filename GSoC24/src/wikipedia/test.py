from src.reader import get_text_of_wiki_page
from fastcoref import spacy_component
import stanza
import re

# from codeswitch.codeswitch import NER


# stanza initialization
# stanza.download("hi")
nlp = stanza.Pipeline(lang="hi", processors="tokenize,pos")
# ner = NER("hin-eng")


article = "ब्रिटिश_कोलम्बिया"
wiki_article = get_text_of_wiki_page(article)

# Tokenization
doc = nlp(wiki_article)
# doc = nlp("मुझसे पूछा िक 'आप िकसका नाम लेना चाहेंगी', तब मैंने अपना नम ¶लय।")
# result = ner.tag("मुझसे पूछा िक 'आप िकसका नाम लेना चाहेंगी', तब मैंने अपना नम ¶लय।")
#     print(f"====== Sentence {i+1} tokens =======")
#     print(
#         *[f"id: {token.id}\ttext: {token.text}" for token in sentence.tokens], sep="\n"
#     )
#     if i == 5:
#         break

# Coreference resolution

# print(
#     *[
#         f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: {word.feats.split("|") if word.feats else "_"}'
#         for word in sentence.words
#     ],
#     sep="\n",
# )

# for i, sentence in enumerate(doc.sentences):
#     # print(f"====== Sentence {i+1} tokens =======")
#     print(
#         *[
#             f"[{word.text}](#)"
#             if (word.upos in ["PRON", "PROPN"])
#             or (word.upos == "DET" and word.xpos == "DEM")
#             else word.text
#             for word in sentence.words
#         ],
#         end="\n===\n",
#     )
#     if i == 5:
#         break


def extract_links_from_markdown(content):
    link_pattern = r"\[(.*?)\]\(#(.*?)\)"
    matches = re.findall(link_pattern, content)

    links = {}
    for mention, link in matches:
        if link not in links:
            links[link] = [mention]
        else:
            links[link].append(mention)

    return links


output = """
[ब्रिटिश](#1) [कोलम्बिया](#1) , ( [अंग्रेज़ी](#1) : [british](#1) [columbia](#1) , [फ्राँसीसी:](#1) a [colombi](#1) [e-britannique](#1) ) [कनाडा](#2) का एक प्रान्त है [जो](#2) [कनाडा](#2) के [प्रशान्त](#3) [महासागर](#3) से लगते पश्चिमी तट पर स्थित है ।

[यह](#2) [कनाडा](#2) का तीसरा सबसे बड़ा प्रान्त है [जिसका](#2) क्षेत्रफल ९,४४,७३५ वर्ग किमी है ।

[२००६](#4) की जनगणना के अनुसार [इस](#2) प्रान्त की कुल जनसंख्या ४१,१३,४८७ थी ।

[इस](#2) प्रान्त की राजधानी [विक्टोरिया](#5) है और राज्य का सबसे बड़ा नगर [वैंकूवर](#6) है ।

[इसी](#6) नगर में [ब्रिटिश](#1) [कोलम्बिया](#1) की लगभग आधी जनसंख्या निवास करती है ( २० लाख ) ।

अन्य बड़े नगर हैं : [केलोव्ना](#7) , [अबोट्स्फोर्ड](#8) , [कैम्लूप्स](#9) , [नानाइमो](#10) और [प्रिन्स](#11) [जॉर्ज।](#11)
"""
print(extract_links_from_markdown(output))
