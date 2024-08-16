import pickle
from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie
from genre.utils import get_entity_spans_fairseq as get_entity_spans

# with open("input/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
#     lang_title2wikidataID = pickle.load(f)
#

with open("input/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

# generate Wikipedia titles and language IDs
model = mGENRE.from_pretrained(
    "models/fairseq_multilingual_entity_disambiguation"
).eval()

sentences = [
    """नेता जी ने 5 जुलाई 1943 को [START] सिंगापुर [END] के टाउन हाल के सामने 'सुप्रीम कमाण्डर' (सर्वोच्च सेनापति) के रूप में सेना को सम्बोधित करते हुए "दिल्ली चलो!" का नारा दिया और जापानी सेना के साथ मिलकर ब्रिटिश व कामनवेल्थ सेना से बर्मा सहित इम्फाल और कोहिमा में एक साथ जमकर मोर्चा लिया।""",
    """[START] नेताजी [END] ने 5 जुलाई 1943 को सिंगापुर के टाउन हाल के सामने 'सुप्रीम कमाण्डर' (सर्वोच्च सेनापति) के रूप में सेना को सम्बोधित करते हुए "दिल्ली चलो!" का नारा दिया और जापानी सेना के साथ मिलकर ब्रिटिश व कामनवेल्थ सेना से बर्मा सहित इम्फाल और कोहिमा में एक साथ जमकर मोर्चा लिया।""",
]
# model.sample(
#     sentences,
#     prefix_allowed_tokens_fn=lambda batch_id, sent: [
#         e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
#     ],
#     text_to_id=lambda x: max(
#         lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
#         key=lambda y: int(y[1:]),
#     ),
#     marginalize=True,
# )

ans = model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e
        for e in trie.get(sent.tolist())
        if e < len(model.task.target_dictionary)
        # for huggingface/transformers
        # if e < len(model2.tokenizer) - 1
    ],
    marginalize=True,
)
for annot in ans:
    annot = sorted(annot, key=lambda x: x["score"], reverse=True)
    print(annot[0]["text"])
