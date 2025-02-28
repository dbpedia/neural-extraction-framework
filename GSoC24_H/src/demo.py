import streamlit as st
import wikipedia
import stanza
import torch
from coref import CorefModel
from coref.tokenizer_customization import *
from coref.mcoref import resolve_pronouns_hindi
from indIE import get_triples
import pickle
from genre.fairseq_model import mGENRE
from genre.trie import Trie, MarisaTrie


@st.cache_resource
def load_coref_model():
    # Coref Model Loading
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
    return coref_model


@st.cache_resource
def load_nlp_model():
    nlp = stanza.Pipeline(lang="hi", processors="tokenize,pos")
    return nlp


@st.cache_resource
def load_el_model():
    with open("input/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
        trie = pickle.load(f)
    el_model = mGENRE.from_pretrained(
        "models/fairseq_multilingual_entity_disambiguation"
    ).eval()
    return trie, el_model


@st.cache_data
def get_text_of_wiki_page(article_name: str):
    """Given an article name(need not be exact title of page),
    return the textual content of the wikipedia article.
    We do a search for the articles and select the top-1 result, in case
    where the article name is not the exact title.

    Args:
        article_name (str): Name of a wikipedia article

    Returns:
        str: The text of that article.
    """
    try:
        wikipedia.set_lang("hi")
        article_name_result = wikipedia.page(
            wikipedia.search(article_name)[0], auto_suggest=False
        )
        article_name_content = article_name_result.content
        article_name_content.replace("\n", "").replace("\t", "")
    except:
        return None
    return article_name_content


@st.cache_data
def run_coreference(content):
    coref_model = load_coref_model()
    nlp = load_nlp_model()
    _, out_sentences = resolve_pronouns_hindi([content], coref_model, nlp)
    content = out_sentences[0]
    return content


@st.cache_data
def run_relation_extraction(content):
    nlp = load_nlp_model()
    doc = nlp(content)
    triples = []
    for sentence in doc.sentences:
        sent = sentence.text
        exts, _ = get_triples(sent)
        triples.extend(exts[0])
    return triples


@st.cache_data
def run_entity_linking(triples):
    trie, el_model = load_el_model()
    el_sents = {}
    for relation in triples:
        s, p, o = relation
        el_sents[s] = f"[START] {s} [END] {p} {o}"
        el_sents[o] = f"{s} {p} [START] {o} [END]"

    linked_triples = []
    ans = el_model.sample(
        list(el_sents.values()),
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e
            for e in trie.get(sent.tolist())
            if e < len(el_model.task.target_dictionary)
        ],
    )
    el_maps = {}
    for surface_l, annot in zip(el_sents.keys(), ans):
        annot = sorted(annot, key=lambda x: x["score"], reverse=True)
        el_maps[surface_l] = annot[0]["text"].split(" >> ")[0]

    for relation in triples:
        s, p, o = relation
        linked_triples.append(
            (
                el_maps[s],
                p,
                el_maps[o],
            )
        )
    return linked_triples


# Streamlit App
st.title("Hindi Wikipedia Neural Extractor")
article_name = st.text_input("Enter Wikipedia article name:", "")
content = None
triples = None

if article_name:
    # Fetch Wikipedia content
    content = get_text_of_wiki_page(article_name)
    if content:
        st.text_area("Article Content", content, height=300)
    else:
        st.error("Wikipedia article not found.")
else:
    st.info("Please enter an article name to fetch content.")

# Checkboxes for features
coreference = st.checkbox("Coreference")
relation_extraction = st.checkbox("Relation Extraction")
entity_linking = st.checkbox("Entity Linking")

if coreference:
    st.subheader("Coreference Resolution")
    content = run_coreference(content)
    st.write(content)
    st.divider()

if relation_extraction:
    st.subheader("Relation Extraction")
    num_triples = st.slider("Number of triples to extract:", 1, 30, 5)
    triples = run_relation_extraction(content)
    st.write(triples[:num_triples])

if entity_linking:
    linked_triples = run_entity_linking(triples)
    st.subheader("Entity Linking")
    num_triples = st.slider("Number of triples to show:", 1, 30, 5)
    st.write("Extracted triples with entity linking:")
    st.write(linked_triples[:num_triples])
