import os
import json
import time
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher
import asyncio
import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, SKOS
from sentence_transformers import SentenceTransformer
from SPARQLWrapper import SPARQLWrapper, JSON


from googletrans import Translator


DBPEDIA_SPARQL_ENDPOINT = "http://dbpedia.org/sparql"
DBO_PREFIX = "http://dbpedia.org/ontology/"
DBR_PREFIX = "http://dbpedia.org/resource/"


# keep catalog and index; embeddings are saved separately as .npy via _cache_bin_path.
ALLOWED_JSON_CACHES: Set[str] = {
    "dbpedia_property_catalog",
    "dbpedia_property_index",
}


def _get_sparql() -> SPARQLWrapper:
    sparql = SPARQLWrapper(DBPEDIA_SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(30)
    return sparql


def _query_with_retries(query: str, retries: int = 2, sleep_seconds: float = 0.8) -> dict:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            sparql = _get_sparql()
            sparql.setQuery(query)
            return sparql.query().convert()
        except Exception as e: 
            last_err = e
            time.sleep(sleep_seconds)
    raise last_err  # type: ignore[misc]


def _cache_path(name: str) -> str:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ontology")
    os.makedirs(base_dir, exist_ok=True)
    safe_name = name.replace("/", "_")
    return os.path.join(base_dir, f"{safe_name}.json")

def _cache_bin_path(name: str, ext: str) -> str:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "ontology")
    os.makedirs(base_dir, exist_ok=True)
    safe_name = name.replace("/", "_")
    return os.path.join(base_dir, f"{safe_name}.{ext}")


def _load_cache(name: str) -> Optional[dict]:
    # Only load from disk for allowlisted cache names
    if name not in ALLOWED_JSON_CACHES:
        return None
    path = _cache_path(name)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache(name: str, data: dict):
    # Only persist to disk for allowlisted cache names
    if name not in ALLOWED_JSON_CACHES:
        return
    path = _cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_property_catalog() -> List[dict]:
    """Parse TTL and build a catalog of dbo:* properties, cached as JSON.

    Returns a list of dicts with keys:
      property_uri, labels_en, labels_hi, alt_labels, comment_en, domains, ranges
    """
    cache_key = "dbpedia_property_catalog"
    cached = _load_cache(cache_key)
    if cached is not None and isinstance(cached, dict) and "properties" in cached:
        return cached["properties"]

    ttl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ontology_input", "ontology--DEV_type=parsed.ttl")

    if not os.path.exists(ttl_path):
        raise FileNotFoundError(f"TTL ontology not found at {ttl_path}")

    g = Graph()
    g.parse(ttl_path, format="turtle")

    properties: List[dict] = []
    for p in g.subjects(RDF.type, RDF.Property):
        p_str = str(p)
        if not p_str.startswith(DBO_PREFIX):
            continue
        labels_en: List[str] = []
        labels_hi: List[str] = []
        alt_labels: List[str] = []
        comment_en: Optional[str] = None
        domains: Set[str] = set()
        ranges: Set[str] = set()

        for _, _, lbl in g.triples((p, RDFS.label, None)):
            try:
                lang = lbl.language or ""
                if lang == "hi":
                    labels_hi.append(str(lbl))
                elif lang == "en" or lang == "":
                    labels_en.append(str(lbl))
            except Exception:
                labels_en.append(str(lbl))

        for _, _, alt in g.triples((p, SKOS.altLabel, None)):
            alt_labels.append(str(alt))

        for _, _, c in g.triples((p, RDFS.comment, None)):
            try:
                if getattr(c, "language", None) == "en" or getattr(c, "language", None) is None:
                    # Keep comment brief to reduce embedding size
                    comment_en = str(c)
                    break
            except Exception:
                comment_en = str(c)

        for _, _, d in g.triples((p, RDFS.domain, None)):
            domains.add(str(d))
        for _, _, r in g.triples((p, RDFS.range, None)):
            ranges.add(str(r))

        properties.append(
            {
                "property_uri": p_str,
                "labels_en": sorted(set(labels_en)),
                "labels_hi": sorted(set(labels_hi)),
                "alt_labels": sorted(set(alt_labels)),
                "comment_en": comment_en,
                "domains": sorted(domains),
                "ranges": sorted(ranges),
            }
        )

    _save_cache(cache_key, {"properties": properties})
    return properties


def _compose_property_text(entry: dict) -> str:
    texts: List[str] = []
    labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
    if labels:
        texts.append(" | ".join(labels))
    else:
        # fallback to local name
        local_name = entry["property_uri"].split("/")[-1]
        texts.append(local_name)
    comment = entry.get("comment_en")
    if comment:
        texts.append(comment)
    return " | ".join(texts)



def _translate_to_english(text: str, src_lang: str) -> str:
    """Translate input text to English using googletrans when available.

    If translation fails or the translator is unavailable, returns the original text.
    """
    if not text:
        return text
    if src_lang == "en":
        return text
    try:
        translator = Translator()
        translated_result = asyncio.run(translator.translate(text, src=src_lang, dest="en"))
        if translated_result and getattr(translated_result, "text", ""):
            return translated_result.text
    except Exception:
        pass
    return text


def ensure_property_embeddings(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> Tuple[np.ndarray, List[dict]]:
    """Ensure property embeddings and index are built and cached.

    Returns:
      (embeddings: np.ndarray [num_props, dim], index: List[{'property_uri','text'}])
    """
    emb_path = _cache_bin_path("dbpedia_property_embeddings", "npy")
    idx_path = _cache_path("dbpedia_property_index")

    if os.path.exists(emb_path) and os.path.exists(idx_path):
        embeddings = np.load(emb_path)
        with open(idx_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        return embeddings, index

    catalog = get_property_catalog()
    index: List[dict] = []
    texts: List[str] = []
    for entry in catalog:
        text = _compose_property_text(entry)
        index.append({"property_uri": entry["property_uri"], "text": text})
        texts.append(text)

    model = SentenceTransformer(model_name)
    # Batch encode for memory efficiency
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    np.save(emb_path, embeddings)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return embeddings, index


# Embedding model singleton to avoid repeated loads
_EMBEDDING_MODEL_SINGLETON: Optional[SentenceTransformer] = None
_EMBEDDING_MODEL_NAME: Optional[str] = None


def get_embedding_model(model_name: str = "intfloat/multilingual-e5-large-instruct") -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for query encoding.

    Streamlit reruns the script but keeps imported modules cached; a module-level
    singleton avoids re-loading the heavy model on every function call.
    """
    global _EMBEDDING_MODEL_SINGLETON, _EMBEDDING_MODEL_NAME
    if _EMBEDDING_MODEL_SINGLETON is None or _EMBEDDING_MODEL_NAME != model_name:
        _EMBEDDING_MODEL_SINGLETON = SentenceTransformer(model_name)
        _EMBEDDING_MODEL_NAME = model_name
    return _EMBEDDING_MODEL_SINGLETON


def _resource_uri_from_title(title: str) -> str:
    return f"{DBR_PREFIX}{title.replace(' ', '_')}"


def fetch_dbpedia_properties_between(s_uri: str, o_uri: str) -> Dict[str, List[str]]:
    """Return ontology properties observed between S and O in both directions.

    Args:
        s_uri: Full resource URI for subject, e.g., http://dbpedia.org/resource/Sachin_Tendulkar
        o_uri: Full resource URI for object
    """
    cache_key = f"props_between::{s_uri}::{o_uri}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    forward_q = f"""
    SELECT ?p WHERE {{
      <{s_uri}> ?p <{o_uri}> .
      FILTER(STRSTARTS(STR(?p), "{DBO_PREFIX}"))
    }}
    """

    reverse_q = f"""
    SELECT ?p WHERE {{
      <{o_uri}> ?p <{s_uri}> .
      FILTER(STRSTARTS(STR(?p), "{DBO_PREFIX}"))
    }}
    """

    f_res = _query_with_retries(forward_q)
    r_res = _query_with_retries(reverse_q)

    forward_props = [b["p"]["value"] for b in f_res["results"]["bindings"]]
    reverse_props = [b["p"]["value"] for b in r_res["results"]["bindings"]]

    out = {"S->O": forward_props, "O->S": reverse_props}
    _save_cache(cache_key, out)
    return out


def fetch_types(uri: str) -> Set[str]:
    """Return rdf:type set for a resource URI.
    Example: http://dbpedia.org/resource/Sachin_Tendulkar
    """
    cache_key = f"types::{uri}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return set(cached.get("types", []))

    q = f"""
    SELECT ?type WHERE {{ <{uri}> a ?type }}
    """
    res = _query_with_retries(q)
    types = {b["type"]["value"] for b in res["results"]["bindings"]}
    _save_cache(cache_key, {"types": sorted(types)})
    return types


def _get_property_domain_range(prop_uri: str) -> Tuple[Set[str], Set[str]]:
    cache_key = f"prop_dr::{prop_uri}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return set(cached.get("domains", [])), set(cached.get("ranges", []))

    q = f"""
    SELECT ?d ?r WHERE {{
      OPTIONAL {{ <{prop_uri}> <http://www.w3.org/2000/01/rdf-schema#domain> ?d }}
      OPTIONAL {{ <{prop_uri}> <http://www.w3.org/2000/01/rdf-schema#range> ?r }}
    }}
    """
    res = _query_with_retries(q)
    domains = {b["d"]["value"] for b in res["results"]["bindings"] if "d" in b}
    ranges = {b["r"]["value"] for b in res["results"]["bindings"] if "r" in b}
    _save_cache(cache_key, {"domains": sorted(domains), "ranges": sorted(ranges)})
    return domains, ranges


def _get_property_labels(prop_uri: str) -> Dict[str, List[str]]:
    cache_key = f"prop_labels::{prop_uri}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    q = f"""
    SELECT ?label ?alt WHERE {{
      <{prop_uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?label .
      FILTER(LANGMATCHES(LANG(?label), 'en') || LANGMATCHES(LANG(?label), 'hi'))
      OPTIONAL {{ <{prop_uri}> <http://www.w3.org/2004/02/skos/core#altLabel> ?alt }}
    }}
    """
    res = _query_with_retries(q)
    labels_en: List[str] = []
    labels_hi: List[str] = []
    alts: List[str] = []
    for b in res["results"]["bindings"]:
        if "label" in b:
            lbl = b["label"]["value"]
            lang = b["label"].get("xml:lang", "")
            if lang == "hi":
                labels_hi.append(lbl)
            else:
                labels_en.append(lbl)
        if "alt" in b:
            alts.append(b["alt"]["value"])
    out = {"en": sorted(set(labels_en)), "hi": sorted(set(labels_hi)), "alt": sorted(set(alts))}
    _save_cache(cache_key, out)
    return out


def _lexical_candidate_properties(surface_predicate: str, lang: str = "hi", limit: int = 30) -> List[str]:
    """Find dbo properties whose label contains the surface text (case-insensitive).

    MVP approach using SPARQL CONTAINS; this is a recall-oriented heuristic.
    """
    text = surface_predicate.strip().lower()
    # sanitize quotes
    text = text.replace("\"", " ").replace("'", " ")

    # try English lexical search by translating non-English predicates
    if lang != "en":
        translated = _translate_to_english(surface_predicate, src_lang=lang).strip().lower()
        translated = translated.replace("\"", " ").replace("'", " ")
        text_for_query = translated if translated else text
        pref_lang = "en"
        if not translated or translated == text:
            # fallback to original language if translation is unavailable/no-op
            pref_lang = "hi" if lang == "hi" else "en"
            text_for_query = text
    else:
        pref_lang = "en"
        text_for_query = text
    q = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?p ?label WHERE {{
      ?p a <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .
      FILTER(STRSTARTS(STR(?p), "{DBO_PREFIX}"))
      ?p rdfs:label ?label .
      FILTER(LANGMATCHES(LANG(?label), '{pref_lang}') || LANGMATCHES(LANG(?label), 'en'))
      FILTER(CONTAINS(LCASE(STR(?label)), "{text_for_query}"))
    }} LIMIT {limit}
    """
    try:
        res = _query_with_retries(q)
    except Exception as e:
        print(f"[PredicateLink] Lexical candidate properties | error: {e}")
        return []
    props = [b["p"]["value"] for b in res["results"]["bindings"]]
    return props


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_candidate(prop_uri: str, s_types: Set[str], o_types: Set[str], s_to_o: List[str], o_to_s: List[str], surface_predicate: str, embedding_score_lookup: Optional[Dict[str, float]] = None) -> Tuple[float, str, Dict[str, float]]:
    labels = _get_property_labels(prop_uri)
    text_candidates: List[str] = labels.get("en", []) + labels.get("hi", []) + labels.get("alt", [])
    lex_score = max((_similarity(surface_predicate, t) for t in text_candidates), default=0.0)

    domains, ranges = _get_property_domain_range(prop_uri)
    type_score = 0.0
    if domains or ranges:
        dom_match = bool(domains.intersection(s_types))
        ran_match = bool(ranges.intersection(o_types))
        type_score = 1.0 if (dom_match and ran_match) else (0.5 if (dom_match or ran_match) else 0.0)

    direction = "none"
    graph_score = 0.0
    if prop_uri in s_to_o:
        graph_score = 1.0
        direction = "S->O"
    elif prop_uri in o_to_s:
        graph_score = 1.0
        direction = "O->S"

    emb_score = 0.0
    if embedding_score_lookup is not None:
        emb_score = float(embedding_score_lookup.get(prop_uri, 0.0))

    w_graph, w_emb, w_lex, w_type = 0.4, 0.3, 0.2, 0.1
    # weighted sum -> prioritize graph, then embedding, then lexical, then type
    score = w_graph * graph_score + w_emb * emb_score + w_lex * lex_score + w_type * type_score
    evidence = {"graph": graph_score, "emb": emb_score, "lex": lex_score, "type": type_score}
    return score, direction, evidence


def link_predicate(surface_predicate: str, subject_title: str, object_title: str, lang: str = "hi") -> dict:
    """Return a mapping for predicate surface to a dbo:* property.

    Output schema:
    {
      "property_uri": str or None,
      "property_label": {"en": str|None, "hi": str|None},
      "score": float,
      "direction": "S->O"|"O->S"|"none",
      "evidence": {"graph": float, "lex": float, "type": float},
      "candidates": [ {"property_uri": str, "score": float} ]  # top few for debugging
    }
    """
    start_ts = time.time()
    print(f"[PredicateLink] Start | S='{subject_title}' O='{object_title}' pred='{surface_predicate}' lang={lang}")
    s_uri = _resource_uri_from_title(subject_title)
    o_uri = _resource_uri_from_title(object_title)

    t0 = time.time()
    between = fetch_dbpedia_properties_between(s_uri, o_uri)
    s_to_o = between.get("S->O", [])
    o_to_s = between.get("O->S", [])
    print(f"[PredicateLink] Graph candidates | S->O={len(s_to_o)} O->S={len(o_to_s)} | dt={(time.time()-t0):.2f}s")

    # Candidate pool: graph properties + lexical + embedding matches
    candidates: Set[str] = set(s_to_o) | set(o_to_s)

    t1 = time.time()
    lexical_props = _lexical_candidate_properties(surface_predicate, lang=lang)
    candidates.update(lexical_props)
    print(f"[PredicateLink] Lexical candidates | n={len(lexical_props)} | dt={(time.time()-t1):.2f}s")

    # Embedding candidates
    try:
        t2 = time.time()
        # replaced sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (faster) 
        # with intfloat/multilingual-e5-large-instruct (better)
        embeddings, index = ensure_property_embeddings(model_name="intfloat/multilingual-e5-large-instruct")

        # Use cached embedding model to avoid repeated loads
        model = get_embedding_model("intfloat/multilingual-e5-large-instruct")
        q_vec = model.encode([surface_predicate], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        # cosine sim with normalized vectors is dot product
        sims = embeddings @ q_vec
        top_k = min(20, sims.shape[0])
        top_idx = np.argpartition(-sims, top_k - 1)[:top_k]
        emb_scores: Dict[str, float] = {}
        for i in top_idx:
            uri = index[i]["property_uri"]
            emb_scores[uri] = float(sims[i])
        candidates.update(emb_scores.keys())
        print(f"[PredicateLink] Embedding candidates | n={len(emb_scores)} | dt={(time.time()-t2):.2f}s")
    except Exception:
        emb_scores = {}
        print("[PredicateLink] Embedding candidates | failed to compute (missing model or build error)")

    if not candidates:
        print("[PredicateLink] No candidates found from graph/lexical/embeddings")
        return {
            "property_uri": None,
            "property_label": {"en": None, "hi": None},
            "score": 0.0,
            "direction": "none",
            "evidence": {"graph": 0.0, "emb": 0.0, "lex": 0.0, "type": 0.0},
            "candidates": [],
        }

    t3 = time.time()
    s_types = fetch_types(s_uri)
    o_types = fetch_types(o_uri)
    print(f"[PredicateLink] Types fetched | |S_types|={len(s_types)} |O_types|={len(o_types)} | dt={(time.time()-t3):.2f}s")

    t4 = time.time()
    scored: List[Tuple[str, float, str, Dict[str, float]]] = []
    BLACKLISTED_PROPERTIES = {
        "http://dbpedia.org/ontology/wikiPageWikiLink",
        "http://dbpedia.org/ontology/wikiPageRedirects",
        "http://dbpedia.org/ontology/wikiPageDisambiguates",
        "http://dbpedia.org/ontology/wikiPageExternalLink",
        "http://dbpedia.org/ontology/wikiPageDisambiguation",
        "http://dbpedia.org/ontology/wikiPageRevision",
        "http://dbpedia.org/ontology/wikiPageSource",
        "http://dbpedia.org/ontology/wikiPageRevisionID",
        "http://dbpedia.org/ontology/hraState",
        "http://dbpedia.org/ontology/logo",
    }

    for p in candidates:
        if p in BLACKLISTED_PROPERTIES:
            continue
        score, direction, evidence = _score_candidate(p, s_types, o_types, s_to_o, o_to_s, surface_predicate, emb_scores)
        scored.append((p, score, direction, evidence))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_uri, best_score, best_dir, best_evidence = scored[0]
    print(f"[PredicateLink] Scoring done | candidates={len(scored)} | top_score={best_score:.3f} | dt={(time.time()-t4):.2f}s")

    labels = _get_property_labels(best_uri)
    label_en = labels.get("en", [None])[0] if labels.get("en") else None
    label_hi = labels.get("hi", [None])[0] if labels.get("hi") else None
    total_dt = time.time() - start_ts
    print(f"[PredicateLink] Top candidate | uri={best_uri} | dir={best_dir} | label_en={label_en} | score={best_score:.3f} | total_dt={total_dt:.2f}s")

    return {
        "property_uri": best_uri,
        "property_label": {"en": label_en, "hi": label_hi},
        "score": float(best_score),
        "direction": best_dir,
        "evidence": best_evidence,
        "candidates": [{"property_uri": p, "score": float(s)} for p, s, _, _ in scored[:5]],
    }

def link_predicates_batch(triples: List[Tuple[str, str, str]], lang: str = "hi", top_k: int = 20) -> List[dict]:
    """Batch predicate linking for multiple (S, P, O) triples.

    Args:
        triples: List of (subject_title, surface_predicate, object_title)
        lang: Language code (default: "hi")
        top_k: Number of embedding candidates to consider per query

    Returns:
        List of result dicts in the same schema as link_predicate, in order.
    """
    if not triples:
        return []

    # Ensure static data and model are ready once
    embeddings, index = ensure_property_embeddings(model_name="intfloat/multilingual-e5-large-instruct")
    model = get_embedding_model("intfloat/multilingual-e5-large-instruct")

    # Encode all surface predicates together for efficiency
    surface_predicates = [p for (_, p, __) in triples]
    q_mat = model.encode(surface_predicates, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    # Precompute top embedding candidates per query
    # sims shape: [num_props, batch] to reuse memory efficiently
    sims = embeddings @ q_mat.T  # [num_props, B]
    B = sims.shape[1]

    results: List[dict] = []
    for idx in range(B):
        s_title, surface_p, o_title = triples[idx]
        print(f"[PredicateLink] Start | S='{s_title}' O='{o_title}' pred='{surface_p}' lang={lang}")

        s_uri = _resource_uri_from_title(s_title)
        o_uri = _resource_uri_from_title(o_title)

        t0 = time.time()
        between = fetch_dbpedia_properties_between(s_uri, o_uri)
        s_to_o = between.get("S->O", [])
        o_to_s = between.get("O->S", [])
        print(f"[PredicateLink] Graph candidates | S->O={len(s_to_o)} O->S={len(o_to_s)} | dt={(time.time()-t0):.2f}s")

        # Collect candidates: graph + lexical + embedding
        candidates: Set[str] = set(s_to_o) | set(o_to_s)

        t1 = time.time()
        lexical_props = _lexical_candidate_properties(surface_p, lang=lang)
        candidates.update(lexical_props)
        print(f"[PredicateLink] Lexical candidates | n={len(lexical_props)} | dt={(time.time()-t1):.2f}s")

        # Embedding candidates using precomputed sims column
        emb_scores: Dict[str, float] = {}
        col = sims[:, idx]
        k = min(top_k, col.shape[0])
        top_idx = np.argpartition(-col, k - 1)[:k]
        for i in top_idx:
            uri = index[i]["property_uri"]
            emb_scores[uri] = float(col[i])
        candidates.update(emb_scores.keys())
        print(f"[PredicateLink] Embedding candidates | n={len(emb_scores)}")

        if not candidates:
            print("[PredicateLink] No candidates found from graph/lexical/embeddings")
            results.append({
                "property_uri": None,
                "property_label": {"en": None, "hi": None},
                "score": 0.0,
                "direction": "none",
                "evidence": {"graph": 0.0, "emb": 0.0, "lex": 0.0, "type": 0.0},
                "candidates": [],
                "triple": (s_title, surface_p, o_title),
            })
            continue

        t3 = time.time()
        s_types = fetch_types(s_uri)
        o_types = fetch_types(o_uri)
        print(f"[PredicateLink] Types fetched | |S_types|={len(s_types)} |O_types|={len(o_types)} | dt={(time.time()-t3):.2f}s")

        BLACKLISTED_PROPERTIES = {
            "http://dbpedia.org/ontology/wikiPageWikiLink",
            "http://dbpedia.org/ontology/wikiPageRedirects",
            "http://dbpedia.org/ontology/wikiPageDisambiguates",
            "http://dbpedia.org/ontology/wikiPageExternalLink",
            "http://dbpedia.org/ontology/wikiPageDisambiguation",
            "http://dbpedia.org/ontology/wikiPageRevision",
            "http://dbpedia.org/ontology/wikiPageSource",
            "http://dbpedia.org/ontology/wikiPageRevisionID",
            "http://dbpedia.org/ontology/hraState",
            "http://dbpedia.org/ontology/logo",
        }

        t4 = time.time()
        scored: List[Tuple[str, float, str, Dict[str, float]]] = []
        for p_uri in candidates:
            if p_uri in BLACKLISTED_PROPERTIES:
                continue
            score, direction, evidence = _score_candidate(p_uri, s_types, o_types, s_to_o, o_to_s, surface_p, emb_scores)
            scored.append((p_uri, score, direction, evidence))

        if not scored:
            results.append({
                "property_uri": None,
                "property_label": {"en": None, "hi": None},
                "score": 0.0,
                "direction": "none",
                "evidence": {"graph": 0.0, "emb": 0.0, "lex": 0.0, "type": 0.0},
                "candidates": [],
                "triple": (s_title, surface_p, o_title),
            })
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        best_uri, best_score, best_dir, best_evidence = scored[0]
        print(f"[PredicateLink] Scoring done | candidates={len(scored)} | top_score={best_score:.3f} | dt={(time.time()-t4):.2f}s")

        labels = _get_property_labels(best_uri)
        label_en = labels.get("en", [None])[0] if labels.get("en") else None
        label_hi = labels.get("hi", [None])[0] if labels.get("hi") else None

        results.append({
            "property_uri": best_uri,
            "property_label": {"en": label_en, "hi": label_hi},
            "score": float(best_score),
            "direction": best_dir,
            "evidence": best_evidence,
            "candidates": [{"property_uri": p, "score": float(s)} for p, s, _, _ in scored[:5]],
            "triple": (s_title, surface_p, o_title),
        })

    return results


__all__ = [
    "link_predicate",
    "link_predicates_batch",
    "get_embedding_model",
    "fetch_dbpedia_properties_between",
    "fetch_types",
]

def main():

    # test cases with English subject/object titles and Hindi predicates. this reflects the state after a successful entity linking step.
    test_triples = [
        ("Sachin_Tendulkar", "जन्म स्थान", "Mumbai"),
        ("Microsoft", "द्वारा स्थापित", "Bill_Gates"),
        ("Satyajit_Ray", "ने निर्देशित किया", "Pather_Panchali"),
        ("India", "राजधानी", "New_Delhi"),
        ("Taj_Mahal", "में स्थित है", "Agra"),
        ("Amitabh_Bachchan", "के पिता", "Harivansh_Rai_Bachchan"),
    ]

    print("RUNNING HINDI BATCH PREDICATE LINKING TEST SUITE")

    start_time = time.time()
    batch_results = link_predicates_batch(test_triples, lang="hi", top_k=15)
    end_time = time.time()
    print(f"\nBatch processing finished in {end_time - start_time:.2f} seconds.")


    for i, result in enumerate(batch_results):
        s, p, o = result['triple']
        print(f"\nTEST CASE {i+1}/{len(batch_results)}")

        print("\nRESULT")
        print(f"Input: '{s}' | '{p}' | '{o}'")
        print(f"Top Property: {result.get('property_uri')}")
        print(f"Best Label (en): {result.get('property_label', {}).get('en')}")
        print(f"Score: {result.get('score'):.3f}")
        print(f"Direction: {result.get('direction')}")
        print(f"Evidence: {json.dumps(result.get('evidence'))}")
        print("-" * 20)

if __name__ == "__main__":
    main()