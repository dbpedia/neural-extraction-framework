"""
embed_catalog_f2lm.py — One-time: embed the full DBpedia property catalog
with F2LM, save to disk. Parses the TTL ontology file DIRECTLY (self-
contained) — does NOT import predicate_linking.py, avoiding its
googletrans/SPARQLWrapper dependencies entirely (which caused an httpx
version conflict that could have broken the training pipeline).

Run once:
    python3 embed_catalog_f2lm.py
"""

import os
import json
import numpy as np
from typing import List, Optional, Set

from rdflib import Graph, RDF, RDFS
from rdflib.namespace import SKOS
from sentence_transformers import SentenceTransformer

TTL_PATH = os.path.expanduser(
    "~/neural-extraction-framework/GSoC25_H/ontology_input/ontology--DEV_type=parsed.ttl"
)
DBO_PREFIX = "http://dbpedia.org/ontology/"

OUTPUT_EMBEDDINGS = os.path.expanduser("~/f2lm_property_embeddings.npy")
OUTPUT_CATALOG = os.path.expanduser("~/f2lm_property_catalog.json")


def parse_property_catalog(ttl_path: str) -> List[dict]:
    """Self-contained TTL parser — same logic as predicate_linking.py's
    get_property_catalog(), but with zero external dependencies beyond
    rdflib itself."""
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
                if getattr(c, "language", None) in ("en", None):
                    comment_en = str(c)
                    break
            except Exception:
                comment_en = str(c)

        properties.append({
            "property_uri": p_str,
            "labels_en": sorted(set(labels_en)),
            "labels_hi": sorted(set(labels_hi)),
            "alt_labels": sorted(set(alt_labels)),
            "comment_en": comment_en,
        })

    return properties


def compose_property_text(entry: dict) -> str:
    texts: List[str] = []
    labels = (entry.get("labels_en") or []) + (entry.get("alt_labels") or [])
    if labels:
        texts.append(" | ".join(labels))
    else:
        local_name = entry["property_uri"].split("/")[-1]
        texts.append(local_name)
    comment = entry.get("comment_en")
    if comment:
        texts.append(comment)
    return " | ".join(texts)


def main():
    print("Parsing DBpedia ontology TTL file directly...")
    catalog = parse_property_catalog(TTL_PATH)
    print(f"Total properties: {len(catalog)}")

    texts = [compose_property_text(entry) for entry in catalog]

    print("Loading F2LM model (codefuse-ai/F2LLM-v2-1.7B)...")
    model = SentenceTransformer("codefuse-ai/F2LLM-v2-1.7B", trust_remote_code=True)
    print(f"Loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    print(f"Embedding {len(texts)} properties...")
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True, batch_size=32,
    ).astype(np.float32)

    np.save(OUTPUT_EMBEDDINGS, embeddings)
    with open(OUTPUT_CATALOG, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False)

    print(f"\nSaved embeddings: {OUTPUT_EMBEDDINGS}  shape={embeddings.shape}")
    print(f"Saved catalog: {OUTPUT_CATALOG}")


if __name__ == "__main__":
    main()
