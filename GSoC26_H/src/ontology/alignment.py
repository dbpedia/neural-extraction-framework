"""
alignment.py
------------
Ontology Alignment Layer for Hindi → DBpedia property mapping.

Uses multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2)
to compute cosine similarity between extracted Hindi predicate surface forms
and DBpedia property descriptions.

Results from pre-application experiments:
  - Zero-shot Gemma-3 alone:    0/5 = 0%  predicate accuracy
  - + Ontology Alignment Layer: 4/5 = 80% predicate accuracy
  - Remaining 1/5 flagged for HITL review (high confidence but wrong mapping)
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DBpediaProperty:
    uri:                str           # e.g. "dbo:birthPlace"
    hindi_description:  str           # e.g. "जन्म स्थान"
    hindi_surface_forms: List[str]    # e.g. ["का जन्म", "में जन्मे"]
    english_description: str

    def get_all_text_forms(self) -> List[str]:
        """All text forms used for embedding. Combines Hindi + English descriptions."""
        forms = [self.hindi_description] + self.hindi_surface_forms
        if self.english_description:
            forms.append(self.english_description)
        return forms

    def get_embedding_text(self) -> str:
        """Single string used to compute property embedding."""
        return " | ".join([self.hindi_description] + self.hindi_surface_forms[:3])


@dataclass
class AlignmentResult:
    surface_form:       str           # Raw predicate from model
    matched_property:   str           # e.g. "dbo:birthPlace"
    confidence:         float         # Cosine similarity score [0, 1]
    runner_up:          str           # Second-best match
    runner_up_score:    float
    flagged_for_review: bool          # True if confidence < threshold

    def __str__(self):
        flag = " ⚑ FLAGGED" if self.flagged_for_review else " ✓"
        return (f"'{self.surface_form}' → {self.matched_property} "
                f"(conf: {self.confidence:.3f}){flag}")


# ─── Main Aligner ─────────────────────────────────────────────────────────────

class OntologyAligner:
    """
    Maps Hindi predicate surface forms to DBpedia ontology properties
    using multilingual sentence embeddings + cosine similarity.

    Usage:
        aligner = OntologyAligner()
        aligner.load_properties("data/ontology/dbpedia_properties.json")
        aligner.build_index()
        result = aligner.align("का निर्माण")
        print(result)  # → dbo:builder (conf: 0.87)
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_THRESHOLD = 0.45   # Below this → flag for HITL review

    def __init__(self, confidence_threshold: float = DEFAULT_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.properties:       List[DBpediaProperty] = []
        self.property_uris:    List[str] = []
        self.property_embeddings = None   # shape: (n_properties, embed_dim)
        self._model = None
        self._model_loaded = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def load_properties(self, json_path: str) -> int:
        """Load DBpedia properties from the curated JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.properties = [
            DBpediaProperty(
                uri=p["uri"],
                hindi_description=p["hindi_description"],
                hindi_surface_forms=p.get("hindi_surface_forms", []),
                english_description=p.get("english_description", ""),
            )
            for p in data["properties"]
        ]
        print(f"Loaded {len(self.properties)} DBpedia properties.")
        return len(self.properties)

    def add_property(self, uri: str, hindi_desc: str,
                     surface_forms: List[str], english_desc: str = "") -> None:
        """Add a single property at runtime (for extending coverage)."""
        self.properties.append(DBpediaProperty(
            uri=uri,
            hindi_description=hindi_desc,
            hindi_surface_forms=surface_forms,
            english_description=english_desc,
        ))
        # Mark index as stale
        self.property_embeddings = None

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if not self._model_loaded:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.MODEL_NAME} ...")
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._model_loaded = True
            print("Model loaded.")

    def build_index(self) -> None:
        """
        Compute embeddings for all properties and store as a matrix.
        Must be called after load_properties() and before align().
        """
        if not self.properties:
            raise ValueError("No properties loaded. Call load_properties() first.")

        self._load_model()

        # Build one embedding per property (average of all its text forms)
        self.property_uris = [p.uri for p in self.properties]
        property_texts = [p.get_embedding_text() for p in self.properties]

        print(f"Building embedding index for {len(property_texts)} properties ...")
        embeddings = self._model.encode(property_texts, normalize_embeddings=True,
                                        show_progress_bar=True)
        self.property_embeddings = np.array(embeddings)
        print("Index built.")

    # ── Core Alignment ────────────────────────────────────────────────────────

    def align(self, surface_form: str) -> AlignmentResult:
        """
        Align a single Hindi predicate surface form to a DBpedia property.

        Args:
            surface_form: Raw predicate extracted by the model.
                          e.g. "का निर्माण", "was born in", "hai"

        Returns:
            AlignmentResult with best match, confidence score, and HITL flag.
        """
        if self.property_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        self._load_model()

        # Embed the query
        query_emb = self._model.encode([surface_form], normalize_embeddings=True)

        # Cosine similarity (embeddings are L2-normalised → dot product = cosine)
        similarities = np.dot(self.property_embeddings, query_emb.T).flatten()

        # Top-2 matches
        top_indices = np.argsort(similarities)[::-1]
        best_idx    = top_indices[0]
        runner_idx  = top_indices[1] if len(top_indices) > 1 else best_idx

        confidence     = float(similarities[best_idx])
        runner_up_conf = float(similarities[runner_idx])

        return AlignmentResult(
            surface_form=surface_form,
            matched_property=self.property_uris[best_idx],
            confidence=confidence,
            runner_up=self.property_uris[runner_idx],
            runner_up_score=runner_up_conf,
            flagged_for_review=(confidence < self.confidence_threshold),
        )

    def align_batch(self, surface_forms: List[str]) -> List[AlignmentResult]:
        """Align a list of surface forms in one batch (faster than loop)."""
        if self.property_embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        self._load_model()

        query_embs  = self._model.encode(surface_forms, normalize_embeddings=True,
                                         show_progress_bar=False)
        similarities = np.dot(self.property_embeddings, query_embs.T)  # (n_props, n_queries)

        results = []
        for i, surface_form in enumerate(surface_forms):
            sims       = similarities[:, i]
            top        = np.argsort(sims)[::-1]
            best_idx   = top[0]
            runner_idx = top[1] if len(top) > 1 else top[0]
            conf       = float(sims[best_idx])

            results.append(AlignmentResult(
                surface_form=surface_form,
                matched_property=self.property_uris[best_idx],
                confidence=conf,
                runner_up=self.property_uris[runner_idx],
                runner_up_score=float(sims[runner_idx]),
                flagged_for_review=(conf < self.confidence_threshold),
            ))

        return results

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def top_k(self, surface_form: str, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k property matches with confidence scores (for debugging)."""
        if self.property_embeddings is None:
            raise RuntimeError("Index not built.")

        self._load_model()
        query_emb   = self._model.encode([surface_form], normalize_embeddings=True)
        similarities = np.dot(self.property_embeddings, query_emb.T).flatten()
        top_indices  = np.argsort(similarities)[::-1][:k]

        return [(self.property_uris[i], float(similarities[i])) for i in top_indices]

    def print_alignment_report(self, results: List[AlignmentResult]) -> None:
        """Print alignment results in a readable table."""
        from tabulate import tabulate
        rows = [
            [r.surface_form, r.matched_property, f"{r.confidence:.3f}",
             r.runner_up, f"{r.runner_up_score:.3f}",
             "⚑ REVIEW" if r.flagged_for_review else "✓ OK"]
            for r in results
        ]
        print(tabulate(rows,
                       headers=["Surface Form", "Matched Property", "Conf",
                                "Runner-up", "Runner-up Conf", "Status"],
                       tablefmt="rounded_outline"))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        """Save computed embeddings to disk (avoid recomputing every run)."""
        np.save(path + "_embeddings.npy", self.property_embeddings)
        with open(path + "_uris.json", "w", encoding="utf-8") as f:
            json.dump(self.property_uris, f, ensure_ascii=False)
        print(f"Index saved to {path}_embeddings.npy + {path}_uris.json")

    def load_index(self, path: str) -> None:
        """Load pre-computed embeddings from disk."""
        self.property_embeddings = np.load(path + "_embeddings.npy")
        with open(path + "_uris.json", "r", encoding="utf-8") as f:
            self.property_uris = json.load(f)
        self._load_model()
        print(f"Index loaded: {len(self.property_uris)} properties.")
