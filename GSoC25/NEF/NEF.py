#!/usr/bin/env python3
# nef_cli.py
# Usage examples are at the bottom of this file (or run with -h)

import os, sys, json, re, time, argparse, textwrap
from typing import List, Tuple, Sequence, Dict, Any, Optional

import numpy as np
from urllib.parse import quote
from getpass import getpass

# =============== Gemini client bootstrap ===============
try:
    from google import genai
    from google.genai import types
except Exception as _e:
    sys.stderr.write("ERROR: google-genai is required. Try: pip install google-genai\n")
    raise

def _bootstrap_gemini_client(api_key: Optional[str]) -> "genai.Client":
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key and sys.stdin.isatty():
        key = getpass("Enter your Google (Gemini) API key: ").strip()
    if not key:
        raise RuntimeError("No Gemini API key found. Pass --api-key or set GEMINI_API_KEY.")
    return genai.Client(api_key=key)

# =============== Utils ===============

_YEAR = re.compile(r"^\d{4}$")

def _year_uri(y: str) -> str:
    return f"http://dbpedia.org/resource/{y}"

def _normalize(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1e-12
    return v / n

def _json_from_model(text: str) -> Any:
    t = (text or "").strip()
    # strip markdown fences if any
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{.*\}|\[.*\]", t, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(0))

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        sys.stdout.write((" ".join(map(str, args)) + "\n").encode("utf-8", "ignore").decode("utf-8"))

# =============== Redis Entity Linking ===============

class RedisEntityLinking:
    """Redis-based entity linking; hard-requires Redis (no surface fallbacks here)."""
    def __init__(
        self,
        host: str,
        port: int,
        password: Optional[str],
        connect_timeout: float = 2.0,
        verbose: bool = True,
    ):
        self.available = False
        self.redis_forms = None
        self.redis_redir = None
        self.verbose = verbose
        try:
            import redis  # local import so script still loads without it
            common = dict(host=host, port=port, password=password,
                          socket_connect_timeout=connect_timeout, socket_timeout=2.0, decode_responses=True)
            self.redis_forms = redis.Redis(db=0, **common)
            self.redis_redir  = redis.Redis(db=1, **common)
            self.available = bool(self.redis_forms.ping() and self.redis_redir.ping())
            if self.verbose:
                _safe_print("✓ Connected to Redis" if self.available else "✗ Redis ping failed")
        except Exception as e:
            _safe_print(f"✗ Redis connection error (pipeline will drop ungrounded triples): {e}")

    def _redirect(self, uri: str, max_hops: int = 10) -> str:
        if not self.available:
            return uri
        seen = set()
        cur = uri
        for _ in range(max_hops):
            if cur in seen:
                break
            seen.add(cur)
            nxt = self.redis_redir.get(cur)
            if not nxt:
                return cur
            cur = nxt
        return cur

    def lookup(self, surface_form: str, top_k: int = 5, thr: float = 0.01) -> List[Tuple[str, float]]:
        """
        Strict Redis grounding (no synonyms). Tries simple, non-semantic variants:
        exact, lower, Title Case, underscores, Title+underscores.
        Aggregates counts across variants and follows redirects in db1.
        """
        if not self.available or not surface_form.strip():
            return []

        variants = [
            surface_form,
            surface_form.lower(),
            surface_form.title(),
            surface_form.replace(" ", "_"),
            surface_form.title().replace(" ", "_"),
        ]

        counts: Dict[str, int] = {}
        seen_keys = set()
        for key in variants:
            if key in seen_keys:
                continue
            seen_keys.add(key)
            raw = self.redis_forms.hgetall(key)  # {uri: count}
            if not raw:
                continue
            for uri, v in raw.items():
                canon = self._redirect(uri)  # db1 redirect if any
                counts[canon] = counts.get(canon, 0) + int(v)

        if not counts:
            return []

        max_support = max(counts.values()) or 1
        items = [(uri, c / max_support) for uri, c in counts.items() if (c / max_support) >= thr]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

# =============== Predicate Retriever (precomputed) ===============

class PredicateEmbeddingRetriever:
    """
    Loads embeddings.npy (N, D) and predicates.csv (URIs or CSV with 'predicate' column),
    retrieves top-K predicates via cosine similarity. NO synonym expansion.
    """
    def __init__(
        self,
        client: "genai.Client",
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        embed_model: str = "embedding-001",
        verbose: bool = True,
    ):
        self.client = client
        self.embed_model = embed_model
        self.verbose = verbose

        emb_path, pred_path = self._find_files(embeddings_path, predicates_path)
        self.E: np.ndarray = np.load(emb_path)  # (N, D)
        self.predicates: List[str] = self._load_predicates(pred_path)
        if self.E.shape[0] != len(self.predicates):
            raise ValueError(f"Row count mismatch: embeddings ({self.E.shape[0]}) vs predicates ({len(self.predicates)})")

        self.D = int(self.E.shape[1])
        self.E_norm = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)

        if self.verbose:
            _safe_print(f"✓ Loaded embeddings: {emb_path} shape={self.E.shape}")
            _safe_print(f"✓ Loaded predicates: {pred_path} count={len(self.predicates)}")

    def _find_files(self, emb_path: Optional[str], pred_path: Optional[str]) -> Tuple[str, str]:
        if emb_path and pred_path and os.path.exists(emb_path) and os.path.exists(pred_path):
            return emb_path, pred_path
        cand_emb = ["./embeddings.npy", "../embeddings.npy", "embeddings.npy"]
        cand_pred = ["./predicates.csv", "../predicates.csv", "predicates.csv"]
        e = next((p for p in cand_emb if os.path.exists(p)), None)
        p = next((q for q in cand_pred if os.path.exists(q)), None)
        if not (e and p):
            raise FileNotFoundError(f"Could not find embeddings.npy and predicates.csv in {os.getcwd()} or ../")
        if self.verbose:
            _safe_print(f"✓ Found files: {e}, {p}")
        return e, p

    def _load_predicates(self, path: str) -> List[str]:
        preds: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
            if "predicate" in head.lower():
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if parts:
                        preds.append(parts[0] if head.lower().startswith("predicate") else parts[-1])
            else:
                preds.append(head.strip())
                for line in f:
                    preds.append(line.strip())
        preds = [p for p in preds if p]
        return preds

    def _embed_text(self, text: str) -> np.ndarray:
        cfg = types.EmbedContentConfig(output_dimensionality=int(self.D))
        resp = self.client.models.embed_content(model=self.embed_model, contents=text, config=cfg)
        v = getattr(resp, "embeddings", None)
        v = (v[0].values if v else resp.embedding.values)
        return _normalize(v)

    def get_top_k_predicates(self, relation_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q = self._embed_text(relation_text)      # (D,)
        sims = self.E_norm @ q                   # (N,)
        order = sims.argsort()[-top_k:][::-1]
        return [(self.predicates[i], float(sims[i])) for i in order]

# =============== LLM Disambiguator ===============

class LLMDisambiguator:
    def __init__(
        self,
        client,
        model_name: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str | None = None,
        verbose: bool = True,
        **kwargs,  # absorb any unexpected args from the pipeline
    ):
        self.client = client
        self.model_name = model_name
        self.thr = float(predicate_threshold)
        self.new_predicate_namespace = new_predicate_namespace
        self.verbose = verbose
        # Optionally stash the rest if you ever want to inspect them:
        self._extra_kwargs = kwargs

    # ------------------------ helpers ------------------------

    @staticmethod
    def _as_pairs(lst: Any) -> List[Tuple[str, float]]:
        """Normalize a list of candidates to [(uri:str, score:float), ...]."""
        out: List[Tuple[str, float]] = []
        if not lst:
            return out
        for item in lst:
            uri, score = "", 1.0
            if isinstance(item, (list, tuple)):
                if len(item) >= 1:
                    uri = str(item[0])
                if len(item) >= 2:
                    try:
                        score = float(item[1])
                    except Exception:
                        score = 1.0
            elif isinstance(item, dict):
                uri = str(item.get("uri") or item.get("id") or item.get("value") or "")
                try:
                    score = float(item.get("score", 1.0))
                except Exception:
                    score = 1.0
            else:
                uri = str(item)
                score = 1.0
            if uri:
                out.append((uri, score))
        return out

    @staticmethod
    def _fmt_indexed(lst: List[Tuple[str, float]]) -> str:
        """Pretty print [(uri, score)] as indexed lines; tolerate empty."""
        if not lst:
            return "(empty)"
        try:
            return "\n".join([f'{i}. "{u}" (sim={s:.3f})' for i, (u, s) in enumerate(lst)]) or "(empty)"
        except Exception:
            # ultra-safe fallback
            return "\n".join([f'{i}. "{str(item)}"' for i, item in enumerate(lst)]) or "(empty)"

    @staticmethod
    def _safe_idx(lst: List[Any], value: Any) -> Optional[int]:
        """Clamp int(value) to valid range; return 0 on parse error; None for empty lst."""
        if not lst:
            return None
        try:
            i = int(value)
        except Exception:
            i = 0
        return max(0, min(i, len(lst) - 1))

    @staticmethod
    def _get_json(text: Optional[str]) -> Dict[str, Any]:
        try:
            return json.loads((text or "").strip() or "{}")
        except Exception:
            return {}

    # --------------------- main entrypoint ---------------------

    def disambiguate_triple(
        self,
        context: str,
        subject_candidates: List[Tuple[str, float]],
        predicate_candidates: List[Tuple[str, float]],
        object_candidates: List[Tuple[str, float]],
    ):
        """
        Returns: (subject_uri, predicate_uri, object_uri, meta)
        - subject_uri, object_uri are ALWAYS from the provided candidate lists.
        - predicate_uri is chosen from candidate predicates >= threshold.
        """

        # Normalize inputs so mixed shapes don't crash anything
        subject_candidates   = self._as_pairs(subject_candidates)
        predicate_candidates = self._as_pairs(predicate_candidates)
        object_candidates    = self._as_pairs(object_candidates)

        total_k = len(predicate_candidates)
        sim_map: Dict[str, Tuple[Optional[float], int]] = {
            u: (s, i) for i, (u, s) in enumerate(predicate_candidates)
        }

        # Filter predicates by threshold
        above = [(u, s) for (u, s) in predicate_candidates if (s is not None and s >= self.thr)]

        # If no predicate passes threshold, fall back to top-1 S/O and return None predicate
        if not above:
            s_uri = subject_candidates[0][0] if subject_candidates else ""
            o_uri = object_candidates[0][0] if object_candidates else ""
            meta = {
                "label": "no_predicate_above_threshold",
                "topk": total_k,
                "threshold": self.thr,
            }
            return (s_uri, None, o_uri, meta)

        # Prepare prompt pieces
        allowed = [u for (u, _s) in above]
        pred_list_text = "\n".join([f'- "{u}"' for u in allowed])
        subj_list_text = self._fmt_indexed(subject_candidates)
        obj_list_text  = self._fmt_indexed(object_candidates)

        prompt = f"""Pick the best RDF triple using ONLY these options.

Allowed predicate URIs:
{pred_list_text}

Subject candidates (choose by INDEX):
{subj_list_text}

Object candidates (choose by INDEX):
{obj_list_text}

Context (helps decide, but does NOT add new options):
{context}

Return ONLY strict JSON on one line (no prose):
{{"subject_index": 0, "predicate": "URI", "object_index": 0}}
Rules:
- "predicate" MUST be exactly one URI from Allowed predicate URIs.
- "subject_index" MUST be an integer index from Subject candidates.
- "object_index" MUST be an integer index from Object candidates.
- Do not invent or modify URIs. Do not swap roles.
"""

        # Call the model (Gemini client style)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )

        data = self._get_json(getattr(resp, "text", None))

        # Validate predicate
        pred_uri = data.get("predicate", "")
        if pred_uri not in allowed:
            pred_uri = allowed[0]

        # Clamp indices and map to URIs
        si = self._safe_idx(subject_candidates, data.get("subject_index", 0))
        oi = self._safe_idx(object_candidates,  data.get("object_index", 0))

        s_uri = subject_candidates[si][0] if (subject_candidates and si is not None) else ""
        o_uri = object_candidates[oi][0] if (object_candidates and oi is not None) else ""

        # Paranoia: role-swap & collision guards
        subj_pool = {u for (u, _s) in subject_candidates}
        obj_pool  = {u for (u, _s) in object_candidates}

        # If s_uri is actually from object pool and o_uri from subject pool, swap them back
        if s_uri in obj_pool and o_uri in subj_pool and s_uri not in subj_pool:
            s_uri, o_uri = o_uri, s_uri

        # Avoid identical subject/object when pools differ (rare but can happen with weird lists)
        if subj_pool and obj_pool and s_uri == o_uri and len(obj_pool) > 1:
            # push object to top-1 object if collision looks suspicious
            o_uri = object_candidates[0][0]

        # Build meta (compatible with your previous code)
        chosen_sim, rank0 = sim_map.get(pred_uri, (None, None))
        meta = {
            "label": "candidate",
            "chosen_similarity": float(chosen_sim) if chosen_sim is not None else None,
            "rank_in_topk": (rank0 + 1) if rank0 is not None else None,
            "topk": total_k,
            "threshold": self.thr,
        }

        return (s_uri, pred_uri, o_uri, meta)

# =============== Orchestrator ===============

class EnhancedNEFPipeline:
    """
    End-to-end pipeline:
      1) Extract triples with Gemini
      2) Redis entity linking (subject/object)  [REQUIRED]
      3) Predicate retrieval via precomputed embeddings (no synonyms)
      4) LLM disambiguation
    """
    def __init__(
        self,
        client: "genai.Client",
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str = "http://nef.local/rel/",
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_password: Optional[str] = None,
        verbose: bool = True,
    ):
        self.redis_host = redis_host or os.getenv("NEF_REDIS_HOST")
        self.redis_port = (
            redis_port
            if redis_port is not None
            else int(os.getenv("NEF_REDIS_PORT"))
        )
        # No hardcoded password; stays None unless provided
        self.redis_password = (
            redis_password
            if redis_password is not None
            else os.getenv("NEF_REDIS_PASSWORD")
        )
        self.verbose = verbose
        self.client = client  # for extractor too
        self.require_redis_grounding = True  # strict
        self.redis_el = RedisEntityLinking(
            host=redis_host, port=int(redis_port), password=redis_password, verbose=verbose
        )
        self.pred = PredicateEmbeddingRetriever(
            client=self.client,
            embeddings_path=embeddings_path,
            predicates_path=predicates_path,
            embed_model="embedding-001",
            verbose=verbose,
        )
        self.llm = LLMDisambiguator(
            client=self.client,
            model_name=llm_model,
            predicate_threshold=predicate_threshold,
            new_predicate_namespace=new_predicate_namespace,
            verbose=verbose,
        )
        if self.verbose:
            _safe_print("✓ Enhanced NEF Pipeline initialized")

    # helpers: lowercase + spacing only
    def _lc_space(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _valid_predicate(self, p: str) -> bool:
        w = re.sub(r"\s+", " ", (p or "").strip()).split()
        return 1 <= len(w) <= 3

    def _resolve_entities(self, mention: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Strict grounding via Redis, except:
          - If mention is a 4-digit year, mint a DBpedia year URI immediately.
        """
        m = (mention or "").strip()
        if _YEAR.fullmatch(m):
            return [(_year_uri(m), 1.0)]  # treat year as an entity, no Redis hit

        cands = self.redis_el.lookup(m, top_k=k)
        fixed: List[Tuple[str, float]] = []
        for uri, score in cands:
            if not (uri.startswith("http://") or uri.startswith("https://")):
                uri = f"http://dbpedia.org/resource/{uri}"
            fixed.append((uri, score))
        return fixed

    def _extract_triples(self, text: str, debug: bool = False) -> list[dict]:
        """
        Strict extractor:
        - forces lowercase S/P/O
        - enforces 1–3 word predicates
        - confidence ≥ 0.5
        - REQUIRES Redis grounding for subject and object
        """
        prompt = f"""
SYSTEM: Return ONLY a valid JSON array (no prose, no markdown fences).

Task: Read the text and extract up to 5 RDF triples with confidence.
You MUST:
- Make subject, predicate, and object ALL lowercase.
- Use the most complete, consistent entity names.
- Resolve clear pronouns (he, she, it, they, this/that, here/there) to the correct entity; if unclear, do not guess.
- Keep predicates extremely concise: 1–3 words max (e.g., "founded", "born in", "wrote").
- Include only items with confidence ≥ 0.5.

Output schema:
[
  {{"subject":"...", "predicate":"...", "object":"...", "confidence":0.0}},
  ...
]

Text:
{text}
""".strip()

        try:
            resp = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )

            if debug:
                _safe_print("\n[DEBUG] Raw model response text:")
                _safe_print((resp.text or "").strip() or "<EMPTY>")

            try:
                items = _json_from_model(resp.text or "[]")
            except Exception as e:
                if debug:
                    _safe_print(f"[DEBUG] JSON parse error: {e}")
                return []

            if debug:
                _safe_print("[DEBUG] Parsed JSON items:",
                            json.dumps(items, indent=2) if isinstance(items, list) else items)

            if not isinstance(items, list):
                if debug:
                    _safe_print("[DEBUG] Parsed payload is not a list → aborting.")
                return []

            out, seen = [], set()
            for idx, it in enumerate(items, 1):
                s_raw = (it.get("subject") or "").strip()
                p_raw = (it.get("predicate") or "").strip()
                o_raw = (it.get("object") or "").strip()
                conf  = it.get("confidence", None)

                try:
                    conf_f = float(conf) if conf is not None else 1.0
                except Exception:
                    conf_f = 0.0

                reasons = []
                if not s_raw or not p_raw or not o_raw:
                    reasons.append("missing field(s)")

                s = self._lc_space(s_raw)
                p = self._lc_space(p_raw)
                o = self._lc_space(o_raw)

                if not self._valid_predicate(p):
                    reasons.append(f"predicate length invalid: '{p}'")

                if conf_f < 0.5:
                    reasons.append(f"confidence < 0.5 (got {conf_f:.3f})")

                sub_cands = self._resolve_entities(s_raw, k=5)
                obj_cands = self._resolve_entities(o_raw, k=5)
                if not sub_cands or not obj_cands:
                    reasons.append("no Redis grounding for subject/object (strict mode)")

                if (s, p, o) in seen:
                    reasons.append("duplicate triple")

                if reasons:
                    if debug:
                        _safe_print(f"[DEBUG] Item #{idx} REJECTED:",
                                    {"subject": s_raw, "predicate": p_raw, "object": o_raw, "confidence": conf},
                                    "→ reasons:", "; ".join(reasons))
                    continue

                seen.add((s, p, o))
                kept = {"subject": s, "predicate": p, "object": o,
                        "_sub_cands": sub_cands, "_obj_cands": obj_cands}
                out.append(kept)

                if debug:
                    _safe_print(f"[DEBUG] Item #{idx} KEPT:",
                                {"subject": s, "predicate": p, "object": o,
                                 "sub_cands": sub_cands[:2], "obj_cands": obj_cands[:2], "confidence": conf_f})

                if len(out) >= 5:
                    break

            if debug and not out:
                _safe_print("[DEBUG] Result: 0 triples kept after filtering.")

            return out

        except Exception as e:
            if getattr(self, "verbose", True) or debug:
                _safe_print(f"✗ Triple extraction error: {e}")
            return []

    def run_pipeline(self, sentence: str, debug: bool = False) -> list[tuple[str, str, str, Dict[str, Any]]]:
        """
        End-to-end for one sentence, strict Redis grounding.
        Returns list of (subjectURI, predicateURI, objectURI, meta)
        """
        if self.verbose:
            _safe_print(f"\n📝 {sentence!r}")

        raw_triples = self._extract_triples(sentence, debug=debug)
        if not raw_triples:
            if self.verbose:
                _safe_print("   ⚠ No triples extracted.")
            return []

        results: list[tuple[str, str, str, Dict[str, Any]]] = []

        for t in raw_triples:
            s_text = t.get("subject", "")
            p_text = t.get("predicate", "")
            o_text = t.get("object", "")
            if not (s_text and p_text and o_text):
                continue

            if self.verbose:
                _safe_print(f"\n🔍 Triple: {s_text} — {p_text} — {o_text}")
                _safe_print("   📍 Using entity candidates collected during extraction...")

            subject_candidates = t.get("_sub_cands") or self._resolve_entities(s_text, k=5)
            object_candidates  = t.get("_obj_cands") or self._resolve_entities(o_text, k=5)

            if self.verbose:
                _safe_print("   [Redis:subject]", subject_candidates[:5] if subject_candidates else "NO CANDIDATES")
                _safe_print("   [Redis:object]",  object_candidates[:5]  if object_candidates  else "NO CANDIDATES")

            if not subject_candidates or not object_candidates:
                if self.verbose:
                    _safe_print("   ⚠ Abandoning triple (no Redis candidates).")
                continue

            predicate_candidates = self.pred.get_top_k_predicates(p_text, top_k=10)
            if self.verbose:
                _safe_print("   [Predicates:top5]", predicate_candidates[:5])

            s_final, p_final, o_final, meta = self.llm.disambiguate_triple(
                sentence, subject_candidates, predicate_candidates, object_candidates
            )
            results.append((s_final, p_final, o_final, meta or {}))

            label = (meta or {}).get("label", "candidate")
            tag_str = "[GENERATED]" if label == "generated" else "[CANDIDATE]"
            sim = (meta or {}).get("chosen_similarity")
            rank = (meta or {}).get("rank_in_topk")
            topk = (meta or {}).get("topk")
            thr = (meta or {}).get("threshold")
            sim_str = f" (sim={sim:.3f})" if isinstance(sim, (int, float)) else ""
            rank_str = f" rank={rank}/{topk}" if (isinstance(rank, int) and isinstance(topk, int)) else ""
            thr_str = f" | thr={thr:.2f}" if isinstance(thr, (int, float)) else ""
            if self.verbose:
                _safe_print("   ✅ Final", f"{tag_str}{sim_str}{rank_str}{thr_str}:",
                            s_final, p_final, o_final, sep="\n            ")

        return results

# =============== CLI ===============

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nef_cli.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Enhanced NEF (strict) — CLI for triple extraction → Redis grounding → predicate disambiguation."
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("-s", "--sentence", nargs="+", help="One or more sentences to process.")
    inp.add_argument("-f", "--file", type=str, help="Path to a UTF-8 text file (one sentence per line).")
    inp.add_argument("--stdin", action="store_true", help="Read sentences from STDIN (one per line).")

    # Output
    p.add_argument("-o", "--output", choices=["json", "jsonl", "tsv", "nt"], default="json",
                   help=textwrap.dedent("""\
                   Output format:
                     json  = single JSON object with 'results' array
                     jsonl = one JSON object per line
                     tsv   = subject\\tpredicate\\tobject
                     nt    = N-Triples (URIs only)
                   """))
    p.add_argument("--no-verbose", dest="verbose", action="store_false", help="Silence progress logs.")
    p.add_argument("--debug", action="store_true", help="Show extractor debug dumps.")
    p.set_defaults(verbose=True)

    # Gemini / models
    p.add_argument("--api-key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY).")
    p.add_argument("--llm-model", type=str, default="gemini-2.5-flash", help="LLM for disambiguation/generation.")
    p.add_argument("--embed-model", type=str, default="embedding-001", help="Embedding model name.")
    p.add_argument("--predicate-threshold", type=float, default=0.5, help="Similarity threshold to accept a predicate.")
    p.add_argument("--new-predicate-namespace", type=str, default="http://nef.local/rel/",
                   help="Namespace for generated predicates.")

    # Predicate embeddings
    p.add_argument("--embeddings", type=str, default=None, help="Path to embeddings.npy")
    p.add_argument("--predicates", type=str, default=None, help="Path to predicates.csv")

    # Redis
    p.add_argument("--redis-host", type=str, default=os.getenv("NEF_REDIS_HOST", ""))
    p.add_argument("--redis-port", type=int, default=int(os.getenv("NEF_REDIS_PORT", "")))
    p.add_argument("--redis-password", type=str, default=os.getenv("NEF_REDIS_PASSWORD", ""))

    return p.parse_args(argv)

def _collect_sentences(args: argparse.Namespace) -> List[str]:
    if args.sentence:
        return [" ".join(args.sentence)] if len(args.sentence) > 1 else [args.sentence[0]]
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if args.stdin:
        return [line.strip() for line in sys.stdin if line.strip()]
    return []

def _emit(results: Dict[str, Any], fmt: str, verbose: bool):
    """results = {'items': [ {'input': str, 'triples':[{'s':..., 'p':..., 'o':..., 'meta':{...}}, ...]}, ... ]}"""
    items = results.get("items", [])

    if fmt == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if fmt == "jsonl":
        for rec in items:
            print(json.dumps(rec, ensure_ascii=False))
        return

    if fmt == "tsv":
        for rec in items:
            for t in rec.get("triples", []):
                print(f"{t['s']}\t{t['p']}\t{t['o']}")
        return

    if fmt == "nt":
        # Only URIs are printed; assume s/p/o are URIs
        for rec in items:
            for t in rec.get("triples", []):
                s, p, o = t["s"], t["p"], t["o"]
                # N-Triples line
                print(f"<{s}> <{p}> <{o}> .")
        return

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        client = _bootstrap_gemini_client(args.api_key)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    # Build pipeline
    try:
        pipe = EnhancedNEFPipeline(
            client=client,
            embeddings_path=args.embeddings,
            predicates_path=args.predicates,
            llm_model=args.llm_model,
            predicate_threshold=args.predicate_threshold,
            new_predicate_namespace=args.new_predicate_namespace,
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_password=args.redis_password,
            verbose=args.verbose,
        )
        # ensure retriever uses requested embed model
        pipe.pred.embed_model = args.embed_model
    except Exception as e:
        sys.stderr.write(f"ERROR initializing pipeline: {e}\n")
        return 3

    sentences = _collect_sentences(args)
    if not sentences:
        sys.stderr.write("No input sentences.\n")
        return 1

    out: Dict[str, Any] = {"items": []}
    had_any = False

    for s in sentences:
        triples = pipe.run_pipeline(s, debug=args.debug)
        norm = [{"s": t[0], "p": t[1], "o": t[2], "meta": t[3]} for t in triples]
        out["items"].append({"input": s, "triples": norm})
        if norm:
            had_any = True

    _emit(out, args.output, args.verbose)
    return 0 if had_any else 4

if __name__ == "__main__":
    # Quick usage help when executed without args in TTY
    if len(sys.argv) == 1 and sys.stdin.isatty():
        _safe_print(textwrap.dedent("""\
            Enhanced NEF CLI

            Examples:
              # Single sentence → JSON
              python nef_cli.py -s "Steve Jobs founded Apple" --embeddings embeddings.npy --predicates predicates.csv

              # Read from file, emit N-Triples
              python nef_cli.py -f sentences.txt --output nt --embeddings embeddings.npy --predicates predicates.csv

              # Read from STDIN (one per line), quiet logs, JSONL
              cat sentences.txt | python nef_cli.py --stdin --no-verbose --output jsonl --embeddings embeddings.npy --predicates predicates.csv

              # Custom Redis and model/threshold
              python nef_cli.py -s "Marie Curie discovered radium" \\
                --redis-host 127.0.0.1 --redis-port 6379 --redis-password secret \\
                --llm-model gemini-2.5-flash --predicate-threshold 0.6 \\
                --embeddings embeddings.npy --predicates predicates.csv
        """))
    sys.exit(main())
