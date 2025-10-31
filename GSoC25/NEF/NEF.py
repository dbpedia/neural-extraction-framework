import os, json, re, time, sys
import numpy as np
from typing import List, Tuple, Sequence, Dict, Any, Optional
from urllib.parse import quote

from getpass import getpass
from google import genai
from google.genai import types

# --- API key bootstrap ---
if not os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass("Enter your Google (Gemini) API key: ").strip()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# =============== Utils ===============

def _normalize(vec: Sequence[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1e-12
    return v / n

def _json_from_model(text: str) -> Any:
    """Extract the first JSON object/array from a model string."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    m = re.search(r"\{.*\}|\[.*\]", t, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(0))

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        # Avoid crashing on weird unicode in some environments
        sys.stdout.write((" ".join(map(str, args)) + "\n").encode("utf-8", "ignore").decode("utf-8"))

# =============== Redis Entity Linking ===============

class RedisEntityLinking:
    """Redis-based entity linking; degrades gracefully if Redis is unavailable."""
    def __init__(
        self,
        host: str = os.getenv("NEF_REDIS_HOST", "91.99.92.217"),
        port: int = int(os.getenv("NEF_REDIS_PORT", "6379")),
        password: str = os.getenv("NEF_REDIS_PASSWORD", "NEF!gsoc2025"),
        connect_timeout: float = 2.0,
    ):
        self.available = False
        self.redis_forms = None
        self.redis_redir = None
        try:
            import redis  # local import so code runs even if redis isn't installed
            common = dict(host=host, port=port, password=password, socket_connect_timeout=connect_timeout,
                          decode_responses=True)
            self.redis_forms = redis.Redis(db=0, **common)
            self.redis_redir  = redis.Redis(db=1, **common)
            self.available = bool(self.redis_forms.ping() and self.redis_redir.ping())
            _safe_print("✓ Connected to Redis" if self.available else "✗ Redis ping failed")
        except Exception as e:
            _safe_print(f"✗ Redis connection error (continuing with fallbacks): {e}")

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
        Returns list of (entity_uri, score). Score is normalized by max support.
        """
        if not self.available or not surface_form.strip():
            return []
        raw = self.redis_forms.hgetall(surface_form)
        if not raw:
            return []
        # aggregate by redirected canonical URI
        counts: Dict[str, int] = {}
        for k, v in raw.items():
            uri = self._redirect(k)
            counts[uri] = counts.get(uri, 0) + int(v)
        if not counts:
            return []
        max_support = max(counts.values()) or 1
        items = [(uri, c / max_support) for uri, c in counts.items() if (c / max_support) >= thr]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_k]

# =============== Predicate Retriever (precomputed) ===============

class PredicateEmbeddingRetriever:
    """
    Loads embeddings.npy (N, D) and predicates.csv (URIs line-by-line or CSV with 'predicate' column),
    then retrieves top-K predicates for a relation text using cosine similarity.
    Optional synonym expansion via Gemini.
    """
    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        use_llm_synonyms: bool = True,
        max_synonyms: int = 6,
        llm_model_for_synonyms: str = "gemini-2.5-flash",
        embed_model: str = "embedding-001",
        verbose: bool = True,
    ):
        self.client = client
        self.embed_model = embed_model
        self.use_llm_synonyms = use_llm_synonyms
        self.max_synonyms = max_synonyms
        self.llm_model_for_synonyms = llm_model_for_synonyms
        self.verbose = verbose

        emb_path, pred_path = self._find_files(embeddings_path, predicates_path)
        self.E: np.ndarray = np.load(emb_path)  # (N, D)
        self.predicates: List[str] = self._load_predicates(pred_path)
        if self.E.shape[0] != len(self.predicates):
            raise ValueError(f"Row count mismatch: embeddings ({self.E.shape[0]}) vs predicates ({len(self.predicates)})")

        self.D = int(self.E.shape[1])
        self.E_norm = self.E / (np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-12)
        self._syn_cache: Dict[Tuple[str, int], List[str]] = {}

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
        """Accepts either CSV with header including 'predicate' or raw newline list."""
        preds: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
            if "predicate" in head:
                # CSV with header
                for line in f:
                    parts = line.rstrip("\n").split(",")
                    if parts:
                        preds.append(parts[0] if head.lower().startswith("predicate") else parts[-1])
            else:
                # treat first line as data too
                preds.append(head.strip())
                for line in f:
                    preds.append(line.strip())
        # clean empties
        preds = [p for p in preds if p]
        return preds

    def _synonyms_for(self, relation_text: str) -> List[str]:
        if not self.use_llm_synonyms:
            return [relation_text]
        key = (relation_text.strip().lower(), self.max_synonyms)
        if key in self._syn_cache:
            return self._syn_cache[key]

        prompt = (
            f"Generate {self.max_synonyms} short synonyms or verb-phrase paraphrases for a knowledge-graph relation.\n"
            f'Return ONLY a JSON array of strings.\n\nRelation: "{relation_text}"'
        )
        out: List[str] = []
        try:
            resp = self.client.models.generate_content(
                model=self.llm_model_for_synonyms,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            arr = _json_from_model(resp.text or "[]")
            if isinstance(arr, list):
                seen = set()
                for s in arr:
                    s = str(s).strip()
                    if not s:
                        continue
                    k = s.lower()
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(s)
                    if len(out) >= self.max_synonyms:
                        break
        except Exception as e:
            if self.verbose:
                _safe_print(f"[synonyms] LLM failed for '{relation_text}': {e}")

        if not out:
            base = relation_text.strip()
            out = [base]
        self._syn_cache[key] = out
        if self.verbose:
            _safe_print(f"↪ Synonyms for '{relation_text}': {out}")
        return out

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        cfg = types.EmbedContentConfig(output_dimensionality=int(self.D))
        vecs: List[np.ndarray] = []
        for t in texts:
            resp = self.client.models.embed_content(model=self.embed_model, contents=t, config=cfg)
            v = getattr(resp, "embeddings", None)
            v = (v[0].values if v else resp.embedding.values)
            vecs.append(_normalize(v))
        return np.stack(vecs, axis=0)

    def get_top_k_predicates(self, relation_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        texts = self._synonyms_for(relation_text)
        Q = self._embed_texts(texts)     # (S, D)
        q = _normalize(Q.mean(axis=0))   # (D,)
        sims = self.E_norm @ q           # (N,)
        order = sims.argsort()[-top_k:][::-1]
        return [(self.predicates[i], float(sims[i])) for i in order]

# =============== LLM Disambiguator ===============

class LLMDisambiguator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str = "http://nef.local/rel/",
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.thr = float(predicate_threshold)
        self.ns = new_predicate_namespace.rstrip("/") + "/"
        self.verbose = verbose

    def _camelize(self, s: str) -> str:
        import re
        s = re.sub(r"[^A-Za-z0-9\s]", " ", s).strip()
        if not s: return "relatedTo"
        parts = re.split(r"\s+", s)
        out = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
        if out[0].isdigit(): out = "rel" + out
        return out[:80]

    def _mint_uri(self, local: str) -> str:
        return self.ns + self._camelize(local)

class LLMDisambiguator:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        predicate_threshold: float = 0.5,
        new_predicate_namespace: str = "http://nef.local/rel/",
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.thr = float(predicate_threshold)
        self.ns = new_predicate_namespace.rstrip("/") + "/"
        self.verbose = verbose

    def _camelize(self, s: str) -> str:
        import re
        s = re.sub(r"[^A-Za-z0-9\s]", " ", s).strip()
        if not s: return "relatedTo"
        parts = re.split(r"\s+", s)
        out = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
        if out[0].isdigit(): out = "rel" + out
        return out[:80]

    def _mint_uri(self, local: str) -> str:
        return self.ns + self._camelize(local)

    def disambiguate_triple(
        self,
        context: str,
        subject_candidates,      # List[(uri, score)]
        predicate_candidates,    # List[(uri, sim)] sorted desc
        object_candidates,       # List[(uri, score)]
    ):
        total_k = len(predicate_candidates)
        sim_map = {u: (s, i) for i, (u, s) in enumerate(predicate_candidates)}

        # Split by threshold
        above = [(u, s) for (u, s) in predicate_candidates if s >= self.thr]

        if above:
            allowed = [u for u, _ in above]
            pred_list_text = "\n".join([f"- {u}" for u in allowed])
            prompt = f"""Pick the best RDF triple using ONLY these predicate URIs.

Allowed predicates:
{pred_list_text}

Context:
{context}

Return ONLY JSON: {{"subject":"URI","predicate":"URI","object":"URI"}}
"""
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                data = _json_from_model(resp.text or "{}")
                pred_uri = data.get("predicate", "")
                if pred_uri not in allowed:
                    # Force top allowed if the model drifts
                    pred_uri = allowed[0]

                chosen_sim, rank0 = sim_map.get(pred_uri, (None, None))
                meta = {
                    "label": "candidate",
                    "chosen_similarity": float(chosen_sim) if chosen_sim is not None else None,
                    "rank_in_topk": (rank0 + 1) if rank0 is not None else None,
                    "topk": total_k,
                    "threshold": self.thr,
                }
                return (
                    data.get("subject", subject_candidates[0][0] if subject_candidates else ""),
                    pred_uri,
                    data.get("object", object_candidates[0][0] if object_candidates else ""),
                    meta,
                )
            except Exception as e:
                if self.verbose:
                    print(f"✗ disambiguation error; using top allowed: {e}")
                best_uri, best_sim = max(above, key=lambda x: x[1])
                _, rank0 = sim_map.get(best_uri, (best_sim, 0))
                meta = {
                    "label": "candidate",
                    "chosen_similarity": float(best_sim),
                    "rank_in_topk": (rank0 + 1),
                    "topk": total_k,
                    "threshold": self.thr,
                }
                return (
                    subject_candidates[0][0] if subject_candidates else "",
                    best_uri,
                    object_candidates[0][0] if object_candidates else "",
                    meta,
                )

        # None ≥ threshold → generate
        prompt = f"""No predicate meets the threshold ({self.thr:.2f}).
Propose a NEW concise camelCase predicate name for this relation in context.
Return ONLY JSON: {{"predicateLocalName":"camelCase"}}.

Context:
{context}
"""
        try:
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            data = _json_from_model(resp.text or "{}")
            local = (data.get("predicateLocalName") or "relatedTo").strip()
            pred_uri = self._mint_uri(local)
            meta = {
                "label": "generated",
                "chosen_similarity": None,
                "rank_in_topk": None,
                "topk": total_k,
                "threshold": self.thr,
            }
            return (
                subject_candidates[0][0] if subject_candidates else "",
                pred_uri,
                object_candidates[0][0] if object_candidates else "",
                meta,
            )
        except Exception as e:
            if self.verbose:
                print(f"✗ generation error; minting default: {e}")
            pred_uri = self._mint_uri("relatedTo")
            meta = {
                "label": "generated",
                "chosen_similarity": None,
                "rank_in_topk": None,
                "topk": total_k,
                "threshold": self.thr,
            }
            return (
                subject_candidates[0][0] if subject_candidates else "",
                pred_uri,
                object_candidates[0][0] if object_candidates else "",
                meta,
            )


# =============== Orchestrator ===============

class EnhancedNEFPipeline:
    """
    End-to-end pipeline:
      1) Extract triples with Gemini
      2) Redis entity linking (subject/object)
      3) Predicate retrieval via precomputed embeddings
      4) LLM disambiguation
    """
    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        predicates_path: Optional[str] = None,
        llm_model: str = "gemini-2.5-flash",
        use_llm_synonyms: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.redis_el = RedisEntityLinking()
        self.pred = PredicateEmbeddingRetriever(
            embeddings_path=embeddings_path,
            predicates_path=predicates_path,
            use_llm_synonyms=use_llm_synonyms,
            max_synonyms=6,
            llm_model_for_synonyms=llm_model,
            embed_model="embedding-001",
            verbose=verbose,
        )
        self.llm = LLMDisambiguator(llm_model)
        if self.verbose:
            _safe_print("✓ Enhanced NEF Pipeline initialized")

    def _extract_triples(self, text: str) -> list[dict]:
        """
        Simple extractor: asks for up to 5 triples with confidence,
        then filters: confidence≥0.5, looks-like-named, Redis-groundable.
        """
        prompt = f"""
    Return up to 5 RDF triples from the text as ONLY JSON:
    [{{"subject":"...", "predicate":"...", "object":"...", "confidence":0.0}}, ...]

    Rules:
    - subject & object: exact surface forms from the text, concrete named entities (no pronouns/generic phrases).
    - predicate: short verb/verb-phrase from the text.
    - confidence in [0,1]. Only include triples you are ≥0.5 confident in.

    Text: {text}
    """.strip()

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            items = _json_from_model(resp.text or "[]")
            if not isinstance(items, list):
                return []

            def _looks_named(x: str) -> bool:
                toks = x.split()
                return any(t and (t[0].isupper() or t.isupper()) for t in toks)

            out, seen = [], set()
            for it in items:
                s = (it.get("subject") or "").strip()
                p = (it.get("predicate") or "").strip()
                o = (it.get("object") or "").strip()
                conf = float(it.get("confidence") or 0.0)

                if not (s and p and o):                   # all fields present
                    continue
                if conf < 0.5:                            # confidence gate
                    continue
                if not (_looks_named(s) and _looks_named(o)):  # avoid generic endpoints
                    continue

                # Redis grounding (no generation)
                if not self._resolve_entities(s, k=1) or not self._resolve_entities(o, k=1):
                    continue

                key = (s, p, o)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"subject": s, "predicate": p, "object": o})
                if len(out) >= 5:
                    break

            return out

        except Exception as e:
            if getattr(self, "verbose", True):
                _safe_print(f"✗ Triple extraction error: {e}")
            return []



    def _resolve_entities(self, mention: str, k: int = 5) -> List[Tuple[str, float]]:
        cands = self.redis_el.lookup(mention, top_k=k)
        # ensure URIs
        fixed: List[Tuple[str, float]] = []
        for uri, score in cands:
            if not (uri.startswith("http://") or uri.startswith("https://")):
                uri = f"http://dbpedia.org/resource/{quote(uri)}"
            fixed.append((uri, score))
        return fixed

    def run_pipeline(self, sentence: str) -> list[tuple[str, str, str]]:
        """
        End-to-end for one sentence.
        Prints minimal tag [CANDIDATE] or [GENERATED] plus sim, rank/topk, thr.
        """
        if self.verbose:
            _safe_print(f"\n📝 {sentence!r}")

        raw_triples = self._extract_triples(sentence)
        if not raw_triples:
            if self.verbose:
                _safe_print("   ⚠ No triples extracted.")
            return []

        results: list[tuple[str, str, str]] = []

        for t in raw_triples:
            s = (t.get("subject") or "").strip()
            p = (t.get("predicate") or "").strip()
            o = (t.get("object") or "").strip()
            if not (s and p and o):
                continue

            if self.verbose:
                _safe_print(f"\n🔍 Triple: {s} — {p} — {o}")
                _safe_print("   📍 Getting entity candidates from Redis...")

            subject_candidates = self._resolve_entities(s, k=5)
            object_candidates  = self._resolve_entities(o, k=5)

            if self.verbose:
                _safe_print("   [Redis:subject]", subject_candidates[:5] if subject_candidates else "NO CANDIDATES")
                _safe_print("   [Redis:object]",  object_candidates[:5]  if object_candidates  else "NO CANDIDATES")

            if not subject_candidates or not object_candidates:
                if self.verbose:
                    _safe_print("   ⚠ Abandoning triple (no Redis candidates for subject/object).")
                continue

            predicate_candidates = self.pred.get_top_k_predicates(p, top_k=10)
            if self.verbose:
                _safe_print("   [Predicates:top5]", predicate_candidates[:5])

            s_final, p_final, o_final, meta = self.llm.disambiguate_triple(
                sentence, subject_candidates, predicate_candidates, object_candidates
            )
            results.append((s_final, p_final, o_final))

            # Minimal tag but rich metrics
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
                _safe_print("   ✅ Final", f"{tag_str}{sim_str}{rank_str}{thr_str}:", s_final, p_final, o_final, sep="\n            ")

        return results


if __name__ == "__main__":
    import argparse, json, sys, os

    p = argparse.ArgumentParser(
        description="Run EnhancedNEFPipeline from the command line."
    )
    p.add_argument("text", nargs="?", help="Sentence to extract triples from. If omitted, read from stdin.")
    p.add_argument("--embeddings", "-e", default=None, help="Path to embeddings.npy")
    p.add_argument("--predicates", "-p", default=None, help="Path to predicates.csv (has a 'predicate' column or 1-col CSV)")
    p.add_argument("--model", "-m", default="gemini-2.5-flash", help="LLM model name")
    p.add_argument("--thr", type=float, default=0.5, help="Predicate threshold (0–1)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose logging")
    p.add_argument("--json", action="store_true", help="Print JSON triples (default prints pretty text)")
    args = p.parse_args()

    # Text source: CLI arg or stdin
    if args.text is not None:
        sentence = args.text.strip()
    else:
        sentence = sys.stdin.read().strip()

    if not sentence:
        sys.stderr.write("No input text provided (arg or stdin).\n")
        sys.exit(2)

    # Ensure API key is present (either env var or prompt via your existing bootstrap)
    if not os.getenv("GEMINI_API_KEY"):
        # your code at the top already prompts via getpass() if missing, so nothing else to do here
        pass

    try:
        pipe = EnhancedNEFPipeline(
            embeddings_path=args.embeddings,
            predicates_path=args.predicates,
            llm_model=args.model,
            verbose=(not args.quiet),
        )
        # also set the disambiguator threshold from flag
        pipe.llm.thr = float(args.thr)

        triples = pipe.run_pipeline(sentence)

        if args.json:
            # JSON array of [subject, predicate, object]
            print(json.dumps(triples, ensure_ascii=False, indent=2))
        else:
            if not triples:
                print("(no triples)")
            else:
                for (s, p_uri, o) in triples:
                    print(f"S: {s}\nP: {p_uri}\nO: {o}\n---")
        sys.exit(0 if triples else 1)
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\n")
        sys.stderr.write("Tip: make sure embeddings.npy and predicates.csv are in the working directory or pass --embeddings/--predicates.\n")
        sys.exit(3)
    except Exception as e:
        # fall back error
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(4)
