import os, json, re, csv
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import time



def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _infer_schema_keys(sample: Dict[str, Any]) -> Tuple[str, str, str, Optional[str], Optional[str]]:
    """Infer sub/rel/obj keys + text key + id key from a sample row."""
    ks = {k.lower(): k for k in sample.keys()}
    s_key = ks.get("subject") or ks.get("sub") or "subject"
    p_key = ks.get("predicate") or ks.get("rel") or ks.get("relation") or "predicate"
    o_key = ks.get("object") or ks.get("obj") or "object"
    t_key = ks.get("text") or ks.get("sentence") or ks.get("sent") or ks.get("context")
    id_key = ks.get("id") or ks.get("docid") or ks.get("uid")
    return s_key, p_key, o_key, t_key, id_key

def _norm(s: str) -> str:
    """Lowercase + collapse spaces (NEF convention)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _triple_norm(t: Dict[str, Any], s_key: str, p_key: str, o_key: str) -> Tuple[str,str,str]:
    return (_norm(t.get(s_key, t.get("subject",""))),
            _norm(t.get(p_key, t.get("predicate",""))),
            _norm(t.get(o_key, t.get("object",""))))

def _multiset_from(triples: List[Dict[str, Any]], s_key: str, p_key: str, o_key: str) -> Counter:
    return Counter(_triple_norm(t, s_key, p_key, o_key) for t in triples)



def _tokenize(s: str) -> List[str]:
    return [w for w in re.split(r"[^a-z0-9]+", _norm(s)) if w]

def _pred_similarity(a: str, b: str) -> float:
    """
    Lightweight similarity in [0,1] (no synonyms):
      - Exact match → 1.0
      - Token Jaccard
      - Small edit-distance proxy (prefix/suffix overlap)
    """
    a_n, b_n = _norm(a), _norm(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 1.0

    ta, tb = set(_tokenize(a_n)), set(_tokenize(b_n))
    j = len(ta & tb) / max(1, len(ta | tb))

    
    def _overlap(x: str, y: str) -> float:
        k = 0
        for i in range(min(len(x), len(y))):
            if x[i] != y[i]: break
            k += 1
        l = 0
        for i in range(1, min(len(x), len(y))+1):
            if x[-i] != y[-i]: break
            l += 1
        return max(k, l) / max(len(x), len(y), 1)

    ov = _overlap(a_n, b_n)
   
    return max(j, ov)

def align_to_allowed_predicates(
    nef_triples: List[Dict[str, str]],
    allowed_predicates: List[str],
    s_key: str, p_key: str, o_key: str,
    min_accept_score: float = 0.6, 
) -> List[Dict[str, Any]]:
    """
    Map each NEF predicate to the closest item in allowed_predicates, deterministically.
    Drop triple if no candidate meets min_accept_score.
    """
    allowed_norm = [ _norm(p) for p in allowed_predicates if p ]
    out: List[Dict[str, Any]] = []

    for it in nef_triples or []:
        s = _norm(it.get("subject",""))
        p = _norm(it.get("predicate",""))
        o = _norm(it.get("object",""))
        if not (s and p and o):
            continue

   
        best_p, best_sc = None, -1.0
        for ap in allowed_norm:
            sc = _pred_similarity(p, ap)
            if sc > best_sc:
                best_sc, best_p = sc, ap

        if best_p is None or best_sc < min_accept_score:
           
            continue

        out.append({s_key: s, p_key: best_p, o_key: o})

    return out



def _infer_triple_keys_from_sample(gt_rows):
    
    tri_sample = None
    for r in gt_rows:
        ts = r.get("triples")
        if isinstance(ts, list) and ts:
            tri_sample = ts[0]
            break
    if tri_sample is None:
        raise ValueError("No triples found in ground truth file.")


    ks = {k.lower(): k for k in tri_sample.keys()}
    s_key = ks.get("subject") or ks.get("sub") or "subject"
    p_key = ks.get("predicate") or ks.get("rel") or ks.get("relation") or "predicate"
    o_key = ks.get("object") or ks.get("obj") or "object"
    return s_key, p_key, o_key

def evaluate(
    ground_truth_path: str,
    test_path: str,
    run_nef_fn,                          
    output_csv: str = "nef_eval_results.csv",
    verbose: bool = True,
) -> Dict[str, float]:
    gt_rows = _load_jsonl(ground_truth_path)
    ts_rows = _load_jsonl(test_path)
    if not gt_rows or not ts_rows:
        raise ValueError("Ground or test file is empty / not found.")

    _, _, _, t_key_row, id_key_row = _infer_schema_keys(gt_rows[0])
    if not t_key_row:
        raise ValueError("Couldn't infer the text field. Expected one of: 'text','sentence','sent','context'.")
    s_key, p_key, o_key = _infer_triple_keys_from_sample(gt_rows)


    gt_by_text: Dict[str, List[Dict[str, Any]]] = {}
    for r in gt_rows:
        sent = r.get(t_key_row, "")
        if not sent:
            continue
        key = _norm(sent)
        gt_by_text.setdefault(key, []).extend(r.get("triples", []) or [])

    per_rows = []
    total_latency_ms = 0.0

    for r in ts_rows:
        _, _, _, t_key_test, id_key_test = _infer_schema_keys(r)
        sent = r.get(t_key_test or t_key_row)
        ex_id = r.get(id_key_test or id_key_row, "")
        if not sent:
            continue
        k = _norm(sent)
        gt_triples = gt_by_text.get(k, [])

     
        t0 = time.perf_counter()
        nef_raw = run_nef_fn(sent) or []   
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        total_latency_ms += latency_ms
     
        nef_aligned = nef_raw

     
        gt_set  = _multiset_from(gt_triples, s_key, p_key, o_key)
        nef_set = _multiset_from(nef_aligned, s_key, p_key, o_key)
        tp = sum(min(nef_set[t], gt_set.get(t, 0)) for t in nef_set)
        fp = sum(nef_set.values()) - tp
        fn = sum(gt_set.values()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

        if verbose:
            print("="*80)
            print(f"ID: {ex_id or '—'}")
            print("SENTENCE:", sent)
            print("GT triples:", gt_triples)
            print("NEF raw:", nef_raw)
            print(f"latency: {latency_ms:.2f} ms")
            print(f"TP={tp} FP={fp} FN={fn}  |  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        per_rows.append({
            "id": ex_id or "",
            "gt_count":   sum(gt_set.values()),
            "nef_count":  sum(nef_set.values()),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "latency_ms": round(latency_ms, 2),   
        })


    if per_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            w.writeheader()
            w.writerows(per_rows)

    macro_p = sum(r["precision"] for r in per_rows)/len(per_rows) if per_rows else 0.0
    macro_r = sum(r["recall"]    for r in per_rows)/len(per_rows) if per_rows else 0.0
    macro_f = 2*macro_p*macro_r/(macro_p+macro_r) if (macro_p+macro_r)>0 else 0.0
    avg_latency_ms = (total_latency_ms / len(per_rows)) if per_rows else 0.0 

    print("\n===== SUMMARY =====")
    print(f"Examples:        {len(per_rows)}")
    print(f"Macro-Precision: {macro_p:.4f}")
    print(f"Macro-Recall:    {macro_r:.4f}")
    print(f"Macro-F1:        {macro_f:.4f}")
    print(f"Avg Latency:     {avg_latency_ms:.2f} ms") 
    print(f"Saved per-example metrics → {output_csv}")

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "avg_latency_ms": avg_latency_ms,  
    }


import re, os, json


nef = EnhancedNEFPipeline(verbose=True)  


def _last_seg(u: str) -> str:
 
    if not u: return ""
    x = re.split(r"[#?/]", u.strip())[-1]
    return x

def _split_camel_unders(x: str) -> str:

    x = re.sub(r"[_\-\.]+", " ", x)
    x = re.sub(r"(?<=[a-z0-9])([A-Z])", r" \1", x)
    return re.sub(r"\s+", " ", x).strip().lower()

def _uri_to_text(u: str) -> str:
    return _split_camel_unders(_last_seg(u))

def _pred_uri_to_text(p: str) -> str:
 
    if p.startswith("http://") or p.startswith("https://"):
        return _split_camel_unders(_last_seg(p))
    return _split_camel_unders(p)


def run_nef_fn(text: str):
    """
    Calls NEF, returns a list of {"subject","predicate","object"} as lowercase phrases
    (derived from NEF URIs). This matches the evaluator’s expected shape.
    """
    triples = nef.run_pipeline(text)  
    out = []
    for s_uri, p_uri, o_uri in triples:
        out.append({
            "subject":   _uri_to_text(s_uri),
            "predicate": _pred_uri_to_text(p_uri),
            "object":    _uri_to_text(o_uri),
        })
    return out


GROUND = "./ground_truth/nef_custom_ground_truth_v9.jsonl"  
TEST   = "./test/nef_custom_test_v9.jsonl"



metrics = evaluate(
    ground_truth_path=GROUND,
    test_path=TEST,
    run_nef_fn=run_nef_fn,               
    output_csv="nef_eval_results.csv",
    verbose=True,
)
print("\nFinal metrics:", metrics)

