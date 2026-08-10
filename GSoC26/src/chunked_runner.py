# ── CHUNKED RUNNER — keeps every burst inside the fast API window ──────────────
# Run AFTER pipeline + breaker + robust cells.
#
# Observation: a run starts fast (~70-90s/sent) for the first ~25 sentences, then
# the SPARQL/OpenRouter connection pools accumulate load and everything crawls to
# 150-200s with rising timeouts. The fast window is real but short-lived.
#
# This runner processes a domain in small CHUNKS, and between chunks it does a hard
# COOLDOWN + connection reset so each chunk starts near-fresh — mimicking a kernel
# restart WITHOUT losing loaded state. It also SAVES after every chunk, so a wedge
# never costs more than one chunk.
#
# It does NOT need kernel restarts and is fully automated.
import time, json, gc
from text2kg_harness import (load_ontology, load_test, load_ground_truth,
                             score_sentence, learn_quote_conventions)
try:
    from text2kg_harness import learn_literal_predicates
except ImportError:
    learn_literal_predicates = lambda s: set()

def _reset_connections():
    """Force-close pooled HTTP/SPARQL sockets so the next chunk starts fresh.
    This is what a kernel restart does implicitly; we do it explicitly."""
    gc.collect()
    try:
        import requests
        # close any module-level sessions the pipeline may hold
        for mod in list(globals().values()):
            s = getattr(mod, "session", None)
            if isinstance(s, requests.Session):
                s.close()
    except Exception:
        pass
    gc.collect()

def chunked_domain(slug, chunk_size=22, chunk_cooldown=45, save_every_chunk=True):
    onto = load_ontology(slug); tests = load_test(slug); gold = load_ground_truth(slug)
    allowed = onto["pids"]
    _LR = {"number","string","date","year"}
    literal = {p for p,r in onto["relations"].items() if str(r.get("range","")).lower() in _LR}
    literal = literal | learn_literal_predicates(slug)
    ranges  = {p: str(r.get("range","")) for p,r in onto["relations"].items()}
    qhints  = learn_quote_conventions(slug)
    try:
        from text2kg_harness import learn_predicate_aliases
        aliases = learn_predicate_aliases(slug)
    except ImportError:
        aliases = {}
    try:
        from text2kg_harness import learn_value_formats
        vformats = learn_value_formats(slug)
    except ImportError:
        vformats = {}
    try:
        from text2kg_harness import retrieve_examples
    except ImportError:
        retrieve_examples = None

    results = {}
    fname = f"results_{slug}.json"
    n = len(tests)
    n_chunks = (n + chunk_size - 1) // chunk_size
    print(f"\n{'#'*66}\n# {slug}: {n} sentences in {n_chunks} chunks of {chunk_size}")
    print(f"# cooldown {chunk_cooldown}s between chunks to reset the API window")
    print(f"{'#'*66}")

    for ci in range(n_chunks):
        chunk = tests[ci*chunk_size : (ci+1)*chunk_size]
        print(f"\n─── CHUNK {ci+1}/{n_chunks}  (sentences {ci*chunk_size+1}-{ci*chunk_size+len(chunk)}) ───", flush=True)
        t_chunk = time.time()
        for j, item in enumerate(chunk, 1):
            sid, sent = item["id"], item["sent"]
            gt = gold.get(sid, [])
            t0 = time.time()
            try:
                fewshot = retrieve_examples(slug, sent) if retrieve_examples else None
                out = extract_all_triples(sent, allowed_predicates=allowed,
                                          literal_predicates=literal,
                                          predicate_ranges=ranges,
                                          quote_hints=qhints,
                                          predicate_aliases=aliases,
                                          value_formats=vformats,
                                          fewshot_examples=fewshot, verbose=False)
                pred = out["triples"]
            except Exception:
                pred = []
            p, r, f = score_sentence(pred, gt)
            results[sid] = {"id":sid,"sent":sent,"gold":gt,"pred":pred,"p":p,"r":r,"f1":f}
            dt = time.time() - t0
            print(f"   [{ci*chunk_size+j:3d}/{n}] F1={f:.2f} ({dt:.0f}s) {sent[:38]}", flush=True)
            time.sleep(0.5)
        # save after each chunk
        if save_every_chunk:
            rows_so_far = [results[t["id"]] for t in tests if t["id"] in results]
            json.dump(rows_so_far, open(fname, "w"), indent=2)
        done = len(results)
        mf = sum(x["f1"] for x in results.values())/done
        chunk_avg = (time.time()-t_chunk)/len(chunk)
        print(f"   chunk done in {chunk_avg:.0f}s/sent | running F1={mf:.4f} | {done}/{n} saved", flush=True)
        # cooldown + connection reset before next chunk (skip after last)
        if ci < n_chunks - 1:
            if chunk_avg > 110:
                cd = chunk_cooldown * 2   # window was already degraded, cool longer
                print(f"   ⚠️ chunk was slow ({chunk_avg:.0f}s/sent) — extending cooldown to {cd}s", flush=True)
            else:
                cd = chunk_cooldown
            _reset_connections()
            print(f"   ❄️ cooldown {cd}s (resetting API window)...", flush=True)
            time.sleep(cd)

    rows = [results[t["id"]] for t in tests if t["id"] in results]
    P = sum(x["p"] for x in rows)/len(rows)
    R = sum(x["r"] for x in rows)/len(rows)
    F = sum(x["f1"] for x in rows)/len(rows)
    z = sum(1 for x in rows if x["f1"]==0)
    json.dump(rows, open(fname, "w"), indent=2)
    print(f"\n>>> {slug}: P={P:.4f} R={R:.4f} F1={F:.4f}  zeros={z}  (chunked, saved to {fname})")
    return P, R, F, rows

print("chunked_domain(slug, chunk_size=22, chunk_cooldown=45) ready.")
print("Processes in bursts inside the fast window, cooling down between to reset it.")
