# ── PER-SENTENCE CIRCUIT BREAKER ──────────────────────────────────────────────
# Run this AFTER autonomous_pipeline_v13 and run_text2kg are defined.
# It wraps extract_all_triples so a single hung sentence is abandoned after
# HARD_LIMIT seconds and the run continues, instead of freezing forever.
#
# Why needed: request_timeout caps ONE llm call, but a hung OpenRouter connection
# that accepts but never streams can slip past it, and a sentence fires many calls.
# This caps the WHOLE sentence with a hard wall-clock limit.
import threading

_HARD_LIMIT = 180   # seconds per sentence; generous — normal is 30-40s

# IDEMPOTENT — safe to re-run this cell any number of times.
# Without this guard, re-running would wrap the WRAPPER (nested 180s timeouts).
# NOTE: re-paste the PIPELINE cell first, then this one, so we capture the NEW
# extract_all_triples. If this cell runs twice in a row we keep the original.
_candidate = extract_all_triples
if getattr(_candidate, "__name__", "") == "extract_all_triples_safe":
    _candidate = _orig_extract_all_triples      # already wrapped: don't re-wrap
_orig_extract_all_triples = _candidate          # the REAL pipeline function

def extract_all_triples_safe(sentence, allowed_predicates=None,
                             literal_predicates=None, predicate_ranges=None,
                             quote_hints=None, verbose=False):
    result = {"sentence": sentence, "processed": sentence,
              "extracted": [], "results": [], "triples": []}
    done = threading.Event()
    box = {}

    def _work():
        try:
            box["r"] = _orig_extract_all_triples(
                sentence, allowed_predicates=allowed_predicates,
                literal_predicates=literal_predicates,
                predicate_ranges=predicate_ranges, quote_hints=quote_hints,
                verbose=verbose)
        except Exception as e:
            box["err"] = e
        finally:
            done.set()

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    finished = done.wait(timeout=_HARD_LIMIT)

    if not finished:
        print(f"   TIMEOUT (>{_HARD_LIMIT}s) — skipping sentence, run continues.")
        print(f"       Sentence: {sentence[:70]}")
        return result            # empty triples; scored as a miss, not a hang
    if "err" in box:
        print(f"   Sentence errored: {str(box['err'])[:80]} — skipping.")
        return result
    return box.get("r", result)

# Point the runner at the safe version.
extract_all_triples = extract_all_triples_safe
print(f"Per-sentence circuit breaker active (hard limit {_HARD_LIMIT}s).")
