"""
prepare_data.py — Build the final Experiment 1 training and validation sets.

Combines three sources into two output files:

  Training set (~/exp1_train_combined.jsonl):
    - Aditya's 20K (all scores) + Noisy 15K, already in slug format
      (phase1_training_data.jsonl)
    - Wikipedia sentences scored < 9 (converted to slug format here)

  Validation set (~/exp1_val_wikipedia_ge9.jsonl):
    - Wikipedia sentences scored >= 9, with coreference resolution applied:
        * no pronoun in a core triplet         -> kept as-is
        * pronoun successfully resolved         -> rewritten sentence +
                                                    freshly re-extracted
                                                    triples used instead
        * pronoun self-contained (demonstrative -> kept as-is; "यह शहर"
          + noun, e.g. "यह शहर", "वह मन्दिर")      / "वह मन्दिर" is a valid
                                                    exact-span subject
                                                    regardless of the
                                                    entity's real name
        * pronoun genuinely ambiguous            -> EXCLUDED (a bare
          (उन्होंने, उसका, इसे... with nothing      "उनका" with no named
           to anchor it)                           referent isn't a
                                                    reliable evaluation
                                                    signal)

Formatting fix applied throughout:
  Triplets with an empty or missing object (common for intransitive Hindi
  verbs, e.g. "सदस्यों | बहिष्कार किया | NONE") are normalized to the
  literal string "NONE" so every slug line stays well-formed.

Schema fix applied at write time:
  Entries coming from different upstream sources have slightly different
  optional fields (e.g. "score" only exists on Wikipedia-derived entries,
  "phase" only on the original 20K/15K entries). HuggingFace's `datasets`
  library infers one fixed schema from a JSONL file and errors out the
  moment a later row has a different key set than the rows it inferred
  from. normalize_schema() gives every entry in a file the same set of
  keys (filling missing ones with null) so this never happens.

Run:
    python3 prepare_data.py

Requires (home directory):
    ~/phase1_training_data.jsonl
    ~/wiki_chunk_0_scored.jsonl, wiki_chunk_1_scored.jsonl, wiki_chunk_2_scored.jsonl
    ~/wiki_val_coref_resolved.jsonl   (optional — see note below)

If wiki_val_coref_resolved.jsonl is missing or incomplete, every flagged
sentence with a genuinely ambiguous pronoun is treated as unresolved
(excluded) rather than the script failing — re-run this script after
coreference resolution finishes to pick up the larger, cleaner validation
set.
"""

import json
import os
import time

HOME = os.path.expanduser("~")

PHASE1_FILE = os.path.join(HOME, "phase1_training_data.jsonl")
WIKI_CHUNKS = [os.path.join(HOME, f"wiki_chunk_{i}_scored.jsonl") for i in [0, 1, 2]]
COREF_FILE  = os.path.join(HOME, "wiki_val_coref_resolved.jsonl")

TRAIN_OUTPUT = os.path.join(HOME, "exp1_train_combined.jsonl")
VAL_OUTPUT   = os.path.join(HOME, "exp1_val_wikipedia_ge9.jsonl")

VALIDATION_SCORE_THRESHOLD = 9

OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""

COT_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Think step by step, then provide the triplets.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""


# ── Pronoun classification (for entries coref_resolve.py couldn't fix) ──

AMBIGUOUS_PRONOUNS = [
    "उन्होंने", "उसने", "इन्होंने", "इसे", "उसे",
    "उसका", "उसकी", "उसके", "इसका", "इसकी", "इसके",
    "उनका", "उनकी", "उनके",
]
DEMONSTRATIVE_PRONOUNS = ["यह", "वह", "ये", "वे"]


def pronoun_verdict(text):
    """
    Classify a single subject/object span.

    "ambiguous"      a bare agentive/objective/possessive pronoun
                      (उन्होंने, उसका, इसे, ...), or a demonstrative with
                      nothing attached (bare "यह"/"वह") -- no way to know
                      who/what it refers to from this sentence alone.
    "self_contained" a demonstrative pronoun directly modifying a noun
                      (यह शहर, वह मन्दिर, ...) -- the exact span IS the
                      correct triplet regardless of the real-world name
                      behind "this city" / "that temple".
    "none"            no pronoun issue.
    """
    text = (text or "").strip()
    for p in AMBIGUOUS_PRONOUNS:
        if text == p or text.startswith(p + " "):
            return "ambiguous"
    for p in DEMONSTRATIVE_PRONOUNS:
        if text == p:
            return "ambiguous"
        if text.startswith(p + " "):
            return "self_contained"
    return "none"


def entry_pronoun_status(entry):
    """
    Scan every core (non-property) triplet in an entry. If any triplet
    contains a genuinely ambiguous pronoun span, the whole entry is
    ambiguous; otherwise it's self-contained.
    """
    content = json.loads(entry["messages"][2]["content"])
    core = [t for t in content.get("extracted_triplets", [])
            if isinstance(t, dict) and t.get("relation", "").strip() != "property"]

    for t in core:
        for span in (t.get("subject"), t.get("object")):
            if pronoun_verdict(span) == "ambiguous":
                return "ambiguous"
    return "self_contained"


# ── Slug conversion ──────────────────────────────────────────

def clean_span(value):
    if value is None:
        return ""
    return value.strip()


def triplets_to_slug(triplets):
    """
    Convert triplet dicts to pipe-separated slug lines.

    Subject and relation must be non-empty or the triplet is dropped
    (a genuine upstream corruption). An empty/None object is normalized
    to the literal string "NONE" rather than dropped, since many correct
    Hindi triplets use intransitive verbs with no grammatical object.

    Non-dict entries (e.g. a malformed model response that returned a
    bare string instead of a {subject, relation, object} object) are
    skipped rather than raised, since a single corrupted triplet should
    never crash processing of the other thousands of entries.
    """
    if not triplets:
        return "NONE"
    lines = []
    for t in triplets:
        if not isinstance(t, dict):
            continue
        s = clean_span(t.get("subject"))
        r = clean_span(t.get("relation"))
        o = clean_span(t.get("object"))
        if not s or not r:
            continue
        if not o:
            o = "NONE"
        lines.append(f"{s} | {r} | {o}")
    return "\n".join(lines) if lines else "NONE"


def build_traces(sentence, triplets, thought, source, score=None):
    slug_ans = triplets_to_slug(triplets)

    optimal = {
        "messages": [
            {"role": "system", "content": OPTIMAL_INSTRUCTION},
            {"role": "user", "content": sentence},
            {"role": "assistant", "content": slug_ans},
        ],
        "source": source,
        "trace_type": "optimal",
    }
    cot = {
        "messages": [
            {"role": "system", "content": COT_INSTRUCTION},
            {"role": "user", "content": sentence},
            {"role": "assistant", "content": f"[REASONING]\n{thought}\n\n[ANSWER]\n{slug_ans}"},
        ],
        "source": source,
        "trace_type": "cot",
    }
    if score is not None:
        optimal["score"] = score
        cot["score"] = score
    return optimal, cot


def convert_entries(entries, source_label):
    traces = []
    skipped = 0
    for e in entries:
        sentence = e["messages"][1]["content"].strip()
        try:
            content = json.loads(e["messages"][2]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            skipped += 1
            continue
        thought = content.get("thought_process", "")
        triplets = content.get("extracted_triplets", [])
        score = e.get("judgement", {}).get("score")
        optimal, cot = build_traces(sentence, triplets, thought, source_label, score)
        traces.append(optimal)
        traces.append(cot)
    if skipped:
        print(f"    ({skipped} entries skipped — malformed assistant content)")
    return traces


def verify_slug(entry):
    """Self-check: confirm a trace is well-formed before we trust it."""
    content = entry["messages"][2]["content"]
    if entry["trace_type"] == "optimal":
        if content == "NONE":
            return True
        for line in content.strip().split("\n"):
            if line.strip() and line.count("|") != 2:
                return False
        return True
    elif entry["trace_type"] == "cot":
        return "[REASONING]" in content and "[ANSWER]" in content
    return False


def normalize_schema(entries):
    """
    Ensure every entry in the list has exactly the same set of top-level
    keys, regardless of which source it came from. HuggingFace's datasets
    library infers a fixed schema from a JSONL file's rows and raises a
    CastError the moment a later row has a different key set than the
    rows the schema was inferred from -- e.g. Wikipedia-derived entries
    carry a "score" field the original 20K/15K entries don't have, and
    the 20K/15K entries carry a "phase" field Wikipedia entries don't have.
    Missing keys are filled with None rather than the entry being dropped.
    """
    all_keys = set()
    for e in entries:
        all_keys.update(e.keys())

    normalized = []
    for e in entries:
        new_e = {k: e.get(k, None) for k in all_keys}
        normalized.append(new_e)
    return normalized


# ── Loading ──────────────────────────────────────────────────

def load_wikipedia_scored():
    """Load and deduplicate all scored Wikipedia chunk files.

    Chunks were scored across multiple runs/processes, so the same
    sentence can appear more than once; the highest-scored copy wins.
    """
    seen = {}
    for path in WIKI_CHUNKS:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                sentence = entry["messages"][1]["content"].strip()
                score = entry.get("judgement", {}).get("score", -1)
                existing = seen.get(sentence, {}).get("judgement", {}).get("score", -1)
                if sentence not in seen or score > existing:
                    seen[sentence] = entry
    return list(seen.values())


def load_coref_resolutions():
    """
    Returns dict: original_sentence -> ("resolved", resolved_entry)
                                      | ("unresolved", None)
                                      | ("error", None)
    """
    resolutions = {}
    if not os.path.exists(COREF_FILE):
        print(f"  NOTE: {COREF_FILE} not found — coreference resolution has "
              f"not been run yet. All ambiguous-pronoun sentences will be "
              f"excluded from validation until it is.")
        return resolutions

    with open(COREF_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            status = row["status"]
            entry = row["entry"]
            if status == "resolved":
                original = entry["original_sentence"]
                resolutions[original] = ("resolved", entry)
            else:
                original = entry["messages"][1]["content"].strip()
                resolutions[original] = (status, None)
    return resolutions


# ── Main ─────────────────────────────────────────────────────

def main():
    start = time.time()

    if not os.path.exists(PHASE1_FILE):
        raise FileNotFoundError(
            f"{PHASE1_FILE} not found. Transfer phase1_training_data.jsonl "
            f"to the HTWK server before running this script."
        )

    print("Loading base training set (20K + noisy 15K, already slug format)...")
    with open(PHASE1_FILE, encoding="utf-8") as f:
        base_entries = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(base_entries)} traces loaded")

    print("\nLoading scored Wikipedia data...")
    wiki_entries = load_wikipedia_scored()
    print(f"  {len(wiki_entries)} unique scored sentences")

    print("\nLoading coreference resolution results...")
    coref_map = load_coref_resolutions()
    print(f"  {len(coref_map)} sentences have a coref decision on file")

    wiki_train_raw = [e for e in wiki_entries
                       if 0 <= e.get("judgement", {}).get("score", -1) < VALIDATION_SCORE_THRESHOLD]
    wiki_val_raw = [e for e in wiki_entries
                     if e.get("judgement", {}).get("score", -1) >= VALIDATION_SCORE_THRESHOLD]

    print(f"\nWikipedia score < {VALIDATION_SCORE_THRESHOLD}: {len(wiki_train_raw)} -> training (coref not applied here)")
    print(f"Wikipedia score >= {VALIDATION_SCORE_THRESHOLD}: {len(wiki_val_raw)} -> validation candidates")

    kept_unflagged = kept_resolved = kept_self_contained = dropped_ambiguous = dropped_error = 0
    final_val_raw = []

    for e in wiki_val_raw:
        sentence = e["messages"][1]["content"].strip()
        decision = coref_map.get(sentence)

        if decision is None:
            final_val_raw.append(e)
            kept_unflagged += 1
            continue

        status, resolved_entry = decision
        if status == "resolved":
            final_val_raw.append(resolved_entry)
            kept_resolved += 1
        elif status == "unresolved":
            # Not every "unresolved" case is truly ambiguous. A demonstrative
            # pronoun directly modifying a noun ("यह शहर", "वह मन्दिर") is a
            # self-contained exact span -- it doesn't need a real-world name
            # to be a valid, checkable triplet, so we keep it as-is.
            if entry_pronoun_status(e) == "self_contained":
                final_val_raw.append(e)
                kept_self_contained += 1
            else:
                dropped_ambiguous += 1
        else:
            dropped_error += 1

    print(f"\nCoreference resolution applied to validation set:")
    print(f"  kept, no pronoun issue:        {kept_unflagged}")
    print(f"  kept, pronoun resolved:        {kept_resolved}")
    print(f"  kept, self-contained (यह+noun): {kept_self_contained}")
    print(f"  dropped, genuinely ambiguous:  {dropped_ambiguous}")
    print(f"  dropped, API error:            {dropped_error}")
    print(f"  final validation candidates:   {len(final_val_raw)}")

    print("\nConverting Wikipedia training entries to slug format...")
    wiki_train_traces = convert_entries(wiki_train_raw, "wikipedia_lt9")
    print(f"  {len(wiki_train_traces)} traces")

    print("\nConverting final validation entries to slug format...")
    wiki_val_traces = convert_entries(final_val_raw, "wikipedia_ge9")
    print(f"  {len(wiki_val_traces)} traces")

    train_traces = base_entries + wiki_train_traces

    print("\nNormalizing schema so all entries share the same field set...")
    train_traces = normalize_schema(train_traces)
    wiki_val_traces = normalize_schema(wiki_val_traces)

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for e in train_traces:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for e in wiki_val_traces:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    elapsed = time.time() - start

    none_object_count = sum(
        1
        for e in wiki_train_traces + wiki_val_traces
        if e["trace_type"] == "optimal"
        for line in e["messages"][2]["content"].split("\n")
        if line.strip().endswith("| NONE")
    )

    invalid_train = sum(1 for e in train_traces if not verify_slug(e))
    invalid_val = sum(1 for e in wiki_val_traces if not verify_slug(e))

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"  Training set:   {len(train_traces)} traces -> {TRAIN_OUTPUT}")
    print(f"    - base (20K+15K):  {len(base_entries)}")
    print(f"    - wikipedia <9:    {len(wiki_train_traces)}")
    print(f"  Validation set: {len(wiki_val_traces)} traces -> {VAL_OUTPUT}")
    print(f"  Triplet lines normalized to NONE object: {none_object_count}")
    print(f"  Time: {elapsed:.1f}s")
    print()
    print(f"  Slug format check — training:   {invalid_train} invalid / {len(train_traces)}")
    print(f"  Slug format check — validation: {invalid_val} invalid / {len(wiki_val_traces)}")
    if invalid_train or invalid_val:
        print("  WARNING: invalid entries found above — inspect before training.")


if __name__ == "__main__":
    main()
