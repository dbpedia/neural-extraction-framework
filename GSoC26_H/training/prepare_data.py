"""
prepare_data.py — Build the final Experiment 1 training and validation sets.

Combines:
  - Aditya's 20K (all scores) + Noisy 15K — already in slug format
    (phase1_training_data.jsonl)
  - Wikipedia score < 9  -> added to training set (converted here)
  - Wikipedia score >= 9 -> validation set (converted here)

Fix applied during conversion:
  Empty/None triplet objects (common for intransitive Hindi verbs like
  "बहिष्कार किया", "हो गया") are normalized to the literal string "NONE"
  instead of being dropped or left blank, so every slug line stays
  well-formed (exactly two "|" separators).

Outputs:
  ~/exp1_train_combined.jsonl
  ~/exp1_val_wikipedia_ge9.jsonl
"""

import json
import os
import time

# ── Paths ────────────────────────────────────────────────────
PHASE1_FILE = os.path.expanduser("~/phase1_training_data.jsonl")
WIKI_CHUNK_FILES = [
    os.path.expanduser(f"~/wiki_chunk_{i}_scored.jsonl") for i in [0, 1, 2]
]

TRAIN_OUTPUT = os.path.expanduser("~/exp1_train_combined.jsonl")
VAL_OUTPUT   = os.path.expanduser("~/exp1_val_wikipedia_ge9.jsonl")

VALIDATION_SCORE_THRESHOLD = 9

# ── Slug format instructions (must match phase1_training_data.jsonl) ──
OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""

COT_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Think step by step, then provide the triplets.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""


def clean_span(value):
    """None or whitespace-only -> empty string."""
    if value is None:
        return ""
    return value.strip()


def triplets_to_slug(triplets):
    """
    Convert a list of triplet dicts to pipe-separated slug lines.

    - Subject and relation must be non-empty, or the triplet is skipped
      entirely (a genuine corruption, e.g. a failed upstream JSON parse).
    - An empty/None object is normalized to the literal string "NONE"
      rather than dropped, since many correct Hindi triplets use
      intransitive verbs with no grammatical object
      (e.g. "सदस्यों | बहिष्कार किया | NONE").
    """
    if not triplets:
        return "NONE"
    lines = []
    for t in triplets:
        s = clean_span(t.get("subject"))
        r = clean_span(t.get("relation"))
        o = clean_span(t.get("object"))
        if not s or not r:
            continue
        if not o:
            o = "NONE"
        lines.append(f"{s} | {r} | {o}")
    return "\n".join(lines) if lines else "NONE"


def convert_to_traces(sentence, triplets, thought, source, score=None):
    """Build the Optimal and CoT training examples for one sentence."""
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
    if score is not None:
        optimal["score"] = score

    cot_content = f"[REASONING]\n{thought}\n\n[ANSWER]\n{slug_ans}"
    cot = {
        "messages": [
            {"role": "system", "content": COT_INSTRUCTION},
            {"role": "user", "content": sentence},
            {"role": "assistant", "content": cot_content},
        ],
        "source": source,
        "trace_type": "cot",
    }
    if score is not None:
        cot["score"] = score

    return optimal, cot


def load_wikipedia_scored():
    """Load and deduplicate all three scored Wikipedia chunk files."""
    seen = {}
    for path in WIKI_CHUNK_FILES:
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
                existing_score = seen.get(sentence, {}).get("judgement", {}).get("score", -1)
                if sentence not in seen or score > existing_score:
                    seen[sentence] = entry
    return list(seen.values())


def convert_batch(entries, source_label):
    traces = []
    skipped = 0
    for e in entries:
        sentence = e["messages"][1]["content"].strip()
        try:
            content = json.loads(e["messages"][2]["content"])
        except (json.JSONDecodeError, KeyError, IndexError):
            skipped += 1
            continue
        thought = content.get("thought_process", "")
        triplets = content.get("extracted_triplets", [])
        score = e.get("judgement", {}).get("score")
        optimal, cot = convert_to_traces(sentence, triplets, thought, source_label, score)
        traces.append(optimal)
        traces.append(cot)
    if skipped:
        print(f"  ({skipped} entries skipped due to malformed content)")
    return traces


def main():
    start = time.time()

    if not os.path.exists(PHASE1_FILE):
        raise FileNotFoundError(
            f"{PHASE1_FILE} not found.\n"
            f"Copy phase1_training_data.jsonl from Google Drive to the "
            f"HTWK server (e.g. drag-and-drop via VS Code Explorer) before "
            f"running this script."
        )

    print("Loading base training set (20K + noisy 15K, already slug format)...")
    with open(PHASE1_FILE, encoding="utf-8") as f:
        base_entries = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(base_entries)} traces loaded")

    print("\nLoading scored Wikipedia data...")
    wiki_entries = load_wikipedia_scored()
    print(f"  {len(wiki_entries)} unique scored sentences")

    wiki_train = [e for e in wiki_entries
                  if 0 <= e.get("judgement", {}).get("score", -1) < VALIDATION_SCORE_THRESHOLD]
    wiki_val = [e for e in wiki_entries
                if e.get("judgement", {}).get("score", -1) >= VALIDATION_SCORE_THRESHOLD]

    print(f"  Wikipedia score < {VALIDATION_SCORE_THRESHOLD}: {len(wiki_train)} -> training")
    print(f"  Wikipedia score >= {VALIDATION_SCORE_THRESHOLD}: {len(wiki_val)} -> validation")

    print("\nConverting Wikipedia training entries to slug format...")
    wiki_train_traces = convert_batch(wiki_train, "wikipedia_lt9")
    print(f"  {len(wiki_train_traces)} traces")

    print("\nConverting Wikipedia validation entries to slug format...")
    wiki_val_traces = convert_batch(wiki_val, "wikipedia_ge9")
    print(f"  {len(wiki_val_traces)} traces")

    train_traces = base_entries + wiki_train_traces
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for e in train_traces:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for e in wiki_val_traces:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    elapsed = time.time() - start

    print(f"\n{'='*55}")
    print("Done.")
    print(f"  Training set:   {len(train_traces)} traces -> {TRAIN_OUTPUT}")
    print(f"    - base (20K+15K): {len(base_entries)}")
    print(f"    - wikipedia <9:   {len(wiki_train_traces)}")
    print(f"  Validation set: {len(wiki_val_traces)} traces -> {VAL_OUTPUT}")
    print(f"  Time: {elapsed:.1f}s")

    none_object_count = sum(
        1
        for e in wiki_train_traces + wiki_val_traces
        if e["trace_type"] == "optimal"
        for line in e["messages"][2]["content"].split("\n")
        if line.strip().endswith("| NONE")
    )
    print(f"  Triplet lines with normalized NONE object: {none_object_count}")


if __name__ == "__main__":
    main()
