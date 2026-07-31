"""
filter_hitl_corrupted.py — Removes rows from alignment_results_full_20k.jsonl
where the sentence field is corrupted (contains coreference-resolution
reasoning-trace text instead of a clean Hindi sentence). Does NOT touch
the official eval numbers (eval_full_scale_results.json,
normalization_results_full_scale.json) — those stay as-is, documented
as having ~4.3% contaminated Wikipedia data. This only cleans what
gets shown for human review.

Run:
    python3 filter_hitl_corrupted.py
"""

import json

INPUT_FILE = "/home/nsingh/alignment_results_full_20k.jsonl"
OUTPUT_FILE = "/home/nsingh/alignment_results_full_20k.jsonl"  # overwrite in place

TELLTALE_PHRASES = [
    "we need to see",
    "need to see if pronoun",
    "refers to an entity",
    "named elsewhere in",
    "the rule:",
    "the instruction:",
]


def is_corrupted(text):
    lower = text.lower()
    return any(phrase in lower for phrase in TELLTALE_PHRASES)


def main():
    print(f"Loading {INPUT_FILE}...")
    rows = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Total rows: {len(rows)}")

    clean_rows = [r for r in rows if not is_corrupted(r.get("sentence", ""))]
    removed = len(rows) - len(clean_rows)

    print(f"Removed (corrupted): {removed}")
    print(f"Remaining (clean): {len(clean_rows)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved cleaned file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
