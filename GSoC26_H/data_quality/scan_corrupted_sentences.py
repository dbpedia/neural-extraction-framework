"""
scan_corrupted_sentences.py — Scans exp1_val_wikipedia_ge9.jsonl for
entries where the sentence field is corrupted (contains English
coreference-reasoning text instead of a clean Hindi sentence) —
detected via character-script ratio, not just keyword matching.

Run:
    python3 scan_corrupted_sentences.py
"""

import json
import re

VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
OUTPUT_FILE = "/home/nsingh/corrupted_sentences.jsonl"

DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
LATIN_RANGE = re.compile(r'[a-zA-Z]')


def is_corrupted(text):
    devanagari_count = len(DEVANAGARI_RANGE.findall(text))
    latin_count = len(LATIN_RANGE.findall(text))
    total = devanagari_count + latin_count
    if total == 0:
        return False, 0.0
    latin_ratio = latin_count / total
    # A clean Hindi sentence should be overwhelmingly Devanagari.
    # Corrupted reasoning-trace text is overwhelmingly Latin/English.
    return latin_ratio > 0.5, latin_ratio


def main():
    print(f"Scanning {VAL_FILE}...")
    total = 0
    corrupted = []

    with open(VAL_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()

            flagged, latin_ratio = is_corrupted(sentence)
            if flagged:
                corrupted.append({
                    "sentence_preview": sentence[:150],
                    "latin_ratio": round(latin_ratio, 3),
                    "full_length": len(sentence),
                })

    print(f"\nTotal entries scanned: {total}")
    print(f"Corrupted entries found: {len(corrupted)} "
          f"({len(corrupted)/total*100:.2f}%)")

    if corrupted:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for c in corrupted:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nSaved details to: {OUTPUT_FILE}")
        print("\nFirst 5 examples:")
        for c in corrupted[:5]:
            print(f"  [latin_ratio={c['latin_ratio']}] {c['sentence_preview']}...")
    else:
        print("\nNo corrupted entries found — this was likely an isolated case.")


if __name__ == "__main__":
    main()
