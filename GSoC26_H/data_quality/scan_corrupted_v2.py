"""
scan_corrupted_v2.py — Precise detection of coreference-reasoning-trace
contamination in exp1_val_wikipedia_ge9.jsonl. Unlike the script-ratio
version, this checks for specific reasoning-trace phrases rather than
just "too much Latin text" — avoiding false positives on legitimate
Hindi sentences containing English proper nouns.

Run:
    python3 scan_corrupted_v2.py
"""

import json

VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
OUTPUT_FILE = "/home/nsingh/corrupted_sentences_v2.jsonl"

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
    print(f"Scanning {VAL_FILE}...")
    total = 0
    seen_sentences = {}
    corrupted = []

    with open(VAL_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            total += 1
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()

            if is_corrupted(sentence):
                corrupted.append({
                    "line_number": i,
                    "sentence_preview": sentence[:200],
                    "full_length": len(sentence),
                })

            seen_sentences[sentence] = seen_sentences.get(sentence, 0) + 1

    duplicated_corrupted = [
        s for s in seen_sentences
        if seen_sentences[s] > 1 and is_corrupted(s)
    ]

    print(f"\nTotal entries scanned: {total}")
    print(f"Corrupted entries found: {len(corrupted)} "
          f"({len(corrupted)/total*100:.2f}%)")
    print(f"Unique corrupted sentences: {len(set(c['sentence_preview'] for c in corrupted))}")
    print(f"Corrupted sentences that are also duplicated: {len(duplicated_corrupted)}")

    if corrupted:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for c in corrupted:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"\nSaved to: {OUTPUT_FILE}")
        print("\nAll examples:")
        for c in corrupted:
            print(f"  [line {c['line_number']}] {c['sentence_preview']}...")
    else:
        print("\nNo corrupted entries found.")


if __name__ == "__main__":
    main()
