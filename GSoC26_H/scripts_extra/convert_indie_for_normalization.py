"""
convert_indie_for_normalization.py — Converts IndIE's raw triples-list
output into the same {"predicted": "subject | relation | object\n...",
"reference": "..."} text format that normalize_full_scale.py expects,
so IndIE's output can be normalized using the exact same, already-
verified pipeline as every other system this week.

Gold ("reference") is matched the same way as score_indie_baseline.py:
Wikipedia/BenchIE from eval_full_scale_results.json, Train from the
training file directly.

Note: IndIE has no property-vs-relational distinction -- every triple
it extracts is written as a plain "head | rel | tail" line, which
normalize_full_scale.py will correctly treat as relational (only the
literal string "property" is excluded).

Run:
    python3 convert_indie_for_normalization.py
"""
import json

INDIE_RESULTS_FILE = "/home/nsingh/indie_baseline_results_fixed.json"
EXTRACTION_RESULTS_FILE = "/home/nsingh/eval_full_scale_results.json"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
OUTPUT_FILE = "/home/nsingh/indie_converted_for_normalization_fixed.json"


def parse_triples(text):
    triples = []
    if not text or text.strip() == "NONE":
        return triples
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line == "NONE":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(tuple(parts))
    return triples


def triples_to_text(triples):
    if not triples:
        return "NONE"
    return "\n".join(f"{h} | {r} | {t}" for h, r, t in triples)


def main():
    with open(INDIE_RESULTS_FILE, encoding="utf-8") as f:
        indie_data = json.load(f)

    with open(EXTRACTION_RESULTS_FILE, encoding="utf-8") as f:
        extraction_data = json.load(f)
    gold_by_sentence = {"wikipedia": {}, "benchie": {}}
    for source in ("wikipedia", "benchie"):
        for r in extraction_data.get(source, []):
            gold_by_sentence[source][r["sentence"].strip()] = r["reference"]

    train_gold_by_sentence = {}
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            gold_text = entry["messages"][2]["content"]
            if "[ANSWER]" in gold_text:
                gold_text = gold_text.split("[ANSWER]")[-1]
            train_gold_by_sentence[sentence] = gold_text.strip()

    output = {"wikipedia": [], "train": [], "benchie": []}
    unmatched_total = 0

    for source in ("wikipedia", "train", "benchie"):
        for r in indie_data.get(source, []):
            sentence = r["sentence"].strip()
            predicted_triples = [tuple(t) for t in r.get("triples", [])]
            predicted_text = triples_to_text(predicted_triples)

            if source == "train":
                reference = train_gold_by_sentence.get(sentence)
            else:
                reference = gold_by_sentence[source].get(sentence)

            if reference is None:
                unmatched_total += 1
                continue

            output[source].append({
                "sentence": sentence,
                "reference": reference,
                "predicted": predicted_text,
            })

    for source in ("wikipedia", "train", "benchie"):
        print(f"{source}: {len(output[source])} converted")
    print(f"Unmatched (no gold found): {unmatched_total}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
