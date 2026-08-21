"""
fix_indie_train_subset.py — Extracts the correct 50-sample Train eval
subset (matching every other system's evaluation) from IndIE's raw
output, which was run on the full 39,621-sentence training file.
Wikipedia and BenchIE sections are already correct and copied through
unchanged.

Run:
    python3 fix_indie_train_subset.py
"""
import json
import random

TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
INDIE_RESULTS_FILE = "/home/nsingh/indie_baseline_results.json"
OUTPUT_FILE = "/home/nsingh/indie_baseline_results_fixed.json"
N_TRAIN = 50
SEED = 42


def main():
    all_train_examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            all_train_examples.append(sentence)

    random.Random(SEED).shuffle(all_train_examples)
    correct_50_sentences = set(all_train_examples[:N_TRAIN])
    print(f"Correct Train eval subset: {len(correct_50_sentences)} sentences")

    with open(INDIE_RESULTS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"IndIE 'train' results currently has: {len(data['train'])} entries")

    matched = [r for r in data["train"] if r["sentence"] in correct_50_sentences]
    print(f"Matched against correct 50-sample subset: {len(matched)} entries")

    if len(matched) != N_TRAIN:
        print(f"WARNING: expected {N_TRAIN}, got {len(matched)} -- investigate before trusting this output")

    data["train"] = matched

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nFixed file saved to: {OUTPUT_FILE}")
    print("Wikipedia and BenchIE sections copied through unchanged (they were already correct).")


if __name__ == "__main__":
    main()
