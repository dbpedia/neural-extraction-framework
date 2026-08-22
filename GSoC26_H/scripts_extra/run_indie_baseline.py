"""
run_indie_baseline.py — Runs the IndIE baseline (external, rule-based +
CRF chunker model, github.com/ritwikmishra/IndIE) across the exact same
evaluation set used for every other comparison this project (1,817
Wikipedia + 50 Train + 112 BenchIE), per Debarghya's request to
establish an IndIE baseline alongside the fine-tuned Gemma model.

Confirmed interface (verified directly from IndIE's own source, not
assumed):
  - ChunckProcessor(language="hi") -- loads model once
  - processor.run(sentence) -- one sentence per call, returns
    (all_sents, exts, ctime, etime, ttaken)
  - exts[0] is a list of [head, rel, tail] triples for that sentence
    (confirmed via grep on utils.py: every .append([head, rel, tail])
    call uses this exact order)

Data loading logic matches evaluate_full_scale.py exactly, same file
paths, same sentence sets -- so IndIE's results are directly comparable
to every other baseline/variant tested this week.

Run (inside tmux -- IndIE's own smoke test took real time even for one
sentence with stanza on CPU, so 1,979 sentences will take a while):
    cd ~/IndIE && source indie_env/bin/activate && bash
    python3 ~/run_indie_baseline.py
"""

import json
import sys

sys.path.insert(0, "/home/nsingh/IndIE")
from chunck import ChunckProcessor

WIKI_VAL_FILE = "/home/nsingh/exp1_val_wikipedia_ge9.jsonl"
TRAIN_FILE = "/home/nsingh/exp1_train_optimal_only.jsonl"
BENCHIE_FILE = "/home/nsingh/benchie_converted.jsonl"
OUTPUT_FILE = "/home/nsingh/indie_baseline_results.json"
SAVE_EVERY = 25


def load_all_wikipedia():
    seen_sentences = set()
    examples = []
    with open(WIKI_VAL_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            examples.append({"sentence": sentence})
    return examples


def load_train_sentences():
    examples = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sentence = entry["messages"][1]["content"].strip()
            examples.append({"sentence": sentence})
    return examples


def load_all_benchie():
    examples = []
    with open(BENCHIE_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            examples.append({"sentence": entry["sentence"]})
    return examples


def run_source(source_name, samples, processor, all_results):
    print(f"\n{'='*60}\nRunning IndIE on: {source_name} ({len(samples)} samples)\n{'='*60}")

    for idx, ex in enumerate(samples):
        sentence = ex["sentence"]
        triples = []
        error = None
        try:
            all_sents, exts, ctime, etime, ttaken = processor.run(sentence)
            if exts and len(exts) > 0:
                for t in exts[0]:
                    if len(t) == 3:
                        head, rel, tail = t
                        triples.append([
                            head if isinstance(head, str) else str(head),
                            rel if isinstance(rel, str) else str(rel),
                            tail if isinstance(tail, str) else str(tail),
                        ])
        except Exception as e:
            error = str(e)

        all_results[source_name].append({
            "sentence": sentence,
            "triples": triples,
            "error": error,
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            n_done = idx + 1
            n_errors = sum(1 for r in all_results[source_name] if r["error"])
            print(f"  [{source_name}] {n_done}/{len(samples)} done, {n_errors} errors so far", flush=True)

        if (idx + 1) % SAVE_EVERY == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)


def main():
    print("Loading IndIE Hindi processor (loads models once)...")
    processor = ChunckProcessor(language="hi")

    print("\nLoading evaluation sentences (same files as evaluate_full_scale.py)...")
    wiki_samples = load_all_wikipedia()
    train_samples = load_train_sentences()
    benchie_samples = load_all_benchie()

    print(f"  Wikipedia: {len(wiki_samples)}")
    print(f"  Train:     {len(train_samples)}")
    print(f"  BenchIE:   {len(benchie_samples)}")

    all_results = {"wikipedia": [], "train": [], "benchie": []}

    run_source("wikipedia", wiki_samples, processor, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("train", train_samples, processor, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    run_source("benchie", benchie_samples, processor, all_results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
