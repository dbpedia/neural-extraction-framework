"""
evaluate.py — Standalone evaluation for a fine-tuned checkpoint.

Two evaluation modes, selectable independently:

  validation   Generates predictions on the held-out Wikipedia validation
               set (exp1_val_wikipedia_ge9.jsonl) and reports:
                 - pass@1            exact string match against the
                                      reference slug answer
                 - valid_format_rate fraction of outputs that are
                                      well-formed slug triples

  benchie      Generates predictions on the BenchIE ground-truth
               benchmark and reports exact-match precision / recall / F1
               against the gold triples (see README — Methodology
               references, KaLLM's "Exact evaluation" tier). Partial-
               match scoring is intentionally not implemented here —
               this is the strictest, least ambiguous tier to start with.

Usage:
    python3 evaluate.py --checkpoint /path/to/checkpoint/final --mode validation
    python3 evaluate.py --checkpoint /path/to/checkpoint/final --mode benchie
    python3 evaluate.py --checkpoint /path/to/checkpoint/final --mode both

Generation and format-checking logic is imported directly from train.py
and prepare_data.py so standalone evaluation and the in-training
evaluation callback share exactly one definition of "correct."
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from train import extract_final_answer, is_valid_slug_format, get_prompt_and_reference
from prepare_data import OPTIMAL_INSTRUCTION


def load_model(checkpoint_path, base_model_name):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_slug_triples(text):
    text = extract_final_answer(text).strip()
    if text == "NONE" or not text:
        return []
    triples = []
    for line in text.split("\n"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append(tuple(parts))
    return triples


def evaluate_validation_set(model, tokenizer, val_file, n_samples, max_new_tokens):
    dataset = load_dataset("json", data_files=val_file, split="train")
    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    correct = 0
    valid_format = 0
    total = len(dataset)

    for i, example in enumerate(dataset):
        prompt, reference = get_prompt_and_reference(example, tokenizer)
        generated = generate(model, tokenizer, prompt, max_new_tokens)
        predicted = extract_final_answer(generated)

        if is_valid_slug_format(generated):
            valid_format += 1
        if predicted.strip() == reference.strip():
            correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{total}] running pass@1={correct/(i+1):.3f}", flush=True)

    return {
        "n_examples": total,
        "pass_at_1": correct / total if total else 0.0,
        "valid_format_rate": valid_format / total if total else 0.0,
    }


def evaluate_benchie(model, tokenizer, benchie_file, max_new_tokens):
    with open(benchie_file, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    gold_by_sentence = {}
    for row in rows:
        sentence = row["sentence"]
        gold = (row["subject"].strip(), row["relation"].strip(), row["object"].strip())
        gold_by_sentence.setdefault(sentence, []).append(gold)

    total_gold = sum(len(v) for v in gold_by_sentence.values())
    total_predicted = 0
    total_correct = 0

    for i, (sentence, gold_triples) in enumerate(gold_by_sentence.items()):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{OPTIMAL_INSTRUCTION}\n\n{sentence}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        generated = generate(model, tokenizer, prompt, max_new_tokens)
        predicted_set = set(parse_slug_triples(generated))
        total_predicted += len(predicted_set)

        for gold in gold_triples:
            if gold in predicted_set:
                total_correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(gold_by_sentence)} sentences]", flush=True)

    precision = total_correct / total_predicted if total_predicted else 0.0
    recall = total_correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n_sentences": len(gold_by_sentence),
        "n_gold_triples": total_gold,
        "n_predicted_triples": total_predicted,
        "n_correct": total_correct,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved LoRA adapter directory")
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--mode", choices=["validation", "benchie", "both"], default="validation")
    parser.add_argument("--val-file", default=os.path.expanduser("~/exp1_val_wikipedia_ge9.jsonl"))
    parser.add_argument("--benchie-file", default=os.path.expanduser("~/ground_truth_198_triples.jsonl"))
    parser.add_argument("--n-samples", type=int, default=None, help="Subsample the validation set for a quick check")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, args.base_model)

    if args.mode in ("validation", "both"):
        if not os.path.exists(args.val_file):
            print(f"Validation file not found: {args.val_file} — skipping")
        else:
            print(f"\nEvaluating on validation set: {args.val_file}")
            r = evaluate_validation_set(model, tokenizer, args.val_file, args.n_samples, args.max_new_tokens)
            print(f"\nValidation results:")
            print(f"  n_examples:         {r['n_examples']}")
            print(f"  pass@1:             {r['pass_at_1']:.3f}")
            print(f"  valid_format_rate:  {r['valid_format_rate']:.3f}")

    if args.mode in ("benchie", "both"):
        if not os.path.exists(args.benchie_file):
            print(f"\nBenchIE ground truth file not found: {args.benchie_file} — skipping")
        else:
            print(f"\nEvaluating on BenchIE ground truth: {args.benchie_file}")
            r = evaluate_benchie(model, tokenizer, args.benchie_file, args.max_new_tokens)
            print(f"\nBenchIE results (exact match):")
            print(f"  n_sentences:         {r['n_sentences']}")
            print(f"  n_gold_triples:      {r['n_gold_triples']}")
            print(f"  n_predicted_triples: {r['n_predicted_triples']}")
            print(f"  n_correct:           {r['n_correct']}")
            print(f"  precision:           {r['precision']:.3f}")
            print(f"  recall:              {r['recall']:.3f}")
            print(f"  f1:                  {r['f1']:.3f}")


if __name__ == "__main__":
    main()
