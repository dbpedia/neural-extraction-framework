"""
test_cross_lingual_robustness.py — Robustness check: does the fine-tuned
Gemma 3 4B extraction model produce SOME triplet output (right or wrong)
on Indic languages it was never trained on? Per the task: correctness
is explicitly not the bar here, only that the pipeline doesn't crash
and does attempt extraction.

50 Gujarati + 50 Rajasthani test sentences, loaded from
cross_lingual_sentences.py (see that file's docstring for a note on
relative confidence between the two languages).

Reuses the exact generate()/build_prompt()/extract_final_answer() logic
already verified working in evaluate_full_scale.py.

Run:
    python3 test_cross_lingual_robustness.py
"""

import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, "/home/nsingh")
from cross_lingual_sentences import GUJARATI_SENTENCES, RAJASTHANI_SENTENCES

BASE_MODEL = "google/gemma-3-4b-it"
CHECKPOINT_PATH = "/home/nsingh/checkpoints/exp1_all_data_aug_lr0.0002/final"
MAX_NEW_TOKENS = 256
OUTPUT_FILE = "/home/nsingh/cross_lingual_robustness_results.json"

OPTIMAL_INSTRUCTION = """Extract all subject-relation-object triplets from this Hindi sentence.
Use the format: subject | relation | object
One triplet per line.
If no triplets exist, write: NONE"""

TEST_SENTENCES = (
    [{"language": "Gujarati", "sentence": s} for s in GUJARATI_SENTENCES]
    + [{"language": "Rajasthani", "sentence": s} for s in RAJASTHANI_SENTENCES]
)


def extract_final_answer(text):
    if "[ANSWER]" in text:
        text = text.split("[ANSWER]")[-1]
    return text.strip()


def build_prompt(tokenizer, sentence):
    content = f"{OPTIMAL_INSTRUCTION}\n\nSentence: {sentence}"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(model, tokenizer, prompt, max_new_tokens=256):
    """Manual greedy decode with KV-cache reuse. Stops on either the base
    eos_token_id OR Gemma's <end_of_turn> token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = input_ids.shape[1]
    generated = input_ids

    eos_id = tokenizer.eos_token_id
    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = {eos_id}
    if end_of_turn_id is not None and end_of_turn_id >= 0:
        stop_ids.add(end_of_turn_id)

    with torch.no_grad():
        past_key_values = None
        current_input = generated
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)], dim=-1
            )
            past_key_values = outputs.past_key_values
            current_input = next_token
            if next_token.item() in stop_ids:
                break

    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)


def main():
    print("Loading tokenizer and fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.eval()

    print(f"\n{'='*70}\nCROSS-LINGUAL ROBUSTNESS TEST\n{'='*70}")
    print(f"Total sentences: {len(TEST_SENTENCES)} "
          f"({len(GUJARATI_SENTENCES)} Gujarati + {len(RAJASTHANI_SENTENCES)} Rajasthani)")
    print("Goal: confirm the pipeline produces SOME output without crashing.")
    print("Correctness is explicitly NOT being evaluated here.\n")

    results = []
    for i, ex in enumerate(TEST_SENTENCES):
        try:
            prompt = build_prompt(tokenizer, ex["sentence"])
            raw_output = generate(model, tokenizer, prompt, MAX_NEW_TOKENS)
            predicted = extract_final_answer(raw_output)
            crashed = False
        except Exception as e:
            predicted = None
            crashed = True
            print(f"[{i+1}/{len(TEST_SENTENCES)}] {ex['language']} — CRASHED: {e}")

        produced_output = (not crashed) and bool(predicted) and predicted.strip() != ""

        results.append({
            "language": ex["language"],
            "sentence": ex["sentence"],
            "predicted": predicted,
            "crashed": crashed,
            "produced_output": produced_output,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(TEST_SENTENCES):
            print(f"  [{i+1}/{len(TEST_SENTENCES)}] processed", flush=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for lang in ("Gujarati", "Rajasthani"):
        lang_results = [r for r in results if r["language"] == lang]
        n = len(lang_results)
        crashed_count = sum(1 for r in lang_results if r["crashed"])
        produced_count = sum(1 for r in lang_results if r["produced_output"])
        empty_count = n - crashed_count - produced_count
        print(f"\n{lang} (n={n})")
        print(f"  Produced output: {produced_count}/{n}")
        print(f"  Empty output:    {empty_count}/{n}")
        print(f"  Crashed:         {crashed_count}/{n}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
