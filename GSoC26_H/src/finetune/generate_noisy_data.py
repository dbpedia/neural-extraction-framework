"""
generate_noisy_data.py
-----------------------
Phase 2: Noisy synthetic dataset generation for staged fine-tuning.

Staged training rationale (Debarghya, Aditya — Week 2 sync):
  Stage 1 (noisy):  score < 8 examples from the GSoC25_H 20K synthetic set
                     + freshly generated noisy examples (this script)
  Stage 2 (clean):  score >= 9 examples from the same 20K set (8,633 rows)
  The model first learns the *shape* of the task on noisy data, then
  refines on verified-clean data — rather than overfitting immediately
  to a small clean set.

Why the noise comes from few-shot seeds, not a weak generator model:
  Two ways exist to make synthetic data noisy:
    (a) use a small/weak LLM as the generator, or
    (b) use the SAME generator the original 20K used, but seed its
        few-shot examples with already-flawed (score < 8) rows.
  This script uses (b). Aditya's original 20K set was generated with a
  full-size model (openai/gpt-oss-120b); per his guidance, the
  noisy-generation step stays at that same model tier rather than
  dropping to a 3-4B model, so the *only* variable that changes is the
  few-shot seed quality. This isolates "what does a strong model do
  when shown flawed examples to imitate" — a more realistic proxy for
  the kind of errors a HITL reviewer needs to learn to catch, since the
  errors come from genuine semantic mistakes (span boundaries, argument
  reversal, missing negation) rather than a weak model's inability to
  follow instructions at all.

Reuses GSoC25_H's synthetic_data_gen_2.py prompt infrastructure
(SEMANTIC_CONCEPTS, STRUCTURE_TEMPLATES, prompt builders) unchanged —
only the generator's few-shot history and output model are swapped.

Verified results (Week 2-3, NVIDIA-hosted openai/gpt-oss-120b):
  ~86% JSON-valid success rate per batch (vs ~55% with a 3B model)
  Zero duplicate sentences, zero empty-triplet rows in generated output
  ~15,000+ examples generated toward the Phase 1 (noisy) staged-training pool
"""

import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from openai import OpenAI


# ─── Config ────────────────────────────────────────────────────────────────

GENERATION_MODEL = "openai/gpt-oss-120b"          # same tier as the original 20K set
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOW_SCORE_THRESHOLD = 8                            # seed examples: score in [0, 8)
NOISY_SEED_COUNT = 6                                # few-shot examples per generation call
DEFAULT_WORKERS = 10
DEFAULT_BATCH_SIZE = 2000

SYSTEM_PROMPT_FOR_FINETUNE = (
    "Extract all subject-relation-object triplets from the given Hindi "
    "sentence. Output a JSON object with 'thought_process' and "
    "'extracted_triplets'."
)

GENERATOR_SYSTEM_INSTRUCTION = (
    "You are an expert in Hindi linguistics and information extraction. "
    "Your task is to generate complex Hindi sentences and then "
    "meticulously extract all subject-relation-object (SRO) triplets "
    "from them, following a strict set of annotation guidelines. Your "
    "output must always be a single, valid JSON object."
)


# ─── Seed selection ──────────────────────────────────────────────────────────

def build_noisy_few_shot_seeds(
    scored_examples: List[dict],
    seed_count: int = NOISY_SEED_COUNT,
    rng_seed: int = 7,
) -> List[Dict]:
    """
    Sample few-shot examples from the LOW-score pool (score < 8) of the
    original 20K dataset. These contain genuine extraction mistakes —
    wrong span boundaries, subject/object reversal, missing negation —
    which a strong generator model will tend to imitate when shown as
    in-context examples.

    Returns a list of dicts: {hindi_sentence, thought_process, extracted_triplets}
    """
    low_score_pool = [
        ex for ex in scored_examples
        if 0 <= ex["judgement"]["score"] < LOW_SCORE_THRESHOLD
    ]
    if len(low_score_pool) < seed_count:
        raise ValueError(
            f"Only {len(low_score_pool)} low-score examples available, "
            f"need at least {seed_count}."
        )

    random.seed(rng_seed)
    sampled = random.sample(low_score_pool, seed_count)

    seeds = []
    for ex in sampled:
        sentence = ex["messages"][1]["content"]
        content = json.loads(ex["messages"][2]["content"])
        seeds.append({
            "hindi_sentence": sentence,
            "thought_process": content.get("thought_process", ""),
            "extracted_triplets": content.get("extracted_triplets", []),
        })
    return seeds


def seeds_to_chat_history(seeds: List[Dict]) -> List[Dict]:
    """Convert noisy seed examples into OpenAI-style chat history for few-shot priming."""
    history = []
    for ex in seeds:
        history.append({"role": "user", "content": ex["hindi_sentence"]})
        history.append({
            "role": "assistant",
            "content": json.dumps({
                "hindi_sentence": ex["hindi_sentence"],
                "thought_process": ex["thought_process"],
                "extracted_triplets": ex["extracted_triplets"],
            }, ensure_ascii=False),
        })
    return history


# ─── Generation ──────────────────────────────────────────────────────────────

def _extract_text(message) -> Optional[str]:
    """
    Defensive text extraction. Some reasoning-style model responses put
    the final answer in a non-standard field rather than `.content`.
    """
    if getattr(message, "content", None):
        return message.content.strip()
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val:
            return val.strip()
    return None


def _parse_generated_json(raw: str) -> Optional[dict]:
    """Strip optional markdown code fences and parse the model's JSON output."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].replace("json", "", 1)
        parsed = json.loads(cleaned)
        if not (
            parsed.get("hindi_sentence")
            and parsed.get("thought_process")
            and isinstance(parsed.get("extracted_triplets"), list)
        ):
            return None
        return parsed
    except (json.JSONDecodeError, IndexError):
        return None


class NoisyDataGenerator:
    """
    Generates noisy Hindi SRO-extraction examples by reusing GSoC25_H's
    prompt-construction functions, but priming the generator with
    flawed (score < 8) few-shot examples instead of clean ones.

    Usage:
        gen = NoisyDataGenerator(
            api_key=NVIDIA_API_KEY,
            seeds=build_noisy_few_shot_seeds(scored_examples),
            semantic_concepts=SEMANTIC_CONCEPTS,        # from synthetic_data_gen_2
            structure_templates=STRUCTURE_TEMPLATES,    # from synthetic_data_gen_2
            prompt_builders=(create_structure_first_prompt,
                              create_multi_relation_prompt,
                              create_targeted_relation_prompt,
                              select_relations_from_different_concepts),
        )
        gen.run_batch(output_path="noisy_synthetic_data.jsonl",
                      n_attempts=2000, n_workers=10)
    """

    def __init__(
        self,
        api_key: str,
        seeds: List[Dict],
        semantic_concepts: Dict[str, List[str]],
        structure_templates: List[str],
        prompt_builders: tuple,
        model: str = GENERATION_MODEL,
    ):
        self.client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
        self.model = model
        self.semantic_concepts = semantic_concepts
        self.structure_templates = structure_templates
        (
            self.create_structure_first_prompt,
            self.create_multi_relation_prompt,
            self.create_targeted_relation_prompt,
            self.select_relations_from_different_concepts,
        ) = prompt_builders
        self.noisy_history = seeds_to_chat_history(seeds)
        self._write_lock = threading.Lock()
        self._all_relations = [
            r for rs in semantic_concepts.values() for r in rs
        ]

    def _build_prompt(self) -> str:
        """Pick a generation strategy, mirroring synthetic_data_gen_2.py's weighting."""
        strategy = random.choices(
            ["structure_first", "multi_relation", "targeted_relation"],
            weights=[0.5, 0.3, 0.2],
        )[0]

        if strategy == "structure_first":
            template = random.choice(self.structure_templates)
            relation = random.choice(self._all_relations)
            return self.create_structure_first_prompt(template, relation)
        elif strategy == "multi_relation":
            rel1, rel2, _, _ = self.select_relations_from_different_concepts()
            return self.create_multi_relation_prompt(rel1, rel2)
        else:
            relation = random.choice(self._all_relations)
            return self.create_targeted_relation_prompt(relation)

    def _generate_one(self, output_path: str) -> bool:
        """Single generation attempt. Returns True if a valid row was written."""
        prompt = self._build_prompt()
        messages = (
            [{"role": "system", "content": GENERATOR_SYSTEM_INSTRUCTION}]
            + self.noisy_history
            + [{"role": "user", "content": prompt}]
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.9,
                top_p=0.9,
                max_tokens=4000,
            )
            raw = _extract_text(response.choices[0].message)
            if raw is None:
                return False
        except Exception:
            return False

        parsed = _parse_generated_json(raw)
        if parsed is None:
            return False

        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_FINETUNE},
                {"role": "user", "content": parsed["hindi_sentence"]},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought_process": parsed["thought_process"],
                        "extracted_triplets": parsed["extracted_triplets"],
                    }, ensure_ascii=False),
                },
            ],
            "generator_model": self.model,
            "is_noisy": True,
        }

        with self._write_lock:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
        return True

    def run_batch(
        self,
        output_path: str,
        n_attempts: int = DEFAULT_BATCH_SIZE,
        n_workers: int = DEFAULT_WORKERS,
        progress_every: int = 250,
    ) -> int:
        """
        Run `n_attempts` generation attempts in parallel, appending
        successful rows to `output_path`. Returns the number generated.

        Note on throughput: NVIDIA's free tier appears to rate-limit at
        the account level rather than per-connection — increasing
        n_workers beyond ~10 has shown diminishing returns in practice.
        """
        generated = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self._generate_one, output_path)
                for _ in range(n_attempts)
            ]
            for i, future in enumerate(as_completed(futures)):
                if future.result():
                    generated += 1
                if (i + 1) % progress_every == 0:
                    print(f"  [{i + 1}/{n_attempts}] generated so far: {generated}")

        print(f"Batch complete. Generated: {generated} / {n_attempts} attempts "
              f"({generated / n_attempts * 100:.1f}% success rate)")
        return generated


# ─── Quality checks ──────────────────────────────────────────────────────────

def validate_output_file(path: str) -> Dict[str, int]:
    """
    Run the same structural checks used during Week 2-3 generation:
    empty-triplet rows and duplicate sentences. Returns a summary dict.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    empty_triplets = 0
    for row in rows:
        content = json.loads(row["messages"][2]["content"])
        if len(content.get("extracted_triplets", [])) == 0:
            empty_triplets += 1

    sentences = [row["messages"][1]["content"] for row in rows]
    unique_sentences = len(set(sentences))

    summary = {
        "total_rows": len(rows),
        "empty_triplet_rows": empty_triplets,
        "unique_sentences": unique_sentences,
        "duplicate_sentences": len(rows) - unique_sentences,
    }
    return summary


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-data", required=True,
                         help="Path to the GSoC25_H scored 20K JSONL file "
                              "(used to source noisy few-shot seeds)")
    parser.add_argument("--gen-script", required=True,
                         help="Path to GSoC25_H's synthetic_data_gen_2.py "
                              "(provides SEMANTIC_CONCEPTS, STRUCTURE_TEMPLATES, "
                              "and the prompt-builder functions)")
    parser.add_argument("--output", default="noisy_synthetic_data.jsonl")
    parser.add_argument("--n-attempts", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--api-key", default=os.environ.get("NVIDIA_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Set --api-key or the NVIDIA_API_KEY environment variable.")

    # Load GSoC25_H's prompt infrastructure dynamically — avoids duplicating
    # SEMANTIC_CONCEPTS / STRUCTURE_TEMPLATES / prompt builders in this repo.
    import types
    with open(args.gen_script, "r", encoding="utf-8") as f:
        source = f.read().split('if __name__ == "__main__":')[0]
    gen_module = types.ModuleType("synthetic_data_gen_2")
    exec(compile(source, args.gen_script, "exec"), gen_module.__dict__)

    with open(args.scored_data, "r", encoding="utf-8") as f:
        scored_examples = [json.loads(line) for line in f if line.strip()]

    seeds = build_noisy_few_shot_seeds(scored_examples)
    print(f"Built {len(seeds)} noisy few-shot seeds "
          f"(scores: {[s.get('score') for s in seeds]})")

    generator = NoisyDataGenerator(
        api_key=args.api_key,
        seeds=seeds,
        semantic_concepts=gen_module.SEMANTIC_CONCEPTS,
        structure_templates=gen_module.STRUCTURE_TEMPLATES,
        prompt_builders=(
            gen_module.create_structure_first_prompt,
            gen_module.create_multi_relation_prompt,
            gen_module.create_targeted_relation_prompt,
            gen_module.select_relations_from_different_concepts,
        ),
    )

    generator.run_batch(
        output_path=args.output,
        n_attempts=args.n_attempts,
        n_workers=args.n_workers,
    )

    summary = validate_output_file(args.output)
    print(f"\nValidation summary: {summary}")
