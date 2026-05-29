"""
gemma_zero_shot.py
------------------
Zero-shot and iterative-prompt Gemma-3 baseline for Hindi triple extraction.

Two prompt strategies:
  1. SIMULTANEOUS  — ask for subject, predicate, object all at once
  2. ITERATIVE     — MILIE-inspired: generate predicate after subject+object
                     (this is the prompt design used in fine-tuning)

Pre-application results (5 sentences):
  Simultaneous zero-shot:
    subject=5/5, predicate=0/5, object=5/5, full_triple=0/5
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from pydantic import BaseModel, ValidationError


# ─── Output Schema (Pydantic) ─────────────────────────────────────────────────

class HindiTriple(BaseModel):
    """Validated output schema for one extracted triple."""
    subject:   str
    predicate: str   # Should be a dbo: property or surface form
    object:    str   # named 'object_' to avoid Python keyword clash
    confidence: Optional[float] = None

    class Config:
        extra = "allow"


class ExtractionOutput(BaseModel):
    """Full output for one sentence."""
    sentence: str
    triples: List[HindiTriple] = []
    raw_output: str = ""
    parse_error: Optional[str] = None


# ─── Prompt Templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Hindi information extraction system specializing in DBpedia knowledge graph construction.
Your task is to extract relational triples from Hindi sentences.
Always output valid JSON. Never mix Hindi and English in predicates.
Predicates must be DBpedia ontology properties (e.g., dbo:birthPlace, dbo:capital, dbo:builder)."""

# Strategy 1: Simultaneous extraction (baseline — known to fail at predicates)
SIMULTANEOUS_PROMPT = """Extract all relational triples from this Hindi sentence.

Sentence: {sentence}

Rules:
- subject: the main entity being described (in Hindi)
- predicate: a DBpedia ontology property (dbo:birthPlace, dbo:capital, dbo:builder, dbo:spouse, etc.)
- object: the entity related to the subject (in Hindi)
- Do NOT use English words as predicates
- Do NOT use copulas (है, था) as predicates

Output ONLY valid JSON (no markdown, no explanation):
{{"triples": [{{"subject": "...", "predicate": "dbo:...", "object": "..."}}]}}"""

# Strategy 2: Iterative slot extraction (MILIE-inspired — used for fine-tuning design)
ITERATIVE_PROMPT = """Extract a relational triple from this Hindi sentence step by step.

Sentence: {sentence}

Step 1 - Identify the main entity (subject) in the sentence:
Subject: {subject_hint}

Step 2 - Identify what entity is related to it (object):
Object: {object_hint}

Step 3 - Given subject="{subject}" and object="{object}", 
what DBpedia ontology property best describes their relationship?
Choose from: dbo:birthPlace, dbo:capital, dbo:builder, dbo:spouse, dbo:nationality, 
dbo:occupation, dbo:author, dbo:director, dbo:location, dbo:award, dbo:country,
dbo:president, dbo:primeMinister, dbo:birthDate, dbo:deathPlace

Output ONLY valid JSON:
{{"subject": "...", "predicate": "dbo:...", "object": "..."}}"""


# ─── Gemma Runner ─────────────────────────────────────────────────────────────

class GemmaZeroShotRunner:
    """
    Runs Gemma-3 in zero-shot mode for Hindi triple extraction.

    Usage:
        runner = GemmaZeroShotRunner(model_id="google/gemma-3-1b-it")
        runner.load()
        result = runner.extract("ताज महल का निर्माण शाहजहाँ ने किया था।")
        print(result)
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-1b-it",
        use_4bit: bool = True,       # QLoRA quantization for Colab T4
        max_new_tokens: int = 256,
        temperature: float = 0.1,    # Low temp for structured extraction
        device: str = "auto",
    ):
        self.model_id       = model_id
        self.use_4bit       = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.device         = device
        self._model         = None
        self._tokenizer     = None

    def load(self) -> None:
        """Load model and tokenizer. Call once before extract()."""
        import torch
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  BitsAndBytesConfig)

        print(f"Loading {self.model_id} ...")

        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        print("Model ready.")

    # ── Extraction Methods ────────────────────────────────────────────────────

    def extract(self, sentence: str, strategy: str = "simultaneous") -> ExtractionOutput:
        """
        Extract triples from a Hindi sentence.

        Args:
            sentence: Hindi sentence string
            strategy: "simultaneous" | "iterative"
                      Use "simultaneous" for the zero-shot baseline.
                      Use "iterative" to replicate fine-tuning prompt design.

        Returns:
            ExtractionOutput with triples or parse error info.
        """
        if strategy == "simultaneous":
            return self._extract_simultaneous(sentence)
        elif strategy == "iterative":
            return self._extract_iterative(sentence)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def extract_batch(self, sentences: List[str],
                      strategy: str = "simultaneous") -> List[ExtractionOutput]:
        """Extract from a list of sentences."""
        results = []
        for i, sent in enumerate(sentences):
            print(f"  [{i+1}/{len(sentences)}] Processing: {sent[:50]}...")
            result = self.extract(sent, strategy=strategy)
            results.append(result)
            time.sleep(0.1)   # Avoid memory spikes on Colab
        return results

    # ── Private Methods ───────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        """Run model inference on a formatted prompt string."""
        import torch
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only new tokens (not the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_chat_prompt(self, user_message: str) -> str:
        """Format as Gemma chat template."""
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]
        return self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def _extract_simultaneous(self, sentence: str) -> ExtractionOutput:
        """Standard zero-shot simultaneous extraction."""
        prompt = self._build_chat_prompt(
            SIMULTANEOUS_PROMPT.format(sentence=sentence)
        )
        raw = self._generate(prompt)
        return self._parse_output(sentence, raw)

    def _extract_iterative(self, sentence: str) -> ExtractionOutput:
        """
        MILIE-inspired two-stage iterative extraction:
          Stage 1: Get subject + object (easy slots — 100% zero-shot accuracy)
          Stage 2: Get predicate conditioned on subject+object context
        """
        # Stage 1: Extract subject and object first
        stage1_prompt = self._build_chat_prompt(
            f"""From this Hindi sentence, identify ONLY the subject and object (not the predicate).

Sentence: {sentence}

Output ONLY valid JSON (no explanation):
{{"subject": "...", "object": "..."}}"""
        )
        raw1 = self._generate(stage1_prompt)
        args = self._parse_json_safe(raw1)
        subject = args.get("subject", "")
        obj     = args.get("object", "")

        # Stage 2: Predict predicate given subject+object context
        stage2_prompt = self._build_chat_prompt(
            ITERATIVE_PROMPT.format(
                sentence=sentence,
                subject_hint=subject or "[identify from sentence]",
                object_hint=obj or "[identify from sentence]",
                subject=subject,
                object=obj,
            )
        )
        raw2 = self._generate(stage2_prompt)
        full_raw = f"STAGE1: {raw1}\nSTAGE2: {raw2}"
        return self._parse_output(sentence, raw2, raw_output=full_raw)

    def _parse_output(self, sentence: str, raw: str,
                      raw_output: str = "") -> ExtractionOutput:
        """Parse model output into validated ExtractionOutput."""
        data = self._parse_json_safe(raw)

        if not data:
            return ExtractionOutput(
                sentence=sentence,
                triples=[],
                raw_output=raw_output or raw,
                parse_error=f"Could not parse JSON from: {raw[:200]}"
            )

        triples = []
        # Handle both {"triples": [...]} and {"subject":..., "predicate":..., "object":...}
        if "triples" in data and isinstance(data["triples"], list):
            for t in data["triples"]:
                try:
                    triples.append(HindiTriple(**t))
                except ValidationError as e:
                    pass  # Skip malformed triples
        elif "subject" in data and "predicate" in data:
            obj_key = "object_" if "object_" in data else "object"
            try:
                triples.append(HindiTriple(
                    subject=data.get("subject", ""),
                    predicate=data.get("predicate", ""),
                    object=data.get(obj_key, data.get("object", "")),
                ))
            except ValidationError:
                pass

        return ExtractionOutput(
            sentence=sentence,
            triples=triples,
            raw_output=raw_output or raw,
            parse_error=None if triples else f"No valid triples parsed from: {raw[:200]}"
        )

    def _parse_json_safe(self, text: str) -> dict:
        """Extract JSON from model output, handling common formatting issues."""
        text = text.strip()

        # Remove markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}


# ─── Convenience Functions ────────────────────────────────────────────────────

def run_zero_shot_baseline(sentences: List[Dict], model_id: str = "google/gemma-3-1b-it",
                            use_4bit: bool = True) -> List[Dict]:
    """
    Run the zero-shot baseline on a list of annotated sentences.

    Args:
        sentences: List of dicts with keys: sentence_id, sentence, gold_subject,
                   gold_predicate, gold_object
        model_id:  HuggingFace model ID
        use_4bit:  Whether to use 4-bit quantization (True for Colab T4)

    Returns:
        List of result dicts ready for ErrorTaxonomy.
    """
    from src.evaluation.error_taxonomy import ExtractionResult

    runner = GemmaZeroShotRunner(model_id=model_id, use_4bit=use_4bit)
    runner.load()

    results = []
    for s in sentences:
        output = runner.extract(s["sentence"], strategy="simultaneous")
        triple = output.triples[0] if output.triples else None

        result = ExtractionResult(
            sentence_id=s["sentence_id"],
            sentence=s["sentence"],
            gold_subject=s["gold_subject"],
            gold_predicate=s["gold_predicate"],
            gold_object=s["gold_object"],
            pred_subject=triple.subject   if triple else "",
            pred_predicate=triple.predicate if triple else "",
            pred_object=triple.object     if triple else "",
            system="gemma3_zero_shot",
        )
        results.append(result)
        print(f"  [{s['sentence_id']}] pred=({result.pred_subject}, "
              f"{result.pred_predicate}, {result.pred_object})")

    return results
