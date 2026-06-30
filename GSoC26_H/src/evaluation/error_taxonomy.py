"""
error_taxonomy.py
-----------------
Error taxonomy for Hindi relational triple extraction failures.

Defines the 6 error types identified across Phase 1 baselines and
Phase 2 fine-tuning evaluation (GSoC 2026):

  1. CORRECT                   — full triple matches gold
  2. PREDICATE_NORMALIZATION   — predicate not alignable to any DBpedia property
  3. PREDICATE_PLACEHOLDER     — copula (है/था/थे) used as predicate; implicit relation
  4. LANGUAGE_MIXING           — English predicate in Hindi sentence
  5. ARGUMENT_SPAN             — predicate correct but subject/object boundaries wrong
  6. MISSING_TRIPLE            — no triple extracted at all

Note: PREDICATE_PLACEHOLDER was measured in phase1_ablation_table.csv but was
missing from the original enum — added in Week 3 to fix the inconsistency.
PREDICATE_PLACEHOLDER is a special case of PREDICATE_NORMALIZATION where the
surface form is a copula, distinguishing it from genuine normalization failures.

Used in: Phase 1 (baseline eval), Phase 2 (fine-tune eval), Phase 3 (HITL labelling).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import Counter
import json


# ─── Error Types ──────────────────────────────────────────────────────────────

class ErrorType(Enum):
    CORRECT                      = "correct"
    PREDICATE_NORMALIZATION      = "predicate_normalization_failure"
    PREDICATE_PLACEHOLDER        = "predicate_placeholder"        # copula as predicate
    LANGUAGE_MIXING              = "language_mixing"
    ARGUMENT_SPAN                = "argument_span_error"
    MISSING_TRIPLE               = "missing_triple"


# Hindi copula forms that signal Predicate Placeholder Error
HINDI_COPULAS = {
    "है", "हैं", "था", "थे", "थी", "थीं",
    "hai", "hain", "tha", "the", "thi", "thin",
    "ho", "hoga", "hogi", "raha", "rahi",
    "रहा", "रही", "रहे", "होना", "हो",
}

# Common English function words — signal Language Mixing
ENGLISH_FUNCTION_WORDS = {
    "is", "are", "was", "were", "has", "have", "had",
    "born", "in", "of", "from", "at", "to", "for",
    "located", "founded", "created", "built", "died"
}


# ─── Single Extraction Result ─────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Represents one extracted triple vs. gold triple, with error classification."""

    sentence_id:          str
    sentence:             str
    gold_subject:         str
    gold_predicate:       str        # e.g. "dbo:birthPlace"
    gold_object:          str

    pred_subject:         str = ""
    pred_predicate:       str = ""   # raw surface form from model
    pred_object:          str = ""

    aligned_predicate:    str = ""   # after ontology alignment layer
    alignment_confidence: float = 0.0

    error_type:           Optional[ErrorType] = None
    error_notes:          str = ""
    system:               str = ""   # "indIE" | "gemma3_zero_shot" | "gsoc25h" | "gemma3_finetuned"

    def classify_error(self) -> "ExtractionResult":
        """Auto-classify error type. Call after setting pred_* fields."""

        effective_pred = self.aligned_predicate or self.pred_predicate

        # ── Case 1: Missing extraction ────────────────────────────────────────
        if not self.pred_subject and not self.pred_predicate and not self.pred_object:
            self.error_type = ErrorType.MISSING_TRIPLE
            self.error_notes = "No triple extracted."
            return self

        # ── Case 2: Predicate matches gold ────────────────────────────────────
        if self._predicates_match(effective_pred, self.gold_predicate):
            if self._args_match():
                self.error_type = ErrorType.CORRECT
            else:
                self.error_type = ErrorType.ARGUMENT_SPAN
                self.error_notes = (
                    f"Predicate correct ({effective_pred}) but "
                    f"args mismatch: subj '{self.pred_subject}' vs '{self.gold_subject}', "
                    f"obj '{self.pred_object}' vs '{self.gold_object}'"
                )
            return self

        # ── Case 3: Predicate is wrong — classify which kind ──────────────────
        raw = self.pred_predicate.strip().lower()

        # Language Mixing: predicate is predominantly ASCII/English
        if self._is_language_mixing(raw):
            self.error_type = ErrorType.LANGUAGE_MIXING
            self.error_notes = (
                f"English predicate '{self.pred_predicate}' extracted for Hindi sentence. "
                f"Expected: {self.gold_predicate}"
            )
            return self

        # Predicate Placeholder: copula extracted as predicate
        if raw in HINDI_COPULAS or any(c in raw for c in HINDI_COPULAS):
            self.error_type = ErrorType.PREDICATE_PLACEHOLDER
            self.error_notes = (
                f"Copula '{self.pred_predicate}' used as predicate. "
                f"Implicit relation not surfaced. Expected: {self.gold_predicate}"
            )
            return self

        # Default: predicate surface form not aligned to DBpedia ontology
        self.error_type = ErrorType.PREDICATE_NORMALIZATION
        self.error_notes = (
            f"Predicate '{self.pred_predicate}' (aligned: '{self.aligned_predicate}') "
            f"≠ gold '{self.gold_predicate}'"
        )
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _predicates_match(self, pred: str, gold: str) -> bool:
        """Normalise and compare predicates."""
        def norm(s):
            return s.strip().lower().replace("dbo:", "").replace("dbp:", "")
        return norm(pred) == norm(gold)

    def _args_match(self) -> bool:
        def norm(s): return s.strip().lower()
        return (norm(self.pred_subject) == norm(self.gold_subject) and
                norm(self.pred_object)  == norm(self.gold_object))

    def _is_language_mixing(self, text: str) -> bool:
        """Returns True if text is predominantly ASCII (English)."""
        if not text:
            return False
        alpha_chars   = [c for c in text if c.isalpha()]
        ascii_alpha   = [c for c in alpha_chars if ord(c) < 128]
        if not alpha_chars:
            return False
        ratio = len(ascii_alpha) / len(alpha_chars)
        words = text.split()
        english_word_hit = any(w.lower() in ENGLISH_FUNCTION_WORDS for w in words)
        return ratio > 0.65 or english_word_hit

    def to_dict(self) -> dict:
        return {
            "sentence_id":          self.sentence_id,
            "sentence":             self.sentence,
            "system":               self.system,
            "gold":                 {"subject": self.gold_subject,
                                     "predicate": self.gold_predicate,
                                     "object": self.gold_object},
            "predicted":            {"subject": self.pred_subject,
                                     "predicate": self.pred_predicate,
                                     "aligned_predicate": self.aligned_predicate,
                                     "object": self.pred_object,
                                     "alignment_confidence": self.alignment_confidence},
            "error_type":           self.error_type.value if self.error_type else None,
            "error_notes":          self.error_notes,
        }


# ─── Taxonomy Aggregator ──────────────────────────────────────────────────────

class ErrorTaxonomy:
    """Collect ExtractionResult objects and compute evaluation metrics."""

    def __init__(self, system_name: str = "unknown"):
        self.system_name = system_name
        self.results: List[ExtractionResult] = []

    def add(self, result: ExtractionResult) -> None:
        result.classify_error()
        self.results.append(result)

    def add_batch(self, results: List[ExtractionResult]) -> None:
        for r in results:
            self.add(r)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def error_breakdown(self) -> Dict[str, dict]:
        """Per-error-type counts and percentages."""
        total = len(self.results)
        counts = Counter(r.error_type for r in self.results)
        return {
            et.value: {
                "count": counts.get(et, 0),
                "pct":   round(counts.get(et, 0) / total * 100, 1) if total else 0.0,
            }
            for et in ErrorType
        }

    def precision_recall_f1(self) -> dict:
        """Precision / Recall / F1 at the triple level."""
        total     = len(self.results)
        correct   = sum(1 for r in self.results if r.error_type == ErrorType.CORRECT)
        missing   = sum(1 for r in self.results if r.error_type == ErrorType.MISSING_TRIPLE)

        extracted = total - missing
        precision = correct / extracted if extracted > 0 else 0.0
        recall    = correct / total if total > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        return {
            "system":     self.system_name,
            "correct":    correct,
            "total":      total,
            "extracted":  extracted,
            "precision":  round(precision, 3),
            "recall":     round(recall, 3),
            "f1":         round(f1, 3),
        }

    def subject_accuracy(self) -> float:
        results_with_output = [r for r in self.results if r.error_type != ErrorType.MISSING_TRIPLE]
        if not results_with_output:
            return 0.0
        correct = sum(
            1 for r in results_with_output
            if r.pred_subject.strip().lower() == r.gold_subject.strip().lower()
        )
        return round(correct / len(results_with_output), 3)

    def object_accuracy(self) -> float:
        results_with_output = [r for r in self.results if r.error_type != ErrorType.MISSING_TRIPLE]
        if not results_with_output:
            return 0.0
        correct = sum(
            1 for r in results_with_output
            if r.pred_object.strip().lower() == r.gold_object.strip().lower()
        )
        return round(correct / len(results_with_output), 3)

    def predicate_accuracy(self, use_aligned: bool = True) -> float:
        results_with_output = [r for r in self.results if r.error_type != ErrorType.MISSING_TRIPLE]
        if not results_with_output:
            return 0.0
        def norm(s): return s.strip().lower().replace("dbo:", "").replace("dbp:", "")
        correct = sum(
            1 for r in results_with_output
            if norm((r.aligned_predicate if use_aligned and r.aligned_predicate
                     else r.pred_predicate)) == norm(r.gold_predicate)
        )
        return round(correct / len(results_with_output), 3)

    def summary(self) -> dict:
        return {
            "system":              self.system_name,
            "metrics":             self.precision_recall_f1(),
            "slot_accuracy": {
                "subject":         self.subject_accuracy(),
                "predicate_raw":   self.predicate_accuracy(use_aligned=False),
                "predicate_aligned": self.predicate_accuracy(use_aligned=True),
                "object":          self.object_accuracy(),
            },
            "error_breakdown":     self.error_breakdown(),
        }

    def print_summary(self) -> None:
        """Pretty-print summary to console."""
        from tabulate import tabulate
        s = self.summary()

        print(f"\n{'='*60}")
        print(f"  System: {self.system_name.upper()}")
        print(f"{'='*60}")

        m = s["metrics"]
        print(tabulate(
            [["Precision", m["precision"]],
             ["Recall",    m["recall"]],
             ["F1",        m["f1"]],
             ["Correct",   f"{m['correct']}/{m['total']}"]],
            headers=["Metric", "Value"], tablefmt="rounded_outline"
        ))

        sa = s["slot_accuracy"]
        print("\nSlot Accuracy:")
        print(tabulate(
            [["Subject",            sa["subject"]],
             ["Predicate (raw)",    sa["predicate_raw"]],
             ["Predicate (aligned)",sa["predicate_aligned"]],
             ["Object",             sa["object"]]],
            headers=["Slot", "Accuracy"], tablefmt="rounded_outline"
        ))

        print("\nError Breakdown:")
        eb = s["error_breakdown"]
        rows = [[et, d["count"], f"{d['pct']}%"] for et, d in eb.items() if d["count"] > 0]
        print(tabulate(rows, headers=["Error Type", "Count", "%"], tablefmt="rounded_outline"))

    def save(self, path: str) -> None:
        """Save full results to JSONL + summary to JSON."""
        import os
        base = path.rstrip("/")
        os.makedirs(base, exist_ok=True)

        with open(f"{base}/{self.system_name}_results.jsonl", "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        with open(f"{base}/{self.system_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

        print(f"Results saved to {base}/")


# ─── Ablation Table Builder ───────────────────────────────────────────────────

def build_ablation_table(taxonomies: List[ErrorTaxonomy]) -> str:
    """Build the Phase 1 milestone deliverable: ablation table comparing all systems."""
    from tabulate import tabulate

    headers = ["System", "Precision", "Recall", "F1",
               "Subj Acc", "Pred Acc (raw)", "Pred Acc (aligned)", "Obj Acc",
               "Norm Fail", "Placeholder", "Lang Mix", "Arg Span", "Missing"]

    rows = []
    for tax in taxonomies:
        s  = tax.summary()
        m  = s["metrics"]
        sa = s["slot_accuracy"]
        eb = s["error_breakdown"]

        rows.append([
            tax.system_name,
            m["precision"], m["recall"], m["f1"],
            sa["subject"], sa["predicate_raw"], sa["predicate_aligned"], sa["object"],
            eb[ErrorType.PREDICATE_NORMALIZATION.value]["count"],
            eb[ErrorType.PREDICATE_PLACEHOLDER.value]["count"],
            eb[ErrorType.LANGUAGE_MIXING.value]["count"],
            eb[ErrorType.ARGUMENT_SPAN.value]["count"],
            eb[ErrorType.MISSING_TRIPLE.value]["count"],
        ])

    return tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".3f")
