"""
merge_hitl_feedback.py — Closes the HITL feedback loop, per the original
proposal's Phase 3 design. Takes a corrections .jsonl file (downloaded
from the HITL app after a review session) and:

  1. Updates normalization_cache_k40.json with human-corrected DBO
     mappings ("modify" decisions) — fixes future normalization runs
     immediately, no re-running the LLM disambiguation needed.
  2. Builds a verified training pairs file from "accept" and "modify"
     decisions — same (predicate, gold_dbo) format as your F2LLM
     training data, ready to fold into a future fine-tuning round.
     Appends to the existing file across multiple review sessions,
     deduping by predicate so re-reviewing the same item twice doesn't
     create duplicate training pairs.
  3. Logs "reject" decisions separately with their error_type, for
     diagnostic review — these are NOT added as training data, since
     a reject means something is fundamentally broken (wrong subject/
     object span, missing relation, etc.), not just a wrong DBO.

Run:
    python3 merge_hitl_feedback.py --corrections /path/to/hitl_corrections.jsonl
"""

import json
import argparse
import os

CACHE_FILE = "/home/nsingh/normalization_cache_k40.json"
VERIFIED_TRAIN_FILE = "/home/nsingh/hitl_verified_train_pairs.jsonl"
REJECTED_LOG_FILE = "/home/nsingh/hitl_rejected_log.jsonl"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_existing_verified():
    if os.path.exists(VERIFIED_TRAIN_FILE):
        with open(VERIFIED_TRAIN_FILE, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    return []


def save_verified(entries):
    with open(VERIFIED_TRAIN_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_existing_rejected():
    if os.path.exists(REJECTED_LOG_FILE):
        with open(REJECTED_LOG_FILE, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    return []


def save_rejected(entries):
    with open(REJECTED_LOG_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", required=True,
                         help="Path to the downloaded hitl_corrections.jsonl file")
    args = parser.parse_args()

    print("=" * 60)
    print("HITL Feedback Merge")
    print("=" * 60)

    print(f"\nLoading corrections from: {args.corrections}")
    with open(args.corrections, encoding="utf-8") as f:
        corrections = [json.loads(l) for l in f if l.strip()]
    print(f"Total decisions: {len(corrections)}")

    accepted = [c for c in corrections if c["action"] == "accept"]
    modified = [c for c in corrections if c["action"] == "modify"]
    rejected = [c for c in corrections if c["action"] == "reject"]

    print(f"  Accepted: {len(accepted)}")
    print(f"  Modified: {len(modified)}")
    print(f"  Rejected: {len(rejected)}")

    # ── 1. Update the cache with corrected mappings ──
    print("\n--- Step 1: Updating normalization cache ---")
    cache = load_cache()
    print(f"  Cache currently has {len(cache)} entries")

    cache_updates = 0
    for c in modified:
        relation = c["relation"]
        corrected_dbo = c["final_dbo_uri"]
        old_value = cache.get(relation)
        if old_value != corrected_dbo:
            cache[relation] = corrected_dbo
            cache_updates += 1

    if cache_updates:
        save_cache(cache)
        print(f"  Updated {cache_updates} entries (from 'modify' decisions)")
    else:
        print("  No cache updates needed")

    # ── 2. Build verified training pairs (accept + modify) ──
    print("\n--- Step 2: Building verified training pairs ---")
    existing_verified = load_existing_verified()
    existing_predicates = {e["predicate"] for e in existing_verified}
    print(f"  Existing verified pairs: {len(existing_verified)}")

    new_pairs = 0
    for c in accepted:
        predicate = c["relation"]
        gold_dbo = c["suggested_dbo_uri"]
        if predicate not in existing_predicates and gold_dbo:
            existing_verified.append({
                "predicate": predicate,
                "gold_dbo": gold_dbo,
                "source": "hitl_accept",
            })
            existing_predicates.add(predicate)
            new_pairs += 1

    for c in modified:
        predicate = c["relation"]
        gold_dbo = c["final_dbo_uri"]
        if predicate not in existing_predicates and gold_dbo:
            existing_verified.append({
                "predicate": predicate,
                "gold_dbo": gold_dbo,
                "source": "hitl_modify",
            })
            existing_predicates.add(predicate)
            new_pairs += 1

    save_verified(existing_verified)
    print(f"  Added {new_pairs} new verified pairs")
    print(f"  Total verified pairs now: {len(existing_verified)}")

    # ── 3. Log rejections separately ──
    print("\n--- Step 3: Logging rejections ---")
    existing_rejected = load_existing_rejected()
    print(f"  Existing rejected log entries: {len(existing_rejected)}")

    for c in rejected:
        existing_rejected.append({
            "sentence": c.get("sentence", ""),
            "subject": c.get("subject", ""),
            "relation": c.get("relation", ""),
            "object": c.get("object", ""),
            "suggested_dbo_uri": c.get("suggested_dbo_uri"),
            "error_type": c.get("error_type", ""),
            "note": c.get("note", ""),
            "timestamp": c.get("timestamp", ""),
        })

    save_rejected(existing_rejected)
    print(f"  Added {len(rejected)} new rejection entries")
    print(f"  Total rejected log entries now: {len(existing_rejected)}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Cache entries fixed:              {cache_updates}")
    print(f"New verified training pairs added: {new_pairs}")
    print(f"New rejections logged:             {len(rejected)}")
    print(f"\nFiles updated:")
    print(f"  {CACHE_FILE}")
    print(f"  {VERIFIED_TRAIN_FILE}")
    print(f"  {REJECTED_LOG_FILE}")


if __name__ == "__main__":
    main()
