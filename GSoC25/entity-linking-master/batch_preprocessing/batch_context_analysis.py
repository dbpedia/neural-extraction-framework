import math
import json
import re
from typing import List, Dict, Union
import pandas as pd
from hybrid_linking.gemini_api import call_gemini

def batch_context_analysis(
    entity_contexts: List[Dict[str, str]],
    chunk_size: int = 10,
    output_format: str = "dataframe"
) -> Union[pd.DataFrame, List[Dict], str]:
    """
    Batch context analysis using Gemini, with chunking, progress, and robust error handling.
    Args:
        entity_contexts: List of dicts with 'mention' and 'context'.
        chunk_size: Max number of pairs per Gemini call (default: 10).
        output_format: 'dataframe', 'json', or 'list'.
    Returns:
        DataFrame, JSON string, or list of dicts with context analysis for each pair.
    """
    results = []
    total = len(entity_contexts)
    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        batch = entity_contexts[i * chunk_size : (i + 1) * chunk_size]
        print(f"[PROGRESS] Processing batch {i+1}/{n_chunks} ({len(batch)} pairs)...")
        prompt = (
            "Given the following list of entity mentions and their contexts, "
            "analyze each pair and return a JSON list of objects with fields: "
            "'mention', 'context', 'entity_type' (person, company, place, product, concept, or other), "
            "'confidence' (0-1), 'keywords' (list), and 'description' (brief description).\n\n"
            "Pairs:\n" +
            "\n".join(f"- mention: {e['mention']}\n  context: {e['context']}" for e in batch)
        )
        try:
            response = call_gemini(prompt)
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                batch_results = json.loads(match.group())
            else:
                batch_results = json.loads(response)
        except Exception as e:
            print(f"[ERROR] Gemini batch failed for batch {i+1}: {e}")
            batch_results = [{**e, "entity_type": None, "confidence": None, "keywords": [], "description": None} for e in batch]
        # Ensure all batch pairs are present
        mention_context_set = {(e['mention'], e['context']) for e in batch}
        found_pairs = {(r.get('mention'), r.get('context')) for r in batch_results}
        for missing in mention_context_set - found_pairs:
            mention, context = missing
            batch_results.append({"mention": mention, "context": context, "entity_type": None, "confidence": None, "keywords": [], "description": None})
        results.extend(batch_results)
        print(f"[PROGRESS] Completed batch {i+1}/{n_chunks}.")
    # Remove duplicates (keep first occurrence)
    seen = set()
    deduped = []
    for r in results:
        key = (r["mention"], r["context"])
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    if output_format == "dataframe":
        return pd.DataFrame(deduped)
    elif output_format == "json":
        return json.dumps(deduped, indent=2)
    else:
        return deduped 