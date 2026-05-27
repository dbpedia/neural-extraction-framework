import math
import json
import re
from typing import List, Dict, Union, Optional
import pandas as pd
from hybrid_linking.gemini_api import call_gemini

def batch_canonical_name_normalization(
    entities: List[str],
    chunk_size: int = 20,
    output_format: str = "dataframe"
) -> Union[pd.DataFrame, List[Dict], str]:
    """
    Batch canonical name normalization using Gemini, with chunking, progress, and robust error handling.
    Args:
        entities: List of entity mentions.
        chunk_size: Max number of entities per Gemini call (default: 20).
        output_format: 'dataframe', 'json', or 'list'.
    Returns:
        DataFrame, JSON string, or list of dicts with 'mention' and 'canonical_name'.
    """
    results = []
    total = len(entities)
    n_chunks = math.ceil(total / chunk_size)
    for i in range(n_chunks):
        batch = entities[i * chunk_size : (i + 1) * chunk_size]
        print(f"[PROGRESS] Processing batch {i+1}/{n_chunks} ({len(batch)} names)...")
        prompt = (
            "Given the following list of entity mentions, return the canonical DBpedia name for each. "
            "Respond as a JSON list of objects with fields 'mention' and 'canonical_name'.\n\n"
            "Entities:\n" +
            "\n".join(f"- {e}" for e in batch)
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
            batch_results = [{"mention": e, "canonical_name": None} for e in batch]
        # Ensure all batch entities are present
        mention_set = set(batch)
        found_mentions = {r["mention"] for r in batch_results if "mention" in r}
        for missing in mention_set - found_mentions:
            batch_results.append({"mention": missing, "canonical_name": None})
        results.extend(batch_results)
        print(f"[PROGRESS] Completed batch {i+1}/{n_chunks}.")
    # Remove duplicates (keep first occurrence)
    seen = set()
    deduped = []
    for r in results:
        key = r["mention"]
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    if output_format == "dataframe":
        return pd.DataFrame(deduped)
    elif output_format == "json":
        return json.dumps(deduped, indent=2)
    else:
        return deduped 