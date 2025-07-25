import pandas as pd
from hybrid_linking.gemini_api import batch_normalize_entities_gemini, call_gemini
import json

entities = [
    "Apple",
    "Microsoft",
    "Paris",
    "Barack Obama",
    "Tesla",
    "Amazon",
    "London",
    "Python (programming language)",
    "Eiffel Tower"
]

# Build the prompt for debugging
prompt = (
    "Given the following list of entity mentions, return the canonical DBpedia name for each. "
    "Respond as a JSON list of objects with fields 'mention' and 'canonical_name'.\n\n"
    "Entities:\n" +
    "\n".join(f"- {e}" for e in entities)
)
print("Prompt sent to Gemini:\n", prompt)

try:
    # Call Gemini directly for debugging
    raw_response = call_gemini(prompt)
    print("\nRaw Gemini response:\n", raw_response)
except Exception as e:
    print("\nException during Gemini API call:", e)
    raw_response = None

try:
    # Now use the batch function
    results = batch_normalize_entities_gemini(entities)
    print("\nDataFrame output:")
    df = pd.DataFrame(results)
    print(df)
    print("\nJSON output:")
    print(json.dumps(results, indent=2))
except Exception as e:
    print("\nException during batch normalization:", e)
    if raw_response:
        print("\nRaw Gemini response (for debugging):\n", raw_response)