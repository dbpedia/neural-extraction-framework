import requests
import os
from dotenv import load_dotenv
import json
import re
import pathlib

# Load environment variables from config.env

root_dir = pathlib.Path(__file__).resolve().parent.parent
env_path = root_dir/'config.env'

load_dotenv(dotenv_path=env_path)


# Try to get API key from environment variables
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY'))
print(f"[DEBUG] GEMINI_API_KEY loaded: {GEMINI_API_KEY[:10]}..." if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY' else "[DEBUG] GEMINI_API_KEY not found, using default")

# Function to update API key at runtime
def set_api_key(api_key: str):
    """Set the API key at runtime"""
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key
    print(f"[DEBUG] API key updated to: {api_key[:10]}...")
GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'


def call_gemini(prompt: str) -> str:
    """
    Call Gemini API with a prompt and return the generated text.
    """
    print("[DEBUG] Entering call_gemini")
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=30)
        print(f"[DEBUG] Gemini API status code: {response.status_code}")
        if not response.ok:
            print(f"[DEBUG] Gemini API response content: {response.content}")
            response.raise_for_status()
        result = response.json()
        # Extract the generated text
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            print("[DEBUG] Exiting call_gemini successfully")
            return text
        except Exception as e:
            print(f"[DEBUG] Error extracting text from Gemini response: {e}")
            print(f"[DEBUG] Full Gemini response: {result}")
            return str(result)
    except Exception as e:
        print(f"[DEBUG] Exception in call_gemini: {e}")
        raise


def batch_normalize_entities_gemini(entities: list[str]) -> list[dict]:
    """
    Given a list of entity mentions, return a list of dicts with 'mention' and 'canonical_name',
    using a single Gemini API call.
    """
    prompt = (
        "Given the following list of entity mentions, return the canonical DBpedia name for each. "
        "Respond as a JSON list of objects with fields 'mention' and 'canonical_name'.\n\n"
        "Entities:\n" +
        "\n".join(f"- {e}" for e in entities)
    )
    response = call_gemini(prompt)
    # Try to extract the JSON list from the response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        try:
            return json.loads(response)
        except Exception:
            return [{"mention": e, "canonical_name": None} for e in entities] 