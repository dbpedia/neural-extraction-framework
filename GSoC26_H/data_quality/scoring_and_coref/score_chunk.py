import sys
import json
import os
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_ID    = int(sys.argv[1])
NUM_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
API_KEY     = sys.argv[3] if len(sys.argv) > 3 else "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"

INPUT_FILE  = f"/home/nsingh/wiki_chunk_{CHUNK_ID}.jsonl"
OUTPUT_FILE = f"/home/nsingh/wiki_chunk_{CHUNK_ID}_scored.jsonl"

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL   = "openai/gpt-oss-120b"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

JUDGING_PROMPT = """You are an expert annotator evaluating Hindi subject-relation-object triplet extractions, aligned with the "Bench-HindIE" benchmark's implicit rules.

**Evaluation Criteria (CRITICAL):**
You will evaluate each data point on a scale of 1 to 10 based on the following criteria.

1. **Source Sentence Quality (Weight: 20%):**
   - 10/10: Grammatically correct, natural, complex Hindi sentence
   - <5/10: Awkward, unnatural, or nonsensical sentence

2. **Span Exactness (Weight: 30%):**
   - 10/10: ALL subjects, relations, objects are exact contiguous substrings of the sentence
   - <5/10: Any span is paraphrased, missing prepositions, or altered

3. **Semantic Correctness of Core Triplets (Weight: 30%):**
   - 10/10: Main verb-based triplets accurately capture primary events and facts
   - <5/10: Core triplets misinterpret sentence meaning or reverse action direction

4. **Property Relation Quality (Weight: 20%):**
   - 10/10: property relations used precisely for adjectives, possessives, attributes
   - 7/10: Generally correct but misses some opportunities
   - <5/10: Used incorrectly or misses many obvious attributes

**Output Format:** Return ONLY a valid JSON object:
{"score": <1-10>, "justification": "<explanation>"}

**Few-Shot Examples:**

Example 1 (GOOD - score 10 - high property density, exact spans):
Sentence: ऐसे कई नाम वास्तव में आधिकारिक फैक्टरी नाम नहीं होते हैं।
Triplets: [{"subject": "ऐसे कई नाम", "relation": "नहीं होते हैं", "object": "आधिकारिक फैक्टरी नाम"}, {"subject": "ऐसे", "relation": "property", "object": "नाम"}, {"subject": "कई", "relation": "property", "object": "नाम"}, {"subject": "आधिकारिक", "relation": "property", "object": "फैक्टरी नाम"}]
Judgement: {"score": 10, "justification": "High property density (75% of triplets) is NOT penalized here because every span is an exact contiguous substring and each property relation captures a genuine adjective-noun attribute. Property-dense sentences from natural text should score well when spans are correct."}

Example 2 (BAD - score 3 - nonsensical property relation):
Sentence: यहां हर धर्म के अनुयायी का स्वागत किया जाता है।
Triplets: [{"subject": "स्वागत", "relation": "किया जाता है", "object": "हर धर्म के अनुयायी"}, {"subject": "हर", "relation": "property", "object": "धर्म"}, {"subject": "धर्म", "relation": "property", "object": "अनुयायी"}, {"subject": "अनुयायी", "relation": "property", "object": "स्वागत"}]
Judgement: {"score": 3, "justification": "Critical error: 'अनुयायी to property to स्वागत' is semantically nonsensical - followers are not an attribute of welcome. This is a genuine property-relation logic failure, not merely high property density."}

Example 3 (BAD - score 5 - incomplete span, missing postposition):
Sentence: भाषा का अध्ययन कर हम संस्कृति को समझ सकते हैं।
Triplets: [{"subject": "हम", "relation": "समझ सकते हैं", "object": "संस्कृति"}, {"subject": "भाषा", "relation": "property", "object": "अध्ययन"}]
Judgement: {"score": 5, "justification": "Object span 'संस्कृति' omits the postposition 'को' present in the original sentence, violating span exactness. The property relation 'भाषा to property to अध्ययन' is also a weak generic link rather than a true adjective-noun property."}

Example 4 (GOOD - score 7 - source sentence flawed but triplets scored fairly):
Sentence: ऐसी कितनी ही गाथाएँ ब्राह्मण ग्रंथों में उदधृत की गई है।
Triplets: [{"subject": "गाथाएँ", "relation": "उदधृत की गई है", "object": "ब्राह्मण ग्रंथों में"}, {"subject": "ऐसी", "relation": "property", "object": "गाथाएँ"}, {"subject": "कितनी ही", "relation": "property", "object": "गाथाएँ"}, {"subject": "ब्राह्मण", "relation": "property", "object": "ग्रंथों"}]
Judgement: {"score": 7, "justification": "Source sentence contains a spelling error (उदधृत instead of उद्धृत) and a minor subject-verb agreement issue, which caps sentence quality. However, all triplet spans are exact substrings and property relations are semantically valid, so the extraction itself is not penalized further."}

Example 5 (GOOD - score 9 - simple sentence, single clean triplet, no unnecessary properties):
Sentence: यहां से बेगूसराय, दलसिंग्सराय, समस्तीपुर, पटना, वैशाली के लिए बस मिलता है।
Triplets: [{"subject": "बस", "relation": "मिलता है", "object": "बेगूसराय, दलसिंग्सराय, समस्तीपुर, पटना, वैशाली के लिए"}]
Judgement: {"score": 9, "justification": "The sentence is grammatically correct and natural. All three spans are exact contiguous substrings. The core triplet accurately captures the meaning. No property relations were needed here since there are no standalone adjectives to extract, and none were incorrectly added."}

Now evaluate this data point:
"""

rate_limit_lock = threading.Lock()
rate_limit_until = 0.0
write_lock = threading.Lock()
counter_lock = threading.Lock()
success = 0
failed = 0
completed = 0

def wait_if_rate_limited():
    printed = False
    while True:
        with rate_limit_lock:
            wait_time = rate_limit_until - time.time()
        if wait_time <= 0:
            return
        if not printed:
            print(f"[DEBUG] Rate limited, waiting ~{wait_time:.0f}s", flush=True)
            printed = True
        time.sleep(min(wait_time, 5))

def trigger_rate_limit_backoff(seconds):
    global rate_limit_until
    with rate_limit_lock:
        rate_limit_until = max(rate_limit_until, time.time() + seconds)

def score_entry(sentence, triplets, thought, retries=3):
    data_str = f"Sentence: `{sentence}`\n\nThought: {thought[:200]}\n\nTriplets:\n{json.dumps(triplets, indent=2, ensure_ascii=False)}"
    prompt = JUDGING_PROMPT + data_str

    for attempt in range(retries + 1):
        wait_if_rate_limited()
        t0 = time.time()
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 2500
            }, timeout=90)

            if resp.status_code == 429:
                backoff = 30 * (attempt + 1)
                trigger_rate_limit_backoff(backoff)
                time.sleep(backoff)
                continue
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            msg = resp.json()["choices"][0]["message"]
            raw = msg.get("content") or msg.get("reasoning_content") or ""
            if not raw.strip():
                raise ValueError("Empty response")

            raw   = raw.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end])
                if "score" in parsed and "justification" in parsed:
                    return parsed
        except Exception as e:
            print(f"[DEBUG] Exception attempt {attempt}: {type(e).__name__}: {e}", flush=True)
            if attempt < retries:
                time.sleep(1)

    return {"score": -1, "justification": "API error after retries"}

def process_entry(entry):
    global success, failed, completed
    sentence = entry["messages"][1]["content"]
    content  = json.loads(entry["messages"][2]["content"])
    thought  = content.get("thought_process", "")
    triplets = content.get("extracted_triplets", [])

    judgement = score_entry(sentence, triplets, thought)
    entry["judgement"] = judgement

    with write_lock:
        if judgement["score"] > 0:
            with open(OUTPUT_FILE, "a") as out_f:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with counter_lock:
        if judgement["score"] > 0:
            success += 1
        else:
            failed += 1
        completed += 1
        if completed % 5 == 0:
            print(f"[{completed}] ok:{success} err:{failed}", flush=True)

with open(INPUT_FILE) as f:
    entries = [json.loads(l) for l in f if l.strip()]
print(f"Chunk {CHUNK_ID}: {len(entries)} entries, {NUM_WORKERS} workers")

done = set()
if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e.get("judgement", {}).get("score", -1) > 0:
                    done.add(e["messages"][1]["content"])
    print(f"Resuming — {len(done)} already scored")

remaining = [e for e in entries if e["messages"][1]["content"] not in done]
print(f"Remaining: {len(remaining)}")

start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_entry, e) for e in remaining]
    for f in as_completed(futures):
        pass

elapsed = time.time() - start_time
print(f"Chunk {CHUNK_ID} done in {elapsed/60:.1f} min. Scored: {success} | Errors: {failed}")