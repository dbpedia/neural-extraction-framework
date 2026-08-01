import json, os, time, requests

API_KEY     = "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"
API_URL     = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS     = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
MODEL       = "openai/gpt-oss-120b"
INPUT_FILE  = os.path.expanduser("~/wikipedia_synthetic_data_clean.jsonl")
OUTPUT_FILE = os.path.expanduser("~/wikipedia_synthetic_data_scored.jsonl")

JUDGING_PROMPT_TEMPLATE = """
**Your Role:** You are an expert Hindi linguist and a meticulous data quality analyst. Your task is to act as a judge, evaluating the quality of synthetically generated data points for a Subject-Relation-Object (SRO) triplet extraction task. Each data point consists of a Hindi sentence and its corresponding SRO extractions, which will be used to fine-tune a smaller language model. Your evaluation must be strict and aligned with the "Bench-HindIE" benchmark's implicit rules.

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

Example 1 (GOOD - score 10):
Sentence: कृत्रिम बुद्धिमत्ता को दर्शाता है वह तकनीकी क्षेत्र, जिसका आरम्भ 2010 में राष्ट्रीय विज्ञान संस्थानों द्वारा शुरू हुई।
Triplets: [{"subject": "वह तकनीकी क्षेत्र", "relation": "दर्शाता है", "object": "कृत्रिम बुद्धिमत्ता को"}, {"subject": "आरम्भ", "relation": "शुरू हुई", "object": "राष्ट्रीय विज्ञान संस्थानों द्वारा"}]
Judgement: {"score": 10, "justification": "Excellent. Complex grammatically correct sentence. All spans are exact contiguous substrings. Core triplets semantically correct. Property relations precise and comprehensive."}

Example 2 (BAD - score 4):
Sentence: डॉ. अलीशा बत्रा को जीवविज्ञान के क्षेत्र में राष्ट्रीय विज्ञान पुरस्कार से सम्मानित किया गया।
Triplets: [{"subject": "डॉ. अलीशा बत्रा", "relation": "सम्मानित किया गया", "object": "जीवविज्ञान के क्षेत्र में राष्ट्रीय विज्ञान पुरस्कार से"}]
Judgement: {"score": 4, "justification": "Critical error: object span is overly broad, lumping reason and award together. This teaches harmful extraction patterns. Should be broken into separate precise triplets."}

Example 3 (BAD - score 2):
Sentence: हिमालय व कश्मीर से प्रसिद्ध नदी भूमिगत जलभण्डार से निरंतर प्रवाहित होता है।
Triplets: [{"subject": "हिमालय व कश्मीर से प्रसिद्ध नदी", "relation": "प्रवाहित होता है", "object": "भूमिगत जलभण्डार"}]
Judgement: {"score": 2, "justification": "Critical semantic error: object misses preposition से, reversing the direction of flow. River flows FROM the reservoir, not TO it. This harmful error would teach wrong patterns."}

Now evaluate this data point:
"""

def score_entry(sentence, triplets, thought_process, retries=3):
    data_str = f"Sentence: `{sentence}`\n\nThought Process: {thought_process[:200]}\n\nExtractions:\n{json.dumps(triplets, indent=2, ensure_ascii=False)}"
    prompt   = JUDGING_PROMPT_TEMPLATE + data_str

    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1000
            }, timeout=120)

            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            choice = resp.json()["choices"][0]["message"]
            raw = choice.get("content") or choice.get("reasoning_content") or ""
            if not raw or raw.strip() == "None":
                raise ValueError("Empty response")
            if not raw.strip():
                raise ValueError("Empty response")

            raw = raw.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]

            parsed = json.loads(raw)
            if "score" in parsed and "justification" in parsed:
                return parsed

        except Exception as e:
            if attempt < retries:
                time.sleep(3)

    return {"score": -1, "justification": "API error after retries"}

# Load input
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_entries = [json.loads(l) for l in f if l.strip()]
print(f"Total entries to score: {len(all_entries)}")

# Resume from checkpoint
done_sentences = set()
if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("judgement", {}).get("score", -1) > 0:
                    done_sentences.add(item["messages"][1]["content"])
    print(f"Resuming — {len(done_sentences)} already scored.")

remaining = [e for e in all_entries if e["messages"][1]["content"] not in done_sentences]
print(f"Remaining: {len(remaining)}")

# Run scoring
success = failed = completed = 0
start_time = time.time()

with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
    for entry in remaining:
        sentence = entry["messages"][1]["content"]
        content  = json.loads(entry["messages"][2]["content"])
        thought  = content.get("thought_process", "")
        triplets = content.get("extracted_triplets", [])

        judgement = score_entry(sentence, triplets, thought)
        entry["judgement"] = judgement

        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if judgement["score"] >= 1:
            success += 1
        else:
            failed += 1

        completed += 1
        if completed % 10 == 0:
            out_f.flush()
        if completed % 100 == 0:
            elapsed = time.time() - start_time
            rate    = completed / elapsed * 3600
            eta     = (len(remaining) - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{len(remaining)}] ok:{success} err:{failed} | {rate:.0f}/hr | ETA {eta:.1f}h")

out_f.flush()
print(f"\nDone. Scored: {success} | Errors: {failed}")
print(f"Output: {OUTPUT_FILE}")
