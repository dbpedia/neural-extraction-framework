import json
import os
import time
import sys
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

NUM_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
API_KEY     = sys.argv[2] if len(sys.argv) > 2 else "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"

WIKI_CHUNKS = [f"/home/nsingh/wiki_chunk_{i}_scored.jsonl" for i in [0, 1, 2]]
OUTPUT_FILE = "/home/nsingh/wiki_val_coref_resolved.jsonl"

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL   = "openai/gpt-oss-120b"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

PRONOUNS = [
    "उन्होंने", "उसने", "वह", "वे", "इसे", "उसे", "उसका", "उसकी", "उसके",
    "इसका", "इसकी", "इसके", "यह", "ये", "इन्होंने", "उनका", "उनकी", "उनके"
]

RESOLVE_PROMPT = """You are checking whether a pronoun in a Hindi sentence can be resolved using only the sentence itself.

Task: The sentence below contains a pronoun (उन्होंने, वह, यह, उसका, etc.) as the subject or object of a core fact. Decide whether the entity that pronoun refers to is named somewhere else in the SAME sentence.

If yes: rewrite the full sentence, replacing the pronoun with the actual entity name, keeping everything else exactly as is.
If no (the entity is only implied, or would need a previous sentence to know): respond with exactly NONE.

Output only the rewritten sentence, or NONE. Nothing else — no explanation.

Examples:

Sentence: पूरा शहर सामान आकार के छह भागों में बँटा है और यह 111 फुट चौड़ी सड़कों से विभाजित है।
Output: पूरा शहर सामान आकार के छह भागों में बँटा है और शहर 111 फुट चौड़ी सड़कों से विभाजित है।

Sentence: मोहम्मद पुर कोआरी: पुसा प्रखंड में स्थित यह एक ऐतिहासिक गाँव है।
Output: मोहम्मद पुर कोआरी: पुसा प्रखंड में स्थित मोहम्मद पुर कोआरी एक ऐतिहासिक गाँव है।

Sentence: गहमर से अतिरिक्त लगाव के कारण उन्होंने अपने नाम के साथ अपने इस ननिहाल को जोड़ लिया।
Output: NONE

Sentence: उन्होंने एक स्वतंत्र उम्मीदवार के रूप में रायगढ़ नगर निगम चुनाव लड़ा और 33,168 मत प्राप्त किए।
Output: NONE

Now do this one:

Sentence: {sentence}
Output:"""

EXTRACT_PROMPT = """You are an expert Hindi linguist. Given a real Hindi sentence, extract ALL subject-relation-object triplets.

CORE VERB TRIPLETS:
- Find every verb phrase as the relation
- Subject = entity performing the action
- Object = what the action is directed at
- Use EXACT spans from the sentence

PROPERTY RELATIONS (MANDATORY):
- Every adjective: subject=adjective, relation=property, object=noun it modifies
- Every possessive: subject=possessor, relation=property, object=possessed noun
- Every number: subject=number, relation=property, object=noun
- Every temporal phrase: subject=time phrase, relation=property, object=verb

OUTPUT - valid JSON only, no markdown:
{{"thought_process": "...", "extracted_triplets": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Sentence: {sentence}"""


def has_pronoun(text):
    text = (text or "").strip()
    return any(text == p or text.startswith(p + " ") for p in PRONOUNS)


def flag_pronoun_entries(entries):
    flagged = []
    for e in entries:
        content = json.loads(e["messages"][2]["content"])
        core = [t for t in content.get("extracted_triplets", []) if t.get("relation", "").strip() != "property"]
        for t in core:
            if has_pronoun(t.get("subject")) or has_pronoun(t.get("object")):
                flagged.append(e)
                break
    return flagged


def call_model(prompt, max_tokens=1500, retries=3):
    for attempt in range(retries + 1):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens
            }, timeout=90)

            if resp.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            msg = resp.json()["choices"][0]["message"]
            raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            if not raw:
                raise ValueError("empty response")
            return raw
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return None


def resolve_and_reextract(entry):
    sentence = entry["messages"][1]["content"]

    resolved = call_model(RESOLVE_PROMPT.format(sentence=sentence), max_tokens=400)
    if resolved is None:
        return entry, "error"

    resolved = resolved.strip()
    if resolved == "NONE" or not resolved:
        return entry, "unresolved"

    raw = call_model(EXTRACT_PROMPT.format(sentence=resolved), max_tokens=2000)
    if raw is None:
        return entry, "error"

    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        return entry, "error"

    try:
        parsed = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return entry, "error"

    if "extracted_triplets" not in parsed:
        return entry, "error"

    new_entry = {
        "messages": [
            entry["messages"][0],
            {"role": "user", "content": resolved},
            {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
        ],
        "source": entry.get("source"),
        "article": entry.get("article"),
        "judgement": entry.get("judgement"),
        "original_sentence": sentence,
    }
    return new_entry, "resolved"


write_lock = threading.Lock()
counter_lock = threading.Lock()
resolved_count = 0
unresolved_count = 0
error_count = 0


def process(entry):
    global resolved_count, unresolved_count, error_count
    result, status = resolve_and_reextract(entry)

    with write_lock:
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps({"status": status, "entry": result}, ensure_ascii=False) + "\n")

    with counter_lock:
        if status == "resolved":
            resolved_count += 1
        elif status == "unresolved":
            unresolved_count += 1
        else:
            error_count += 1
        total = resolved_count + unresolved_count + error_count
        if total % 10 == 0:
            print(f"[{total}] resolved:{resolved_count} unresolved:{unresolved_count} error:{error_count}", flush=True)


all_entries = []
for path in WIKI_CHUNKS:
    with open(path) as f:
        all_entries += [json.loads(l) for l in f if l.strip()]

validation = [e for e in all_entries if e.get("judgement", {}).get("score", 0) >= 9]
flagged = flag_pronoun_entries(validation)
print(f"Validation candidates: {len(validation)}, flagged for coref: {len(flagged)}")

done_sentences = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                sent = row["entry"]["messages"][1]["content"] if row["status"] == "resolved" else row["entry"]["original_sentence"] if "original_sentence" in row["entry"] else row["entry"]["messages"][1]["content"]
                done_sentences.add(sent)
    print(f"Resuming — {len(done_sentences)} already processed")

remaining = [e for e in flagged if e["messages"][1]["content"] not in done_sentences]
print(f"Remaining: {len(remaining)}")

start = time.time()
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    futures = [pool.submit(process, e) for e in remaining]
    for f in as_completed(futures):
        pass

elapsed = time.time() - start
print(f"\nDone in {elapsed/60:.1f} min")
print(f"Resolved: {resolved_count} | Unresolved (dropped): {unresolved_count} | Errors: {error_count}")