import openai
import os
import json
import time
from tqdm import tqdm

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGING_MODEL = "gpt-4o-mini" 
INPUT_FILE = "synthetic_bench_hindie_data_gpt_oss_120b.jsonl"
OUTPUT_FILE = "synthetic_bench_hindie_data_gpt_oss_120b-scored.jsonl"

JUDGING_PROMPT_TEMPLATE = """
**Your Role:** You are an expert Hindi linguist and a meticulous data quality analyst. Your task is to act as a judge, evaluating the quality of synthetically generated data points for a Subject-Relation-Object (SRO) triplet extraction task. Each data point consists of a Hindi sentence and its corresponding SRO extractions, which will be used to fine-tune a smaller language model. Your evaluation must be strict and aligned with the "Bench-HindIE" benchmark's implicit rules.

**Evaluation Criteria (CRITICAL):**
You will evaluate each data point on a scale of 1 to 10 based on the following criteria. Your primary goal is to determine if this data point is "good enough for fine-tuning," meaning it will teach the smaller model correct, precise, and complete extraction patterns.

1.  **Source Sentence Quality (Weight: 20%):**
    *   **10/10:** The Hindi sentence is grammatically correct, natural, and complex enough to be a valuable training example.
    *   **<5/10:** The sentence is grammatically awkward, unnatural, or nonsensical. Training on this could harm the model.

2.  **Span Exactness (Weight: 30%):**
    *   **10/10:** ALL subjects, relations, and objects in `extracted_triplets` are **exact, contiguous substrings** of the original `hindi_sentence`. There are no added, removed, or altered words.
    *   **<5/10:** One or more spans are not exact substrings (e.g., they paraphrase, miss prepositions like 'से', or are otherwise altered). This is a critical failure.

3.  **Semantic Correctness of Core Triplets (Weight: 30%):**
    *   **10/10:** The main verb-based triplets accurately capture the primary events and facts of the sentence. The subject-relation-object mapping is logical and correct (e.g., passive voice agents are handled correctly, causal agents are not confused with subjects of state-of-being verbs).
    *   **<5/10:** The core triplets misinterpret the sentence's meaning (e.g., reversing the direction of an action, assigning the wrong subject to a verb). This is a critical failure.

4.  **`property` Relation Quality (Weight: 20%):**
    *   **10/10:** `property` relations are used precisely and consistently for adjectival, possessive, and circumstantial attributes. The extractions are comprehensive without being overly granular or redundant.
    *   **7/10:** `property` relations are generally correct but miss some opportunities or are slightly inconsistent.
    *   **<5/10:** `property` relations are used incorrectly, are overly granular (e.g., breaking down every word), or miss many obvious attributes.

**Output Format:**
For each data point you evaluate, you MUST return a single, valid JSON object with the following structure. Do not add any text before or after the JSON object.

```json
{
  "score": <an integer from 1 to 10>,
  "justification": "A detailed explanation for the score, referencing the specific criteria. Point out any errors in span exactness, semantic correctness, or property usage. If it's a good example, explain why."
}
```

**Few-Shot Examples for Judging (How to Apply the Criteria):**

---
**Example to Judge 1 (This is a GOOD example):**
*   **Sentence:** `कृत्रिम बुद्धिमत्ता को दर्शाता है वह तकनीकी क्षेत्र, जिसका आरम्भ 2010 में राष्ट्रीय विज्ञान संस्थानों द्वारा शुरू हुई।`
*   **Triplets:** `[{"subject": "वह तकनीकी क्षेत्र", "relation": "दर्शाता है", "object": "कृत्रिम बुद्धिमत्ता को"}, {"subject": "आरम्भ", "relation": "शुरू हुई", "object": "राष्ट्रीय विज्ञान संस्थानों द्वारा"}, ...]`
*   **Your Judgement:**
    ```json
    {
      "score": 10,
      "justification": "Excellent. The source sentence is complex and grammatically correct. All spans are exact and contiguous. The core triplets for 'दर्शाता है' and 'शुरू हुई' are semantically correct. The property relations are precise and well-justified. This is a high-quality data point for fine-tuning."
    }
    ```

---
**Example to Judge 2 (This is a BAD example):**
*   **Sentence:** `डॉ. अलीशा बत्रा को जीवविज्ञान एवं रसायन विज्ञान के क्षेत्र में नैनोप्रौद्योगिकी के नवाचार के लिए राष्ट्रीय विज्ञान पुरस्कार (विज्ञान मंत्रालय द्वारा) से सम्मानित किया गया...`
*   **Triplets:** `[{"subject": "डॉ. अलीशा बत्रा", "relation": "सम्मानित किया गया", "object": "जीवविज्ञान एवं रसायन विज्ञान के क्षेत्र में नैनोप्रौद्योगिकी के नवाचार के लिए राष्ट्रीय विज्ञान पुरस्कार (विज्ञान मंत्रालय द्वारा) से"}, ...]`
*   **Your Judgement:**
    ```json
    {
      "score": 4,
      "justification": "Not good enough. The main triplet suffers from a critical 'Overly Broad Object' error. The object for 'सम्मानित किया गया' lumps together the reason, the award, and the giver into one long, imprecise span. This would teach the smaller model a harmful pattern. A high-quality extraction would break this down into multiple, more precise triplets for the award, the reason, and the giver separately. While property relations are okay, the core triplet's flaw is too significant."
    }
    ```

---
**Example to Judge 3 (This is a BAD example):**
*   **Sentence:** `हिमालय व कश्मीर से प्रसिद्ध नदी...भूमिगत जलभण्डार से निरंतर प्रवाहित होता है।`
*   **Triplets:** `[{"subject": "हिमालय व कश्मीर से प्रसिद्ध नदी", "relation": "प्रवाहित होता है", "object": "भूमिगत जलभण्डार"}, ...]`
*   **Your Judgement:**
    ```json
    {
      "score": 2,
      "justification": "Not good enough due to a critical semantic error. The main triplet's object is 'भूमिगत जलभण्डार', but the sentence says the river flows *from* ('से') the reservoir. The object span is not exact as it misses the crucial preposition 'से', reversing the meaning of the action. This is a harmful error that would teach the model to ignore prepositions."
    }
    ```
---

**Your Task:** Now, evaluate the following data point and provide your judgement in the specified JSON format.
"""

def get_judgement(data_point_str):
    """
    Prompts the judge LLM to evaluate a single data point.
    """
    final_prompt = f"{JUDGING_PROMPT_TEMPLATE}\n\n**Data Point to Evaluate:**\n{data_point_str}"

    try:
        response = client.chat.completions.create(
            model=JUDGING_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        judgement_str = response.choices[0].message.content
        return json.loads(judgement_str)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"score": -1, "justification": f"API Error: {e}"}


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        data_points = [json.loads(line) for line in infile]
    
    print(f"[*] Found {len(data_points)} data points in '{INPUT_FILE}'.")
    print(f"[*] Starting evaluation with model '{JUDGING_MODEL}'.")
    print(f"[*] Scored data will be saved to '{OUTPUT_FILE}'.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for data_point in tqdm(data_points, desc="Judging Data Points"):
            # prepare the data point string for the judge
            # we don't need the 'messages' or 'role' wrappers for the judge, just the content
            user_content = next((msg['content'] for msg in data_point['messages'] if msg['role'] == 'user'), '')
            
            # The assistant content is a JSON string, so we need to parse it first
            assistant_content_str = next((msg['content'] for msg in data_point['messages'] if msg['role'] == 'assistant'), '{}')
            try:
                assistant_content_json = json.loads(assistant_content_str)
            except json.JSONDecodeError:
                assistant_content_json = {"thought_process": "INVALID JSON", "extracted_triplets": []}

            # format it nicely for the judge to read
            data_to_judge_str = f"Sentence: `{user_content}`\n\nExtractions:\n```json\n{json.dumps(assistant_content_json, indent=2, ensure_ascii=False)}\n```"
            # print(f"data_to_judge_str: {data_to_judge_str}")

            judgement = get_judgement(data_to_judge_str)
            data_point['judgement'] = judgement
            outfile.write(json.dumps(data_point, ensure_ascii=False) + "\n")

            time.sleep(1) # not required because running locally
    print(f"\n[+] Evaluation complete. Scored data saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    main()
