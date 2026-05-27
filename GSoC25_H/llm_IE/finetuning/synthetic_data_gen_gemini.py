# Dependencies
# pip install google-generativeai tqdm

import google.generativeai as genai
import os
import argparse
import time
import json
from tqdm import tqdm
import shutil

# API_KEY = os.environ.get("GEMINI_API_KEY")
# if not API_KEY:
#     raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")

genai.configure(api_key="")

PROMPT_TEMPLATE = """
You are an expert in Hindi linguistics and information extraction, following the ReAct (Reason, Act) methodology. Your task is to generate a high-quality synthetic data point for training a relation extraction model.

**Your Goal:** For the given topic, generate one diverse and informative Hindi sentence, then extract its relation triples into a structured JSON format.

**Methodology: ReAct (Reason -> Act)**

**1. Reason (Think step-by-step):**
   - First, analyze the topic: **{topic}**.
   - Brainstorm potential scenarios or facts. Think about entities, actions, properties, and nested information. Aim for complexity.
   - Formulate a natural-sounding Hindi sentence.
   - Plan the extraction. Identify main triples and any property-of relations. Group into clusters if necessary.

**2. Act (Generate JSON):**
   - After reasoning, generate a single, valid JSON object.
   - **Important:** Do not output your reasoning process or any other text. Output ONLY the final JSON object.

**JSON Schema and Instructions:**

The JSON object must have two keys: `thought` and `extraction`.

- `thought`: A string containing your reasoning process as described above.
- `extraction`: A JSON object with the following structure:
  - `sentence`: (string) The generated Hindi sentence.
  - `clusters`: (array of objects) A list of extraction clusters. Usually just one, but more if there are distinct interpretations.
    - Each cluster object has:
      - `comment`: (string) A brief explanation of what this cluster represents.
      - `extractions`: (array of objects) A list of all extractions for this cluster.
        - Each extraction object has:
          - `subject`: (string) The subject of the triple.
          - `relation`: (string) The relation of the triple.
          - `object`: (string) The object of the triple.
          - `id`: (optional string) A unique letter (e.g., 'a', 'b', 'c') if this extraction is a property that will be referenced by another extraction.
  - **Formatting Rules for `subject` and `object` strings:**
    - To mark a part of an argument as optional, enclose it in square brackets. Example: `प्राग [को]`
    - To link to a property extraction, use curly braces with the `id` of the property extraction. Example: `[चेकोस्लोवाकिया की राजधानी]{{a}} प्राग को` references the property extraction with `id: 'a'`.

**Example of the Final JSON Output:**

```json
{{
  "thought": "The topic is 'history'. I will create a sentence about the Soviet forces liberating Prague after capturing Berlin. This involves multiple actions and locations. Main extraction: Soviets liberated Prague. A property of the Soviets is that they captured Berlin first. A property of Prague is that it's the capital of Czechoslovakia. This nesting requires multiple property extractions. I will create one cluster for this primary interpretation.",
  "extraction": {{
    "sentence": "बर्लिन पर क़ब्ज़ा करने के बाद सोवियत दस्तों ने चेकोस्लोवाकिया की राजधानी प्राग को आज़ाद कराया .",
    "clusters": [
      {{
        "comment": "Primary extraction focusing on the liberation of Prague, with related properties.",
        "extractions": [
          {{
            "subject": "सोवियत दस्तों ने",
            "relation": "आज़ाद कराया",
            "object": "[चेकोस्लोवाकिया की राजधानी]{{a}} प्राग को"
          }},
          {{
            "subject": "सोवियत दस्तों ने",
            "relation": "property",
            "object": "बर्लिन पर क़ब्ज़ा [करने के बाद]"
          }},
          {{
            "id": "a",
            "subject": "[चेकोस्लोवाकिया की]{{b}} राजधानी",
            "relation": "property",
            "object": "प्राग [को]"
          }},
          {{
            "id": "b",
            "subject": "राजधानी",
            "relation": "property",
            "object": "चेकोस्लोवाकिया की"
          }}
        ]
      }}
    ]
  }}
}}
```

Now, for the topic **"{topic}"**, generate one such JSON object.
"""

def generate_synthetic_data(model, topic):
    """
    Calls the Gemini API to generate a single synthetic data point as a JSON object.
    Returns the parsed JSON object or None if an error occurs.
    """
    prompt = PROMPT_TEMPLATE.format(topic=topic)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.8,
            )
        )
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON from API response: {e}")
        print(f"Received text: {response.text}")
        return None
    except Exception as e:
        print(f"An API error occurred: {e}")
        time.sleep(60)
        return None

def main():
    """Main function to run the data generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate synthetic Hindi BenchIE data as JSONL using Gemini and ReAct.")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=8000,
        help="Total number of examples to generate."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash",
        help="Name of the Gemini model to use."
    )
    args = parser.parse_args()

    output_dir = "synthetic_data"
    os.makedirs(output_dir, exist_ok=True)

    topics = [
        'इतिहास', 'विज्ञान', 'बॉलीवुड', 'खेल', 'प्रौद्योगिकी', 'राजनीति', 
        'साहित्य', 'यात्रा', 'स्वास्थ्य', 'अर्थव्यवस्था', 'कला', 'शिक्षा', 
        'पर्यावरण', 'समाज', 'अंतरिक्ष', 'दर्शन', 'भूगोल', 'संगीत'
    ]
    
    model = genai.GenerativeModel(args.model_name)
    
    start_count = 0
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    # check if the output directory exists and resume from the last example
    if json_files:
        try:
            # find the highest number from filenames like '1.json', '2.json'
            # start from last number instead of count of files
            max_num = max([int(f.split('.')[0]) for f in json_files if f.split('.')[0].isdigit()])
            start_count = max_num
        except ValueError:
            print("Warning: Could not determine start count from filenames. Starting from 0.")
            start_count = 0

    print(f"Starting generation, saving individual files to '{output_dir}/'")
    print(f"Resuming from example number {start_count + 1}.")
    print(f"Target: {args.num_examples} examples.")
    
    pbar = tqdm(total=args.num_examples, initial=start_count)
    
    current_examples_count = start_count
    topic_index = 0
    
    while current_examples_count < args.num_examples:
        topic = topics[(start_count + topic_index) % len(topics)]
        
        generated_json = generate_synthetic_data(model, topic)
        
        if generated_json and 'extraction' in generated_json:
            current_examples_count += 1
            file_path = os.path.join(output_dir, f"{current_examples_count}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(generated_json, f, ensure_ascii=False, indent=4)
            pbar.update(1)
        else:
            print(f"Skipping a failed generation for topic '{topic}'. Will retry.")

        topic_index += 1
        # delay for rate limits 
        time.sleep(2)
            
    pbar.close()
    print("\nGeneration complete. Zipping the output directory...")
    try:
        # zip the output directory before exiting
        shutil.make_archive('synthetic_data', 'zip', output_dir)
        print("Successfully created synthetic_data.zip.")
    except Exception as e:
        print(f"An error occurred during zipping or cleanup: {e}")

if __name__ == "__main__":
    main() 