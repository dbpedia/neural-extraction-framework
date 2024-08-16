import requests
import json
import time


def get_prompt(text):
    template = {
        "triplets": [
            {
                "subject": "Hindi Subject",
                "predicate": "Hindi Predicate",
                "object": "Hindi Object",
            }
        ],
    }
    max_triplets = 3
    example_output = {
        "triplets": [
            {
                "subject": "एबिलीन टेक्सास",
                "predicate": "देश",
                "object": "संयुक्त राज्य अमेरिक",
            }
        ]
    }
    """
    Some examples are also given below:
    Text: एबिलीन, टेक्सास संयुक्त राज्य अमेरिका में है।
    {json.dumps(example_output, ensure_ascii= False)}
    """

    prompt = f"""
Generate Hindi knowledge triplets from the following text. Each triplet should be in the format (subject, predicate, object). Extract up to {max_triplets} triplets and present them in JSON format. The output should be in hindi and should look like this:
{json.dumps(template, ensure_ascii=False)}

Here's the text to analyze:
{text}"""
    return prompt


def get_llm_triplets(user_text):
    model = "mistral:latest"
    prompt = get_prompt(user_text)
    data = {
        "prompt": prompt,
        "model": model,
        "format": "json",
        "stream": False,
        "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
    }

    start_time = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate", json=data, stream=False
    )
    ttaken = round(time.time() - start_time, 3)
    json_data = json.loads(response.text)
    result = json.loads(json_data["response"])
    return result["triplets"], ttaken


if __name__ == "__main__":
    user_text = "एबिलीन, टेक्सास संयुक्त राज्य अमेरिका में है।"
    print(get_llm_triplets(user_text))
