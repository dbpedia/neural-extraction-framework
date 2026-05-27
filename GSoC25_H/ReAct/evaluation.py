import os
import json
import time
from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import config
from data_loader import BenchieDataLoader
from prompt_factory import PromptFactory
from llm_interface import LLMInterface 

class Triplet(BaseModel):
    """Represents a single Subject-Relation-Object triplet."""
    subject: str = Field(..., description="The identified subject")
    relation: str = Field(..., description="The identified relation")
    object: str = Field(..., description="The identified object")

class RelationParameters(BaseModel):
    """Parameters for the extract_relation function, accepting a list of triplets."""
    triplets: List[Triplet] = Field(..., description="A list of extracted (subject, relation, object) triplets.")

class FunctionCall(BaseModel):
    """A model to represent a function call from the LLM."""
    name: str
    parameters: RelationParameters
def extract_relation(triplets: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Processes the extracted relation triplets.
    """     
    return {"status": "success", "data": triplets}

def parse_function_call(model_response: str) -> Optional[FunctionCall]:
    """
    Parses the model's response to find and validate a JSON function call.
    This is improved to find a JSON block even if it's mixed with other text.
    """
    # Use regex to find a JSON object block in the response
    json_match = re.search(r'{\s*"name":\s*"extract_relation"[\s\S]*}', model_response)
    if not json_match:
        return None
        
    json_str = json_match.group(0)
    
    try:
        data = json.loads(json_str)
        return FunctionCall(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # This can help in debugging if the JSON is malformed
        print(f"JSON parsing/validation failed: {e}")
        return None


def run_evaluation():
    """Main function to run the ReAct evaluation framework."""
    
    # 1. Load Configuration and Initialize Components
    cfg = config
    cfg.validate()
    
    data_loader = BenchieDataLoader(cfg.evaluation_data_path)
    prompt_factory = PromptFactory(cfg)

    if len(data_loader) == 0:
        print("No sentences to process. Exiting.")
        return

    # 2. Iterate Through All Combinations of Models and Strategies
    for model_key in cfg.experiment.models:
        model_config = cfg.get_model_config(model_key)
        llm_interface = LLMInterface(model_config, max_retries=cfg.experiment.max_retries)
        
        for strategy_key in cfg.experiment.prompt_strategies:
            strategy_config = cfg.get_prompt_strategy(strategy_key)
            
            print("="*80)
            print(f"Running evaluation for Model: '{model_config.name}' with Strategy: '{strategy_config.name}'")
            print("="*80)

            all_results = []
            raw_extractions = []
            total_time = 0
            
            sentences = data_loader.get_all_sentences()
            sorted_sent_ids = sorted(sentences.keys(), key=lambda x: int(x))
            triplets = []
            # 3. Process Each Sentence
            for i, sent_id in enumerate(sorted_sent_ids):
                sentence = sentences[sent_id]
                print(f"  Processing sentence {i+1}/{len(sorted_sent_ids)} (ID: {sent_id})...")

                start_time = time.time()
                
                # a. Create prompt
                messages = prompt_factory.create_prompt(strategy_key, sentence)
                
                # b. Get model response
                response = llm_interface.generate_response(messages)

                if not response:
                    print(f"    -> Skipping sentence {sent_id} due to model error") 
                    # timeout etc can occur
                    # still store a result so we know it was attempted
                    result = {
                        "sent_id": sent_id,
                        "sentence": sentence,
                        "model": model_key,
                        "strategy": strategy_key,
                        "llm_response": "",
                        "extracted_triplets": [],
                        "time_taken": (time.time() - start_time)
                    }
                    all_results.append(result)
                    continue

                
                model_response = response['message']['content']
                function_call = parse_function_call(model_response)
                
                if function_call and function_call.name == "extract_relation":
                    try:
                        params = function_call.parameters
                        result = extract_relation(triplets=[triplet.dict() for triplet in params.triplets])
                        triplets = result.get('data', [])
                    except Exception as e:
                        print(f"Error executing function call: {e}")
                        triplets = [] # Ensure triplets is a list
                else:
                    print("Model did not return a valid function call.")
                    triplets = [] # Ensure triplets is a list
                
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                
                # d. Store results
                result = {
                    "sent_id": sent_id,
                    "sentence": sentence,
                    "model": model_key,
                    "strategy": strategy_key,
                    "llm_response": dict(response['message']),
                    "extracted_triplets": triplets,
                    "time_taken": duration
                }
                all_results.append(result)

                if triplets:
                    print(f"    -> Extracted {len(triplets)} triplet(s) in {duration:.2f}s.")
                    for t in triplets:
                        raw_extractions.append(f"{sent_id}\t{t['subject']}\t{t['relation']}\t{t['object']}")
                else:
                    print(f"    -> No triplets extracted. ({duration:.2f}s)")

            # 4. Save Results for this Combination
            avg_time = total_time / len(sentences) if sentences else 0
            print(f"Finished. Total time: {total_time:.2f}s, Avg time/sentence: {avg_time:.2f}s")

            output_dir = os.path.join(os.path.dirname(__file__), cfg.experiment.output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            model_name_safe = model_key.replace(":", "_").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # save  results
            json_filename = f"detailed_analysis_{model_name_safe}_{strategy_key}_{timestamp}.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            # print("\n--- Preparing to save JSON. Checking data types... ---")
            # for i, res in enumerate(all_results):
            #     llm_resp_type = type(res.get("llm_response"))
            #     if llm_resp_type is not dict:
            #         print(f"  [Entry {i+1}] llm_response is of type: {llm_resp_type} <-- POTENTIAL ISSUE")
            #     else:
            #         print(f"  [Entry {i+1}] llm_response is of type: {llm_resp_type}")
            # print("---------------------------------------------------------\n")

            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Detailed results saved to: {json_filepath}")
            
            # Save raw extractions for compatibility with benchie
            extraction_filename = f"extractions_{model_name_safe}_{strategy_key}.txt"
            extraction_filepath = os.path.join(output_dir, extraction_filename)
            with open(extraction_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(raw_extractions))
            print(f"Raw extractions saved to: {extraction_filepath}")
            print("-" * 80)

if __name__ == "__main__":
    run_evaluation() 