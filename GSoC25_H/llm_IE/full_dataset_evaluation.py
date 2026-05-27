#!/usr/bin/env python3
"""
Full dataset evaluation script - runs on all 112 sentences
"""

import os
import time
import traceback
from typing import Dict

from llm_interface import OllamaInterface
from config import config
from prompt_templates import PromptTemplateManager

class BenchieDataLoader:
    """Load and parse Hindi-Benchie golden standard data"""
    
    def __init__(self, golden_standard_path: str):
        self.golden_standard_path = golden_standard_path
        self.sentences = {}
        self.load_data()
    
    def load_data(self):
        """Load sentences from golden standard file"""
        with open(self.golden_standard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the benchie format
        sections = content.split('==================================================================================================================================================================================')
        
        for section in sections:
            if 'sent_id:' in section:
                lines = section.strip().split('\n')
                if len(lines) > 0:
                    # Extract sentence ID and text
                    first_line = lines[0]
                    if '\t' in first_line:
                        sent_info, sentence = first_line.split('\t', 1)
                        sent_id = sent_info.replace('sent_id:', '').strip()
                        self.sentences[sent_id] = sentence.strip()
    
    def get_all_sentences(self) -> Dict[str, str]:
        """Get all sentences"""
        return self.sentences

def run_full_dataset_evaluation():
    """Run evaluation on complete Hindi-Benchie dataset"""
    
    print("Starting evaluation on complete Hindi-Benchie dataset...")
    
    models_to_run = config.experiment.models
    strategies_to_run = config.experiment.prompt_strategies
    print(f"Models to run: {models_to_run}")
    print(f"Strategies to run: {strategies_to_run}")
    
    golden_standard_path = config.evaluation["benchie_gold_file"]
    output_dir = "full_dataset_results"

    print("Loading complete Hindi-Benchie dataset...")
    data_loader = BenchieDataLoader(golden_standard_path)
    sentences = data_loader.get_all_sentences()
    # sentences = dict(list(data_loader.get_all_sentences().items())[:10]) # for testing
    print(f"Loaded {len(sentences)} sentences for evaluation.")
    
    template_manager = PromptTemplateManager()

    # evaluate each model on each strategy
    for model_key in models_to_run:
        print(f"\nEvaluating model: {model_key}")
        print("-" * 50)
        
        model_config = config.get_model_config(model_key)
        try:
            model_interface = OllamaInterface(model_config)
        except ConnectionError as e:
            print(f"Could not initialize model {model_key}: {e}")
            continue
        
        for strategy in strategies_to_run:
            print(f"  Testing strategy: {strategy}")
            print(f"     Processing {len(sentences)} sentences...")
            
            # Extract relations for all sentences
            extractions = []
            successful_extractions = 0
            total_time = 0
            failed_sentences = []
            
            start_time = time.time()
            
            for i, (sent_id, sentence) in enumerate(sentences.items(), 1):
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i if i > 0 else 0
                    remaining = (len(sentences) - i) * avg_time
                    print(f"     Progress: {i}/{len(sentences)} ({i/len(sentences)*100:.1f}%) - ETA: {remaining/60:.1f} min")
                
                prompt = template_manager.generate_prompt(strategy, sentence)
                result = model_interface.extract_relations(sentence, prompt)
                total_time += result.processing_time
                
                if result.success:
                    successful_extractions += 1
                    for triplet in result.parsed_triplets:
                        extractions.append((sent_id, triplet.get('subject', ''), triplet.get('relation', ''), triplet.get('object', '')))
                else:
                    failed_sentences.append((sent_id, result.error))
            
            print(f"     Extraction completed in {total_time/60:.1f} minutes")
            print(f"     Success: {successful_extractions}/{len(sentences)} sentences")
            print(f"     Total extractions: {len(extractions)}")

            # Save the extractions to a file for the detailed analyzer to use
            extraction_filename = f"extractions_{model_interface.model_config.name.replace(':', '_')}_{strategy}.txt"
            extraction_filepath = os.path.join(output_dir, extraction_filename)
            with open(extraction_filepath, 'w', encoding='utf-8') as f:
                for sent_id, subject, relation, obj in extractions:
                    f.write(f"{sent_id}\t{subject}\t{relation}\t{obj}\n")
            
            print(f"     Extractions saved to: {extraction_filepath}")

    print("\nExtraction generation complete.")

if __name__ == "__main__":
    try:
        print("Starting extraction generation for the full dataset...")
        run_full_dataset_evaluation()
        print("\nExtraction file generation completed.")
        
    except Exception as e:
        print(f"\nAn error occurred during extraction: {e}")
        traceback.print_exc() 