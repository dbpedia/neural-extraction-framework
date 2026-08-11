# ----------------------------------------------------------------------------------
# This file is the initial version of using the end-2-end framework.
# Modernized for GSoC 2026 standards (Transformers + Outlines v1.2 API)
# ----------------------------------------------------------------------------------
import sys
from pathlib import Path
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix path injection to ensure imports work
try:
    script_path = Path(__file__).resolve()
    PROJECT_ROOT = script_path.parent.parent 
except NameError:
    PROJECT_ROOT = Path().resolve() / "neural-extraction-framework"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Modern Outlines Import
import outlines

# Internal Imports
from GSoC24.Data.collector import get_text_of_wiki_page
from GSoC24.RelationExtraction.methods import get_triples_from_sentence

def main():
    parser = argparse.ArgumentParser(description='An end-2-end utility program')
    parser.add_argument("--sentence", default=None, 
                        help="The sentence on which the user wants to run triple extraction")
    parser.add_argument("--text", default="", 
                        help="The text on which the user wants to run triple extraction")
    parser.add_argument("--wikipage", default=None, 
                        help="The title of wikipedia page on which to perform relation extraction")
    parser.add_argument("--save_filename", default=None, 
                        help="The file name of the csv of triples. If specified, the file will be saved.")
    parser.add_argument("--v", default=0, help="If set to 1, print the triples dataframe")
    parser.add_argument("--text_filepath", default="", 
                        help="The text file on which the user wants to run triple extraction")
    
    args = parser.parse_args()

    # --- 1. Load Model (Modernized) ---
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_name}...")
    
    # 1. Detect Device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32  # CPU doesn't support float16 well
        
    print(f"   Using device: {device.upper()}")

    try:
        # 2. Load via Transformers
        # We explicitly handle device_map to prevent CPU crashes
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=dtype,
            device_map=device  # Auto-dispatch to correct device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 3. Wrap with Outlines
        model = outlines.from_transformers(hf_model, tokenizer)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Determine Input Source ---
    sentences = None
    if args.sentence:
        sentences = args.sentence
    elif args.text:
        sentences = args.text
    elif args.wikipage:
        print(f"Fetching Wikipedia page: {args.wikipage}...")
        sentences = get_text_of_wiki_page(args.wikipage)
    elif args.text_filepath:
        with open(args.text_filepath, "r") as f:
            print(f"Reading text from file: {args.text_filepath}")
            sentences = f.read()

    if not sentences:
        print("No input text provided. Use --sentence, --text, --wikipage, or --text_filepath")
        return

    # --- 3. Extract Triples ---
    print("🚀 Extracting triples (this may take a moment)...")
    try:
        # Pass the modernized model to the extraction method
        sentence_triples = get_triples_from_sentence(user_prompt=sentences, model=model)
        
        # --- 4. Process Results ---
        triples = [trip for trip in sentence_triples]
        triples_dataframe = pd.DataFrame(data=triples)
        
        print("Extraction Complete.")

        if args.save_filename:
            triples_dataframe.to_csv(args.save_filename, index=False)
            print(f"Triples saved to: {args.save_filename}")

        if int(args.v) == 1:
            print("\n--- Extracted Triples ---")
            print(triples_dataframe)
            print("-------------------------\n")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()