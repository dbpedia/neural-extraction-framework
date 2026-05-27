import pandas as pd
import sys
import os

def convert_hdf5_to_tab_format(hdf5_file='benchie_indie_new.h5', lang='hi', output_file='benchie_indie_converted_llm.txt'):
    """
    Convert HDF5 extractions to tab-separated format expected by code.py
    
    Args:
        hdf5_file: Path to the HDF5 file containing extractions
        lang: Language key for the HDF5 data
        output_file: Output file path for tab-separated data
    """
    
    print(f"Reading HDF5 file: {hdf5_file}")
    
    try:
        # Read the HDF5 file
        df = pd.read_hdf(hdf5_file, key=lang)
        print(f"Loaded {len(df)} sentences")
        
        # Open output file for writing
        with open(output_file, 'w', encoding='utf-8') as f:
            
            # Process each row
            for idx, row in df.iterrows():
                sentence_number = str(idx + 1)  # 1-indexed sentence numbers
                
                # Process regular extractions
                if isinstance(row['extractions'], list) and len(row['extractions']) > 0:
                    # extractions is [[extraction1, extraction2, ...]]
                    for extraction_list in row['extractions']:
                        if isinstance(extraction_list, list):
                            for ext in extraction_list:
                                if isinstance(ext, (list, tuple)) and len(ext) >= 3:
                                    # Each extraction is [arg1, relation, arg2]
                                    arg1 = str(ext[0]).strip()
                                    relation = str(ext[1]).strip()
                                    arg2 = str(ext[2]).strip()
                                    # Format: sentence_number \t arg1 \t relation \t arg2
                                    line = f"{sentence_number}\t{arg1}\t{relation}\t{arg2}"
                                    f.write(line + '\n')
                
                # Process augmented extractions  
                if isinstance(row['augmented_exts'], list) and len(row['augmented_exts']) > 0:
                    # augmented_exts is [[extraction1, extraction2, ...]]
                    for extraction_list in row['augmented_exts']:
                        if isinstance(extraction_list, list):
                            for ext in extraction_list:
                                if isinstance(ext, (list, tuple)) and len(ext) >= 3:
                                    # Each extraction is [arg1, relation, arg2]
                                    arg1 = str(ext[0]).strip()
                                    relation = str(ext[1]).strip()
                                    arg2 = str(ext[2]).strip()
                                    # Format: sentence_number \t arg1 \t relation \t arg2
                                    line = f"{sentence_number}\t{arg1}\t{relation}\t{arg2}"
                                    f.write(line + '\n')
        
        print(f"Conversion complete! Output written to: {output_file}")
        print(f"You can now run code.py with this file")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False
    
    return True

def main():
    # Default parameters
    hdf5_file = 'benchie_indie_new.h5'
    lang = 'hi'
    output_file = 'benchie_indie_converted_gemma3_12b_rule_react_updated_mdt_info.txt'
    
    # Check if HDF5 file exists
    if not os.path.exists(hdf5_file):
        print(f"Error: HDF5 file '{hdf5_file}' not found!")
        return
    
    # Convert the file
    success = convert_hdf5_to_tab_format(hdf5_file, lang, output_file)
    
    if success:
        print("\nNext steps:")
        print(f"1. Check the output file: {output_file}")
        print("2. Modify code.py to read from this file instead of the original extractions")
        print("3. Run code.py to get your statistics")
    else:
        print("Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 