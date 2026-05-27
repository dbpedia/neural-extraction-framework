import os
from typing import Dict

class BenchieDataLoader:
    """Loads sentences from the Hindi-Benchie golden standard file."""
    def __init__(self, golden_standard_path: str):
        self.golden_standard_path = golden_standard_path
        self.sentences: Dict[str, str] = {}
        if os.path.exists(self.golden_standard_path):
            self.load_data()
        else:
            print(f"Warning: Golden standard file not found at {self.golden_standard_path}")

    def load_data(self):
        """Reads the golden standard file and parses sentences."""
        try:
            with open(self.golden_standard_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # The separator is a long line of '=' characters
            sections = content.split('==================================================================================================================================================================================')
            
            for section in sections:
                if 'sent_id:' in section:
                    lines = section.strip().split('\n')
                    if lines and '\t' in lines[0]:
                        sent_info, sentence = lines[0].split('\t', 1)
                        sent_id_str = sent_info.replace('sent_id:', '').strip()
                        if sent_id_str:
                            self.sentences[sent_id_str] = sentence.strip()
        except Exception as e:
            print(f"Error loading or parsing golden standard file: {e}")

    def get_all_sentences(self) -> Dict[str, str]:
        """Returns all loaded sentences as a dictionary."""
        return self.sentences