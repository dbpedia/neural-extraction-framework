
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.benchie_utils import parse_benchie_sentences


class BenchieDataLoader:
    """Loads sentences from the Hindi-Benchie golden standard file."""

    def __init__(self, golden_standard_path: str):
        self.golden_standard_path = golden_standard_path
        self.sentences: Dict[str, str] = {}

        if os.path.exists(self.golden_standard_path):
            self.load_data()
        else:
            print(
                f"Warning: Golden standard file not found at "
                f"{self.golden_standard_path}"
            )

    def load_data(self):
        """Reads the golden standard file and parses sentences."""
        try:
            with open(self.golden_standard_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.sentences.update(parse_benchie_sentences(content))

        except Exception as e:
            print(f"Error loading or parsing golden standard file: {e}")

    def get_all_sentences(self) -> Dict[str, str]:
        """Returns all loaded sentences as a dictionary."""
        return self.sentences

    def __len__(self) -> int:
        """Return the number of loaded sentences."""
        return len(self.sentences)