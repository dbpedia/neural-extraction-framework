import ollama
import json
import time
from typing import List, Dict, Any

from config import ModelConfig

class LLMInterface:
    """Interface for interacting with the language model via Ollama."""
    def __init__(self, model_config: ModelConfig, max_retries: int = 1, timeout: int = 60):
        self.model_config = model_config
        self.max_retries = max_retries
        self.client = ollama.Client(timeout=timeout)

    def generate_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a response from the LLM, with retries for handling errors.
        Now sends a standard chat request without the 'tools' parameter.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                response = self.client.chat(
                    model=self.model_config.name,
                    messages=messages,
                    options={
                        "temperature": self.model_config.temperature,
                        "top_p": self.model_config.top_p,
                    }
                )
                return response

            except Exception as e:
                retries += 1
                print(f"Error calling model '{self.model_config.name}': {e}. Retrying ({retries}/{self.max_retries})...")
                time.sleep(2 ** retries)
        
        print(f"Failed to get a valid response from model '{self.model_config.name}' after {self.max_retries} retries.")
        return None

