import os
import requests
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from config import ModelConfig
from output_parser import OutputParser

@dataclass
class ExtractionResult:
    """Result from LLM extraction"""
    success: bool
    raw_output: str
    parsed_triplets: List[Dict[str, str]] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

class OllamaInterface:
    """Unified interface for interacting with Ollama models"""

    def __init__(self, model_config: ModelConfig, base_url: str = "http://localhost:11434"):
        self.model_config = model_config
        self.base_url = base_url.rstrip('/')
        self.api_endpoint = f"{self.base_url}/api"
        self.output_parser = OutputParser()
        
        if not self._is_available():
            print(f"Warning: Ollama model '{self.model_config.name}' not found locally. Trying to pull it...")
            if not self._pull_model():
                raise ConnectionError(f"Failed to pull or connect to Ollama model {self.model_config.name}")

    def _is_available(self) -> bool:
        """Check if the Ollama model is available locally"""
        try:
            response = requests.get(f"{self.api_endpoint}/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(m['name'] == self.model_config.name for m in models)
        except requests.exceptions.RequestException:
            return False

    def _generate_text(self, prompt: str) -> str:
        """Generic text generation using the configured Ollama model."""
        start_time = time.time()
        
        payload = {
            "model": self.model_config.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "top_k": self.model_config.top_k,
                "num_predict": self.model_config.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_endpoint}/generate",
                json=payload,
                timeout=self.model_config.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Error during Ollama API request: {e}")
            return ""

    def extract_relations(self, sentence: str, prompt: str) -> ExtractionResult:
        """Extracts relations from a sentence using a given prompt."""
        start_time = time.time()
        
        raw_output = self._generate_text(prompt)
        processing_time = time.time() - start_time
        
        if not raw_output:
            return ExtractionResult(
                success=False,
                raw_output="",
                processing_time=processing_time,
                error="Failed to generate text from model."
            )
        
        parsed_triplets, _ = self.output_parser.parse_and_format(raw_output)
        
        return ExtractionResult(
            success=len(parsed_triplets) > 0,
            raw_output=raw_output,
            parsed_triplets=parsed_triplets,
            processing_time=processing_time
        )

    def _pull_model(self) -> bool:
        """Pull the model from the Ollama registry."""
        print(f"Pulling model: {self.model_config.name}. This may take a while...")
        try:
            response = requests.post(
                f"{self.api_endpoint}/pull",
                json={"name": self.model_config.name, "stream": False},
                timeout=300  # 5-minute timeout for pulling
            )
            response.raise_for_status()
            print(f"Model '{self.model_config.name}' pulled successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to pull model '{self.model_config.name}': {e}")
            return False 