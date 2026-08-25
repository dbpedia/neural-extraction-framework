import time
import logging
import ollama
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """Unified Configuration for the LLM model."""
    name: str = "gemma3:12b-it-qat"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 2000
    timeout: int = 60
    max_retries: int = 3

class LLMService:
    """
    Centralized service for LLM interactions.
    Replaces duplicative logic in IndIE/llm_extractor.py and llm_IE/llm_interface.py
    """
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.client = ollama.Client(host=self.config.host, timeout=self.config.timeout)
        self.logger = logging.getLogger(self.__class__.__name__)
        self._ensure_model_available()

    def _ensure_model_available(self):
        """Checks if model exists, attempts to pull if missing (logic from llm_IE)."""
        try:
            # List available models to check if our model exists
            available_models_response = self.client.list()
            
            # Handle different possible response structures
            if hasattr(available_models_response, 'models'):
                models_list = available_models_response.models
            elif isinstance(available_models_response, dict):
                models_list = available_models_response.get('models', [])
            elif isinstance(available_models_response, list):
                models_list = available_models_response
            else:
                self.logger.warning("Could not parse model list response. Attempting direct model check...")
                try:
                    self.client.chat(
                        model=self.config.name,
                        messages=[{'role': 'user', 'content': 'test'}],
                        options={'num_predict': 1}
                    )
                    self.logger.debug(f"Model '{self.config.name}' is available (verified via chat)")
                    return
                except Exception as e:
                    # FIX: Log the failure reason for debugging purposes
                    self.logger.debug(f"Fallback model check failed: {e}")
                    models_list = []
            
            # Extract model names safely
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    model_names.append(model.get('name', ''))
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                elif isinstance(model, str):
                    model_names.append(model)
            
            # Smart matching for tags (e.g. 'gemma3' vs 'gemma3:latest')
            model_available = any(
                self.config.name == name or 
                name.startswith(self.config.name + ':') or
                self.config.name.startswith(name + ':')
                for name in model_names
            )

            if not model_available:
                self.logger.info(f"Model '{self.config.name}' not found locally. Attempting to pull...")
                self.client.pull(self.config.name)
                self.logger.info(f"Successfully pulled model '{self.config.name}'")
            else:
                self.logger.debug(f"Model '{self.config.name}' is already available")
                
        except Exception as e:
            self.logger.error(f"Model availability check/pull failed for '{self.config.name}': {e}", exc_info=True)

    def generate_response(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generates a response with standardized retry logic (logic from IndIE).
        """
        retries = 0
        while retries < self.config.max_retries:
            try:
                response = self.client.chat(
                    model=self.config.name,
                    messages=messages,
                    options={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "num_predict": self.config.num_predict
                    }
                )
                return response

            except Exception as e:
                retries += 1
                if retries >= self.config.max_retries:
                    self.logger.error(
                        f"Error calling model '{self.config.name}': {e}. "
                        f"Exhausted {self.config.max_retries} retries.",
                        exc_info=True,
                    )
                    break
                
                wait_time = 2 ** retries
                self.logger.warning(
                    f"Error calling model '{self.config.name}': {e}. "
                    f"Retrying ({retries}/{self.config.max_retries}) in {wait_time}s..."
                )
                time.sleep(wait_time)
        
        return None