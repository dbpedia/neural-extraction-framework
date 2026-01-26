import time
import logging
import ollama
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """Unified Configuration for the LLM model."""
    name: str = "gemma3:12b-it-qat"
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
        self.client = ollama.Client(timeout=self.config.timeout)
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
                # Fallback: try a lightweight chat call to verify model availability
                self.logger.warning(f"Could not parse model list response. Attempting direct model check...")
                try:
                    self.client.chat(
                        model=self.config.name,
                        messages=[{'role': 'user', 'content': 'test'}],
                        options={'num_predict': 1}
                    )
                    self.logger.debug(f"Model '{self.config.name}' is available (verified via chat)")
                    return
                except Exception:
                    # Model not available, will attempt to pull
                    models_list = []
            
            # Extract model names from the list
            model_names = []
            for model in models_list:
                if isinstance(model, dict):
                    model_names.append(model.get('name', ''))
                elif hasattr(model, 'name'):
                    model_names.append(model.name)
                elif isinstance(model, str):
                    model_names.append(model)
            
            if self.config.name not in model_names:
                self.logger.info(f"Model '{self.config.name}' not found locally. Attempting to pull...")
                # Pull the model
                self.client.pull(self.config.name)
                
                # Verify the model was pulled successfully by listing again
                available_models_after = self.client.list()
                
                # Parse the response again
                if hasattr(available_models_after, 'models'):
                    models_list_after = available_models_after.models
                elif isinstance(available_models_after, dict):
                    models_list_after = available_models_after.get('models', [])
                elif isinstance(available_models_after, list):
                    models_list_after = available_models_after
                else:
                    models_list_after = []
                
                model_names_after = []
                for model in models_list_after:
                    if isinstance(model, dict):
                        model_names_after.append(model.get('name', ''))
                    elif hasattr(model, 'name'):
                        model_names_after.append(model.name)
                    elif isinstance(model, str):
                        model_names_after.append(model)
                
                if self.config.name not in model_names_after:
                    # Final verification: try a lightweight chat call
                    try:
                        self.client.chat(
                            model=self.config.name,
                            messages=[{'role': 'user', 'content': 'test'}],
                            options={'num_predict': 1}
                        )
                        self.logger.info(f"Successfully pulled and verified model '{self.config.name}'")
                    except Exception as verify_error:
                        raise RuntimeError(
                            f"Failed to pull model '{self.config.name}'. "
                            f"Model not available after pull attempt: {verify_error}"
                        )
                else:
                    self.logger.info(f"Successfully pulled model '{self.config.name}'")
            else:
                self.logger.debug(f"Model '{self.config.name}' is already available")
                
        except Exception as e:
            self.logger.error(f"Model availability check/pull failed for '{self.config.name}': {e}", exc_info=True)
            raise

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
                wait_time = 2 ** retries
                print(f"Error calling model '{self.config.name}': {e}. Retrying ({retries}/{self.config.max_retries}) in {wait_time}s...")
                time.sleep(wait_time)
        
        print(f"Failed to get response from '{self.config.name}' after {self.config.max_retries} retries.")
        return None