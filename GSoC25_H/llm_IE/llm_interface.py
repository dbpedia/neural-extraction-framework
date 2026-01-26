import sys
import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Add parent directory to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm_core import LLMService, ModelConfig as SharedConfig
from output_parser import OutputParser
from config import ModelConfig

@dataclass
class ExtractionResult:
    """Result from LLM extraction"""
    success: bool
    raw_output: str
    parsed_triplets: List[Dict[str, str]] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None

class OllamaInterface:
    """
    Unified interface for interacting with Ollama models.
    Refactored to use the shared src.llm_core.LLMService instead of raw requests.
    """

    def __init__(self, model_config: ModelConfig, base_url: str = "http://localhost:11434"):
        self.output_parser = OutputParser()
        
        # ADAPTER: Convert local llm_IE config to the Shared Config
        # We map 'max_tokens' (from llm_IE) to 'num_predict' (shared core)
        shared_config = SharedConfig(
            name=model_config.name,
            host=base_url,  # <--- Fix: Pass the base_url here!
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            num_predict=getattr(model_config, 'num_predict', 2000),
            timeout=getattr(model_config, 'timeout', 60),
            max_retries=getattr(model_config, 'max_retries', 3)
        )
        
        # Initialize the shared service
        self.service = LLMService(shared_config)

    def extract_relations(self, sentence: str, prompt: str) -> ExtractionResult:
        """Extracts relations from a sentence using the shared LLM service."""
        # Fix: Explicitly mark 'sentence' as unused to satisfy linter
        _ = sentence 
        
        start_time = time.time()
        
        # Prepare standard message format for the shared service
        messages = [{"role": "user", "content": prompt}]
        
        # Use shared service (Handles retries and connection automatically)
        response = self.service.generate_response(messages)
        processing_time = time.time() - start_time
        
        if not response:
            return ExtractionResult(
                success=False,
                raw_output="",
                processing_time=processing_time,
                error="Failed to generate text from model."
            )
        
        # Extract text content from the Ollama response dictionary
        raw_output = response.get("message", {}).get("content", "").strip()
        
        # Parse output using existing parser
        parsed_triplets, _ = self.output_parser.parse_and_format(raw_output)
        
        return ExtractionResult(
            success=len(parsed_triplets) > 0,
            raw_output=raw_output,
            parsed_triplets=parsed_triplets,
            processing_time=processing_time
        )