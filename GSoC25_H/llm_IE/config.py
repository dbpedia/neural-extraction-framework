import os
from typing import Dict, List, Any, Union
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    description: str
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 1024
    timeout: int = 300

@dataclass 
class ExperimentConfig:
    models: List[str] = field(default_factory=list)
    prompt_strategies: List[str] = field(default_factory=list)
    evaluation_sets: List[str] = field(default_factory=list)
    output_dir: str = "results"
    batch_size: int = 10
    max_retries: int = 3
    random_seed: int = 42

AVAILABLE_MODELS = {
    "mistral:latest": ModelConfig(
        name="mistral:latest",
        description="Mistral 7B",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "gemma3:4b": ModelConfig(
        name="gemma3:4b", 
        description="Google Gemma 3 4B",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "gemma3:1b": ModelConfig(
        name="gemma3:1b",
        description="Google Gemma 3 1B",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "gemma3:4b-it-fp16": ModelConfig(
        name="gemma3:4b-it-fp16", 
        description="Google Gemma 3 4B - fp16 precision",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "llama3.2:3b-instruct-fp16": ModelConfig(
        name="llama3.2:3b-instruct-fp16", 
        description="Llama 3.2 3B - fp16 instruction-tuned",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    
    "llama3.2:1b-text-fp16": ModelConfig(
        name="llama3.2:1b-text-fp16", 
        description="Llama 3.2 1B - fp16 text-tuned",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "qwen3:4b-fp16": ModelConfig(
        name="qwen3:4b-fp16",
        description="Qwen 3 4B - fp16 precision",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024
    ),
    "gemma3:12b-it-qat": ModelConfig(
        name="gemma3:12b-it-qat",
        description="Google Gemma 3 12B - 4-bit Quantized (best performer yet - 33.7% F1)",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        max_tokens=1024,
        timeout=300
    )
}

EVALUATION_CONFIG = {
    "benchie_gold_file": "../hindi-benchie/hindi_benchie_gold.txt"
}

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: str = None):
        self.models = AVAILABLE_MODELS
        self.evaluation = EVALUATION_CONFIG
        
        self.experiment = ExperimentConfig(
            # best performing model is gemma3:12b-it-qat
            models=["gemma3:4b", "gemma3:12b-it-qat"], 
            prompt_strategies=["chain_of_thought_ER_english_hindi"], 
            evaluation_sets=["hindi_benchie_gold.txt"]
        )
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def get_model_config(self, model_key: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found in available models")
        return self.models[model_key]
    
    def get_models(self) -> Dict[str, ModelConfig]:
        """Get all available models"""
        return self.models
  
    def validate(self) -> bool:
        """Validate configuration settings"""
        for model_key, model_config in self.models.items():
            if not model_config.name:
                raise ValueError(f"Model '{model_key}' has no name specified")
        
        benchie_file = self.evaluation["benchie_gold_file"]
        if not os.path.exists(benchie_file):
            print(f"Warning: Benchie gold file not found: {benchie_file}")
        
        return True

config = Config() 