import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    description: str
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout: int = 120

@dataclass
class ToolConfig:
    """Configuration for a tool that can be used by the model."""
    name: str
    description: str
    parameters: Dict[str, Any]

@dataclass
class PromptStrategyConfig:
    """Configuration for a function-calling prompt strategy."""
    name: str
    description: str
    tool_names: List[str]

@dataclass
class ExperimentConfig:
    """Configuration for the evaluation experiment."""
    models: List[str] = field(default_factory=list)
    prompt_strategies: List[str] = field(default_factory=list)
    output_dir: str = "results_react"
    max_retries: int = 1
    random_seed: int = 42

# --- Available Components ---

AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "gemma3:12b-it-qat": ModelConfig(
        name="gemma3:12b-it-qat",
        description="Google Gemma 3 12B Instruct, QAT."
    ),
    "gemma3:4b": ModelConfig(
        name="gemma3:4b",
        description="Google Gemma 3 4B Instruct."
    ),
    "gemma3:4b-it-fp16": ModelConfig(
        name="gemma3:4b-it-fp16",
        description="Google Gemma 3 4B Instruct, fp16 precision."
    ),
    "llama3.2:3b-instruct-fp16": ModelConfig(
        name="llama3.2:3b-instruct-fp16",
        description="Meta Llama 3.2 3B Instruct, fp16 precision."
    ),
}

AVAILABLE_TOOLS: Dict[str, ToolConfig] = {
    "extract_triplets": ToolConfig(
        name="extract_triplets",
        description="Extracts one or more information triplets (subject, relation, object) from a given Hindi sentence.",
        parameters={
            "type": "object",
            "properties": {
                "triplets": {
                    "type": "array",
                    "description": "A list of triplets found in the sentence.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "The subject of the relation."},
                            "relation": {"type": "string", "description": "The relation connecting the subject and object."},
                            "object": {"type": "string", "description": "The object of the relation."}
                        },
                        "required": ["subject", "relation", "object"]
                    }
                }
            },
            "required": ["triplets"]
        }
    )
}

AVAILABLE_PROMPT_STRATEGIES: Dict[str, PromptStrategyConfig] = {
    "react_cot_with_evidence": PromptStrategyConfig(
        name="react_cot_with_evidence",
        description="A ReAct based strategy that uses the model's native tool-calling feature.",
        tool_names=["extract_triplets"]
    )
}

class Config:
    def __init__(self):
        self.models = AVAILABLE_MODELS
        self.tools = AVAILABLE_TOOLS
        self.prompt_strategies = AVAILABLE_PROMPT_STRATEGIES
        
        self.experiment = ExperimentConfig(
            models=["gemma3:4b", "gemma3:12b-it-qat"],
            prompt_strategies=["react_cot_with_evidence"]
        )
        self.evaluation_data_path = os.path.join(os.path.dirname(__file__), "hindi_benchie_gold.txt")

    def get_model_config(self, model_key: str) -> ModelConfig:
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not found.")
        return self.models[model_key]
        
    def get_prompt_strategy(self, strategy_key: str) -> PromptStrategyConfig:
        if strategy_key not in self.prompt_strategies:
            raise ValueError(f"Prompt strategy '{strategy_key}' not found.")
        return self.prompt_strategies[strategy_key]

    def get_tool(self, tool_name: str) -> ToolConfig:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return self.tools[tool_name]

    def validate(self) -> bool:
        """Validates the configuration."""
        for model_key in self.experiment.models:
            if model_key not in self.models:
                raise ValueError(f"Experiment model '{model_key}' is not defined in available models.")
        
        for strategy_key in self.experiment.prompt_strategies:
            if strategy_key not in self.prompt_strategies:
                raise ValueError(f"Experiment strategy '{strategy_key}' is not defined in available strategies.")
            strategy = self.prompt_strategies[strategy_key]
            for tool_name in strategy.tool_names:
                if tool_name not in self.tools:
                    raise ValueError(f"Tool '{tool_name}' for strategy '{strategy_key}' not found.")
        
        if not os.path.exists(self.evaluation_data_path):
            print(f"Warning: Evaluation data file not found: {self.evaluation_data_path}")
            
        print("Configuration is valid.")
        return True

config = Config() 