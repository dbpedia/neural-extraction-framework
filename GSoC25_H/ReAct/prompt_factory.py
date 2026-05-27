from abc import ABC, abstractmethod
from typing import List, Dict, Any

from config import Config, ToolConfig


class BasePromptStrategy(ABC):
    """Base class for a ReAct prompt strategy."""
    def __init__(self, strategy_config):
        self.name = strategy_config.name
        self.description = strategy_config.description
        self.tool_names = strategy_config.tool_names

    @abstractmethod
    def create_prompt(self, sentence: str, tools: List[ToolConfig]) -> List[Dict[str, Any]]:
        """Creates the full prompt messages for the LLM."""
        pass


class ReactCoTWithEvidence(BasePromptStrategy):
    """A ReAct based  strategy that uses the model's native tool-calling feature."""
    def create_prompt(self, sentence: str, tools: List[ToolConfig]) -> List[Dict[str, Any]]:
        system_prompt = """You are an expert AI for extracting information from Hindi text. Your task is to first reason step-by-step and then call a function with all extracted relation triplets.

**PRIMARY INSTRUCTION:**
First, provide your step-by-step reasoning. After your reasoning is complete, you MUST call the `extract_relation` function by providing a single, clean JSON object containing all the triplets you found.

**REASONING PROCESS:**
For each relation you find in the sentence, follow this thinking process:
1.  **Subject:** Identify the subject.
2.  **Evidence for Subject:** Quote the exact words from the sentence.
3.  **Relation:** Identify the relation.
4.  **Evidence for Relation:** Quote the exact words.
5.  **Object:** Identify the object.
6.  **Evidence for Object:** Quote the exact words.
7.  **Constructed Triplet:** Assemble the `(Subject, Relation, Object)` triplet.

**FUNCTION CALL FORMAT:**
After your reasoning, provide the JSON object in this exact format. Do not add any text after the JSON object.

{
    "name": "extract_relation",
    "parameters": {
        "triplets": [
            {"subject": "...", "relation": "...", "object": "..."}
        ]
    }
}

**EXAMPLE RESPONSE:**

**वाक्य:** भारतीय टीम ने फाइनल मैच में ऑस्ट्रेलिया को हराकर विश्व कप जीता।

**तर्क-वितर्क (Reasoning):**
*   **संबंध 1:**
    1.  **विषय:** भारतीय टीम ने
    2.  **सबूत:** "भारतीय टीम ने"
    3.  **संबंध:** हराकर
    4.  **सबूत:** "हराकर"
    5.  **कर्म:** ऑस्ट्रेलिया को
    6.  **सबूत:** "ऑस्ट्रेलिया को"
    7.  **अंतिम त्रिपद:** (भारतीय टीम ने, हराकर, ऑस्ट्रेलिया को)
*   **संबंध 2:**
    1.  **विषय:** भारतीय टीम ने
    2.  **सबूत:** "भारतीय टीम ने"
    3.  **संबंध:** जीता
    4.  **सबूत:** "जीता"
    5.  **कर्म:** विश्व कप
    6.  **सबूत:** "विश्व कप"
    7.  **अंतिम त्रिपद:** (भारतीय टीम ने, जीता, विश्व कप)

{
    "name": "extract_relation",
    "parameters": {
        "triplets": [
            {
                "subject": "भारतीय टीम ने",
                "relation": "हराकर",
                "object": "ऑस्ट्रेलिया को"
            },
            {
                "subject": "भारतीय टीम ने",
                "relation": "जीता",
                "object": "विश्व कप"
            }
        ]
    }
}

If you cannot find any relations, simply respond with a message explaining why. Do not call the function.
"""
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"**वाक्य:** {sentence}"}
            ]
        return messages

class PromptFactory:
    """Factory for creating prompts based on a selected strategy."""
    def __init__(self, config: Config):
        self.config = config
        self._strategies = {
            "react_cot_with_evidence": ReactCoTWithEvidence
        }

    def create_prompt(self, strategy_name: str, sentence: str) -> List[Dict[str, Any]]:
        """
        Creates a prompt for a given strategy and sentence.
        Returns the list of messages for the API call.
        """
        if strategy_name not in self._strategies:
            raise ValueError(f"Strategy '{strategy_name}' not supported.")
            
        strategy_config = self.config.get_prompt_strategy(strategy_name)
        strategy = self._strategies[strategy_name](strategy_config)
        
        tools = [self.config.get_tool(t_name) for t_name in strategy.tool_names]
        
        return strategy.create_prompt(sentence, tools)