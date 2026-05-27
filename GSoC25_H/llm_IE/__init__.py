"""
LLM Information Extraction Pipeline for Hindi Relation Extraction

This package provides a comprehensive evaluation framework for assessing
different Large Language Models (LLMs) on Hindi relation extraction tasks
using the hindi-benchie evaluation framework.

Key Components:
- Ollama model support
- Structured prompt engineering
- Hindi-Benchie Integration (using the actual evaluation code from the benchie repository)
- error analysis and reporting
"""

__version__ = "1.0.0"
__author__ = "GSoC 2025 Hindi Neural Extraction Framework"

from .config import Config, AVAILABLE_MODELS
from .llm_interface import OllamaInterface, ExtractionResult
from .benchie_evaluator import ProperBenchieEvaluator
from .prompt_templates import PromptTemplateManager
from .output_parser import OutputParser
from .error_analyzer import ErrorAnalyzer
from .reporter import Reporter

__all__ = [
    "Config",
    "AVAILABLE_MODELS",
    "OllamaInterface",
    "ExtractionResult",
    "ProperBenchieEvaluator",
    "PromptTemplateManager",
    "OutputParser",
    "ErrorAnalyzer",
    "Reporter"
]