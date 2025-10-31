from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this LLM provider."""
        pass

class GeminiProvider(LLMProvider):
    """Gemini implementation of the LLM provider interface."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.api_url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent'
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        import requests
        
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        response = requests.post(self.api_url, headers=headers, params=params, json=data)
        response.raise_for_status()
        result = response.json()
        
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return str(result)
    
    def get_name(self) -> str:
        return "Gemini"

class OpenAIProvider(LLMProvider):
    """OpenAI implementation (placeholder for future extension)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        # Placeholder for OpenAI implementation
        return "OpenAI response placeholder"
    
    def get_name(self) -> str:
        return "OpenAI"

class LLMRegistry:
    """Registry for managing multiple LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
    
    def register(self, name: str, provider: LLMProvider):
        """Register an LLM provider."""
        self._providers[name] = provider
    
    def get(self, name: str) -> Optional[LLMProvider]:
        """Get an LLM provider by name."""
        return self._providers.get(name)
    
    def list_available(self) -> list[str]:
        """List all available LLM providers."""
        return list(self._providers.keys())
    
    def generate_with_all(self, prompt: str, **kwargs) -> Dict[str, str]:
        """Generate text using all registered providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = provider.generate_text(prompt, **kwargs)
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[name] = f"Error: {e}"
        return results 