from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """
    Abstract Base Class for Neural Models.
    Ensures that any future model (llama.cpp, vLLM, HuggingFace) 
    adheres to the same contract.
    """
    
    @abstractmethod
    def extract(self, text):
        """
        Extracts entities and relations from text.
        Returns: (subject, object, predicate, confidence)
        """
        pass
