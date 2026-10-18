"""On-demand loading of models and NLTK resources."""

from functools import lru_cache

GENRE_MODEL_NAME = "facebook/genre-linking-blink"


@lru_cache(maxsize=1)
def load_models():
    """Load and cache the GENRE tokenizer and model on first request."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GENRE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENRE_MODEL_NAME)

    return tokenizer, model


def ensure_nltk_resources():
    """Ensure the NLTK punkt resource is available, downloading it if needed."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        if not nltk.download("punkt", quiet=True):
            raise RuntimeError("Failed to download the NLTK 'punkt' resource.")
