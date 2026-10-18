import subprocess
import sys
import unittest
import types
from unittest.mock import Mock, patch

from GSoC25 import models


class TestLazyModelLoading(unittest.TestCase):

    def test_import_does_not_import_heavy_dependencies(self):
        """Verify importing models does not load Transformers or NLTK."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import GSoC25.models; "
                    "print('transformers' in sys.modules); "
                    "print('nltk' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertEqual(result.stdout.strip().splitlines(), ["False", "False"])

    def test_load_models_caches_result(self):
        """Verify repeated calls reuse the cached tokenizer and model."""
        tokenizer = object()
        model = object()

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.AutoTokenizer = Mock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
        mock_transformers.AutoModelForSeq2SeqLM = Mock()
        mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.return_value = model

        models.load_models.cache_clear()
        try:
            with patch.dict(sys.modules, {"transformers": mock_transformers}):
                first_result = models.load_models()
                second_result = models.load_models()

            self.assertIs(first_result, second_result)
            self.assertIs(first_result[0], tokenizer)
            self.assertIs(first_result[1], model)
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                models.GENRE_MODEL_NAME
            )
            mock_transformers.AutoModelForSeq2SeqLM.from_pretrained.assert_called_once_with(
                models.GENRE_MODEL_NAME
            )
        finally:
            models.load_models.cache_clear()

    def test_nltk_resource_check_skips_download_when_available(self):
        """Verify an available NLTK resource does not trigger a download."""
        mock_nltk = types.ModuleType("nltk")
        mock_nltk.data = Mock()
        mock_nltk.download = Mock()

        with patch.dict(sys.modules, {"nltk": mock_nltk}):
            models.ensure_nltk_resources()

        mock_nltk.data.find.assert_called_once_with("tokenizers/punkt")
        mock_nltk.download.assert_not_called()

    def test_nltk_resource_is_downloaded_when_missing(self):
        """Verify a missing NLTK resource is downloaded."""
        mock_nltk = types.ModuleType("nltk")
        mock_nltk.data = Mock()
        mock_nltk.data.find.side_effect = LookupError
        mock_nltk.download = Mock(return_value=True)

        with patch.dict(sys.modules, {"nltk": mock_nltk}):
            models.ensure_nltk_resources()

        mock_nltk.download.assert_called_once_with("punkt", quiet=True)


if __name__ == "__main__":
    unittest.main()
