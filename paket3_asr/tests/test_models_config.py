"""
Birim Test: paket3_asr/config/models.yaml ve benchmark.py Model Doğrulama Testleri
"""

import sys
import os
import unittest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from benchmark import get_model_spec, MODELS_YAML

class TestModelsConfig(unittest.TestCase):

    def test_models_yaml_exists(self):
        self.assertTrue(os.path.exists(MODELS_YAML))

    def test_models_yaml_parsable(self):
        with open(MODELS_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("models", cfg)
        self.assertGreater(len(cfg["models"]), 0)

    def test_active_turbo_finalist_valid(self):
        spec = get_model_spec("faster-whisper-large-v3-turbo-int8")
        self.assertEqual(spec["id"], "faster-whisper-large-v3-turbo-int8")
        self.assertEqual(spec["engine"], "faster-whisper")
        self.assertEqual(spec["model_name"], "large-v3-turbo")
        self.assertTrue(spec["enabled"])

    def test_deprecated_model_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_model_spec("faster-whisper-turbo-int8")
        self.assertIn("deprecated", str(ctx.exception).lower())

    def test_disabled_model_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_model_spec("whisper-large-v3")
        self.assertIn("enabled: false", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
