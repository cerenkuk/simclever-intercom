"""
Birim Test: paket3_asr/scripts/calculate_metrics.py
WER ve CER Hesaplama Motoru Testleri
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from calculate_metrics import evaluate_asr_pair

class TestCalculateMetrics(unittest.TestCase):

    def test_exact_match(self):
        ref = "doktor fatih kamis"
        hyp = "doktor fatih kamis"
        res = evaluate_asr_pair(ref, hyp, normalize=True)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["cer"], 0.0)

    def test_substitution(self):
        ref = "doktor fatih kamis"
        hyp = "doktor fatih cetin"
        res = evaluate_asr_pair(ref, hyp, normalize=True)
        self.assertAlmostEqual(res["wer"], 1/3, places=4)

    def test_insertion(self):
        ref = "doktor fatih kamis"
        hyp = "doktor fatih fatih kamis kamis"
        res = evaluate_asr_pair(ref, hyp, normalize=True)
        self.assertAlmostEqual(res["wer"], 2/3, places=4)

    def test_deletion(self):
        ref = "doktor fatih kamis"
        hyp = "doktor fatih"
        res = evaluate_asr_pair(ref, hyp, normalize=True)
        self.assertAlmostEqual(res["wer"], 1/3, places=4)

    def test_empty_strings(self):
        res = evaluate_asr_pair("", "", normalize=True)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["cer"], 0.0)

if __name__ == "__main__":
    unittest.main()
