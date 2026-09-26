"""
Birim Test: paket3_asr/scripts/medical_term_score.py
Tıbbi Terim İsabeti ve Halüsinasyon Puanlama Testleri
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from medical_term_score import (
    calculate_medical_term_recall,
    count_hallucinations_in_silence,
    DEFAULT_MEDICAL_TERMS
)

class TestMedicalTermScore(unittest.TestCase):

    def test_default_medical_terms(self):
        self.assertIn("anamnez", DEFAULT_MEDICAL_TERMS)
        self.assertIn("tansiyon", DEFAULT_MEDICAL_TERMS)

    def test_calculate_medical_term_recall(self):
        ref = "Hastaya anamnez uygulaması yapıldı. Tansiyon normal, solunum rahat."
        hyp = "Hastaya anamnez yapıldı. Tansiyon normal, solunum iyi."
        res = calculate_medical_term_recall(ref, hyp)
        self.assertGreaterEqual(res["medical_term_accuracy"], 0.5)
        self.assertIn("anamnez", res["found_terms"])

    def test_count_hallucinations_in_silence(self):
        silence_hyp = "karın ağrısı karın ağrısı karın ağrısı"
        count = count_hallucinations_in_silence(silence_hyp)
        self.assertEqual(count, 6)

if __name__ == "__main__":
    unittest.main()
