import unittest
import os
import sys
import csv

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_tests_dir, ".."))
_scripts_dir = os.path.join(_pkg_root, "scripts")
sys.path.insert(0, _scripts_dir)

from hallucination_analysis_v3 import run_v3_analysis, classify_loop_v3

class TestManualReviewIntegrity(unittest.TestCase):
    
    def test_manual_review_csv_exists_and_valid(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_manual_review.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Harici manuel inceleme sonuç paketi mevcut değil.")
        
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            
        self.assertEqual(len(rows), 9, "Manuel gözden geçirme CSV'sinde tam 9 kayıt olmalı.")
        
        verdicts = set(r["manual_verdict"] for r in rows)
        self.assertIn("CONFIRMED_PATHOLOGICAL_ASR_LOOP", verdicts)
        self.assertIn("CONFIRMED_PATHOLOGICAL_ASR_LOOP_FROM_REAL_TOKEN", verdicts)
        self.assertIn("CONFIRMED_REAL_SPEECH_REPETITION_NOT_LOOP", verdicts)
        self.assertIn("CONFIRMED_MEDICAL_TERM_SUBSTITUTION_WITH_OVER_REPETITION", verdicts)

    def test_v3_csv_contains_manual_verdicts(self):
        v3_path = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_v3_results.csv")
        if not os.path.exists(v3_path):
            self.skipTest("Harici V3 sonuç paketi mevcut değil.")
        
        with open(v3_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            
        completed_rows = [r for r in rows if r.get("manual_review_status") == "completed"]
        self.assertEqual(len(completed_rows), 9, "V3 CSV'sinde tam 9 satır manual_review_status=completed olmalı.")

if __name__ == "__main__":
    unittest.main()
