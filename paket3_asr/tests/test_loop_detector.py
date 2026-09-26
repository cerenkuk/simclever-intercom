import unittest
import os
import sys

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_tests_dir, ".."))
_scripts_dir = os.path.join(_pkg_root, "scripts")
sys.path.insert(0, _scripts_dir)

from hallucination_analysis_v3 import classify_loop_v3

class TestLoopDetector(unittest.TestCase):
    
    def test_case_1_consecutive_token_pathological(self):
        """1. 'çek' x74 ardışık -> pathological"""
        loop_detected, severity, reason = classify_loop_v3(longest_token_run=74, consec_token_count=73, longest_phrase_run=37, ins_rate=0.12, hyp_ref_ratio=1.06)
        self.assertEqual(loop_detected, "EVET")
        self.assertEqual(severity, "pathological")

    def test_case_2_distributed_phrase_not_loop(self):
        """2. 'işe yaramazsa' farklı yerlerde x5 -> loop değil (HAYIR, none)"""
        loop_detected, severity, reason = classify_loop_v3(longest_token_run=0, consec_token_count=0, longest_phrase_run=0, ins_rate=0.04, hyp_ref_ratio=0.92)
        self.assertEqual(loop_detected, "HAYIR")
        self.assertEqual(severity, "none")

    def test_case_3_short_natural_reduplication_not_loop(self):
        """3. kısa doğal ikileme ('çok çok') -> loop değil"""
        loop_detected, severity, reason = classify_loop_v3(longest_token_run=2, consec_token_count=1, longest_phrase_run=0, ins_rate=0.01, hyp_ref_ratio=1.0)
        self.assertEqual(loop_detected, "HAYIR")
        self.assertEqual(severity, "none")

    def test_case_4_consecutive_phrase_pathological(self):
        """4. uzun ardışık phrase tekrarı ('herhalde herhalde' x36 bitişik) -> pathological"""
        loop_detected, severity, reason = classify_loop_v3(longest_token_run=2, consec_token_count=2, longest_phrase_run=12, ins_rate=0.06, hyp_ref_ratio=0.97)
        self.assertEqual(loop_detected, "EVET")
        self.assertEqual(severity, "pathological")

if __name__ == "__main__":
    unittest.main()
