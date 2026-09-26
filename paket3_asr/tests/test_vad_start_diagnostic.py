import unittest
import os
import sys
import wave

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_tests_dir, ".."))
_scripts_dir = os.path.join(_pkg_root, "scripts")
sys.path.insert(0, _scripts_dir)

from run_vad_start_diagnostic import (
    create_60s_snippet, get_existing_completed_keys, get_diagnostic_reference_text,
    calculate_leading_deletion, OUT_CSV, DIAGNOSTIC_DIR, SNIPPETS_DIR
)

class TestVadStartDiagnosticFast(unittest.TestCase):

    def test_60s_snippet_creation_and_duration(self):
        src_wav = os.path.join(_pkg_root, "data", "audio_16k", "video_01_genel_anamnez.wav")
        if not os.path.exists(src_wav):
            self.skipTest("video_01_genel_anamnez.wav mevcut değil.")

        dst_snippet = os.path.join(SNIPPETS_DIR, "test_snippet_video_01.wav")
        create_60s_snippet(src_wav, dst_snippet, max_seconds=60.0)

        self.assertTrue(os.path.exists(dst_snippet))
        with wave.open(dst_snippet, 'rb') as f:
            frames = f.getnframes()
            rate = f.getframerate()
            dur = frames / float(rate)
            self.assertLessEqual(dur, 60.01)
            self.assertLessEqual(frames, 960000)

    def test_diagnostic_reference_loading_word_counts(self):
        ref1 = get_diagnostic_reference_text("video_01_genel_anamnez")
        ref6 = get_diagnostic_reference_text("video_06_gogus_hastaliklari_oyku_alma")

        w1 = ref1.split()
        w6 = ref6.split()

        self.assertGreaterEqual(len(w1), 49)
        self.assertGreaterEqual(len(w6), 94)

    def test_leading_deletion_calculation_and_bound_rule(self):
        ref = ["bütün", "öğrencilerime", "merhaba", "ben", "iç", "hastalıkları", "öğretim"]
        hyp_del5 = ["hastalıkları", "öğretim", "üyesi"]
        leading = calculate_leading_deletion(ref, hyp_del5)
        self.assertEqual(leading, 5)
        # Fail-fast kuralı: leading_deletion hiçbir zaman total deletions (5) değerini aşamaz
        self.assertLessEqual(leading, 5)

        hyp_del0 = ["bütün", "öğrencilerime", "merhaba"]
        leading0 = calculate_leading_deletion(ref, hyp_del0)
        self.assertEqual(leading0, 0)

    def test_benchmark_results_csv_isolation(self):
        bm_csv = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        self.assertNotEqual(OUT_CSV, bm_csv)
        self.assertTrue(OUT_CSV.endswith("vad_start_diagnostic_results.csv"))

    def test_unique_key_resume_support(self):
        completed = get_existing_completed_keys(OUT_CSV)
        self.assertIsInstance(completed, set)

if __name__ == "__main__":
    unittest.main()
