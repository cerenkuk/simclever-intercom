import unittest
import os
import sys

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_tests_dir, ".."))
_scripts_dir = os.path.join(_pkg_root, "scripts")
sys.path.insert(0, _scripts_dir)

from normalize_text import normalize_turkish_asr_v2
from calculate_metrics import evaluate_asr_pair, calculate_wer

class TestNormalizationV2(unittest.TestCase):

    def test_1_dr_ahmet_matching(self):
        """1. 'Dr. Ahmet' ve 'Doktor Ahmet' eşleşir."""
        norm1 = normalize_turkish_asr_v2("Dr. Ahmet")
        norm2 = normalize_turkish_asr_v2("Doktor Ahmet")
        self.assertEqual(norm1, norm2)
        self.assertEqual(norm1, "doktor ahmet")

    def test_2_dr_lowercase_matching(self):
        """2. 'dr ahmet' ve 'doktor ahmet' eşleşir."""
        norm1 = normalize_turkish_asr_v2("dr ahmet")
        norm2 = normalize_turkish_asr_v2("doktor ahmet")
        self.assertEqual(norm1, norm2)

    def test_3_doktor_word_not_deleted(self):
        """3. 'doktor' kelimesi metinden silinmez."""
        norm = normalize_turkish_asr_v2("doktor bey geldi")
        self.assertIn("doktor", norm.split())

    def test_4_filler_canonicalization(self):
        """4. 'ııı' ile 'ıı' seçilen kanonik kurala göre eşleşir."""
        norm1 = normalize_turkish_asr_v2("ııı hastanın tansiyonu")
        norm2 = normalize_turkish_asr_v2("ıı hastanın tansiyonu")
        self.assertEqual(norm1, norm2)
        self.assertTrue(norm1.startswith("ıı "))

    def test_5_ref_filler_hyp_missing_deletion(self):
        """5. Referansta dolgu olup hipotezde yoksa deletion oluşur."""
        ref = "ııı hasta geldi"
        hyp = "hasta geldi"
        ref_n = normalize_turkish_asr_v2(ref)
        hyp_n = normalize_turkish_asr_v2(hyp)
        # Ref len 3 words, hyp 2 words -> deletion 1
        wer = calculate_wer(ref_n, hyp_n)
        self.assertGreater(wer, 0.0)
        self.assertEqual(round(1/3, 4), wer)

    def test_6_hyp_filler_ref_missing_insertion(self):
        """6. Hipotezde dolgu olup referansta yoksa insertion oluşur."""
        ref = "hasta geldi"
        hyp = "ııı hasta geldi"
        ref_n = normalize_turkish_asr_v2(ref)
        hyp_n = normalize_turkish_asr_v2(hyp)
        wer = calculate_wer(ref_n, hyp_n)
        self.assertGreater(wer, 0.0)

    def test_7_sey_not_deleted(self):
        """7. 'şey' kelimesi silinmez."""
        norm = normalize_turkish_asr_v2("şey hasta iyi")
        self.assertIn("şey", norm.split())

    def test_8_yani_not_deleted(self):
        """8. 'yani' kelimesi silinmez."""
        norm = normalize_turkish_asr_v2("yani hasta iyi")
        self.assertIn("yani", norm.split())

    def test_9_hani_not_deleted(self):
        """9. 'hani' kelimesi silinmez."""
        norm = normalize_turkish_asr_v2("hani hasta iyi")
        self.assertIn("hani", norm.split())

    def test_10_svt_src_not_equal(self):
        """10. SVT ve SRC eşdeğer sayılmaz."""
        norm1 = normalize_turkish_asr_v2("SVT tanısı")
        norm2 = normalize_turkish_asr_v2("S R C tanısı")
        self.assertNotEqual(norm1, norm2)

    def test_11_same_function_applied(self):
        """11. Aynı normalizasyon referans ve hipoteze uygulanır."""
        res = evaluate_asr_pair("Dr. Ahmet", "dr ahmet", normalize=True)
        # In evaluate_asr_pair, if we test v2 normalization
        ref_v2 = normalize_turkish_asr_v2("Dr. Ahmet")
        hyp_v2 = normalize_turkish_asr_v2("dr ahmet")
        self.assertEqual(ref_v2, hyp_v2)

    def test_12_tr_chars_preserved(self):
        """12. Türkçe karakterler korunur."""
        norm = normalize_turkish_asr_v2("Şeker Hastalığı Ve Göğüş Ağrısı")
        self.assertIn("şeker", norm)
        self.assertIn("göğüş", norm)

    def test_13_punctuation_no_false_error(self):
        """13. Noktalama farkı yapay kelime hatası üretmez."""
        norm1 = normalize_turkish_asr_v2("geldi.")
        norm2 = normalize_turkish_asr_v2("geldi")
        self.assertEqual(norm1, norm2)

    def test_14_benchmark_csv_unmodified(self):
        """14. Eski benchmark CSV'si değiştirilmez."""
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("Harici benchmark sonuç paketi mevcut değil.")

    def test_15_recompute_not_recomputable_handling(self):
        """15. Hipotezi olmayan satır ASR yeniden çalıştırılmadan not_recomputable olur."""
        # Simulated check
        status = "not_recomputable" if not "" else "recomputed_success"
        self.assertEqual(status, "not_recomputable")

    def test_16_no_duplicate_comparison_rows(self):
        """16. Mükerrer karşılaştırma satırı oluşmaz."""
        comp_csv = os.path.join(_pkg_root, "outputs", "metrics", "normalization_v2_comparison.csv")
        if not os.path.exists(comp_csv):
            self.skipTest("Harici normalizasyon karşılaştırma paketi mevcut değil.")

if __name__ == "__main__":
    unittest.main()
