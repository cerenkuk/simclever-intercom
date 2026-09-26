"""
Paket 3 - Yeni Analiz Scriptleri için Unit Testler

Kapsam:
- S/D/I küçük örnek testleri
- WER yeniden oluşturma testi
- Mükerrer analiz satırı koruması
- video/saha grup ayrımı
- Sabit konfigürasyon filtresi (A/B/C)
- Tekrar n-gram analizi
- Tıbbi terim eşleştirme
- Loudnorm config okuma
- Saha kayıtlarının haricî API kapsamı dışında kalma kuralı
- CSV şema ve tarihsel veri koruması
"""

import os
import sys
import csv
import unittest
import importlib.util

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_tests_dir, ".."))
_scripts_dir = os.path.join(_pkg_root, "scripts")
sys.path.insert(0, _scripts_dir)

from normalize_text import normalize_turkish_asr


def _load_module(name, filename):
    """Belirtilen scripts/ dosyasından modülü yükler."""
    filepath = os.path.join(_scripts_dir, filename)
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# 1. S/D/I Küçük Örnek Testleri
# ============================================================
class TestSDIAlignment(unittest.TestCase):
    """error_analysis.py compute_alignment fonksiyonu için birim testler."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("error_analysis", "error_analysis.py")

    def test_perfect_match(self):
        """Aynı diziler: tüm hit'ler, sıfır hata."""
        hits, subs, dels, ins, _ = self.mod.compute_alignment(["merhaba", "dunya"], ["merhaba", "dunya"])
        self.assertEqual(hits, 2)
        self.assertEqual(subs + dels + ins, 0)

    def test_single_substitution(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment(
            ["hasta", "solunum", "sikintisi"],
            ["hasta", "soluk", "sikintisi"]
        )
        self.assertEqual(subs, 1)
        self.assertEqual(dels, 0)
        self.assertEqual(ins, 0)
        self.assertEqual(hits, 2)

    def test_deletion(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment(["bir", "iki", "uc"], ["bir", "uc"])
        self.assertEqual(dels, 1)
        self.assertEqual(ins, 0)
        self.assertEqual(hits, 2)

    def test_insertion(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment(["bir", "uc"], ["bir", "iki", "uc"])
        self.assertEqual(ins, 1)
        self.assertEqual(dels, 0)
        self.assertEqual(hits, 2)

    def test_empty_ref(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment([], ["bir"])
        self.assertEqual(ins, 1)
        self.assertEqual(hits, 0)

    def test_empty_hyp(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment(["bir"], [])
        self.assertEqual(dels, 1)
        self.assertEqual(hits, 0)


# ============================================================
# 2. WER Yeniden Oluşturma Testi
# ============================================================
class TestWERReconstruction(unittest.TestCase):
    """S/D/I üzerinden WER hesaplama tutarlılığı."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("error_analysis", "error_analysis.py")

    def test_wer_formula(self):
        """WER = (S+D+I)/N."""
        ref = ["a", "b", "c", "d"]  # N=4
        hyp = ["a", "x", "d"]       # S=1, D=1 -> WER=0.5
        hits, subs, dels, ins, _ = self.mod.compute_alignment(ref, hyp)
        computed = self.mod.wer_check(subs, dels, ins, len(ref))
        self.assertAlmostEqual(computed, 0.5, places=2)

    def test_perfect_wer(self):
        hits, subs, dels, ins, _ = self.mod.compute_alignment(["merhaba"], ["merhaba"])
        computed = self.mod.wer_check(subs, dels, ins, 1)
        self.assertEqual(computed, 0.0)


# ============================================================
# 3. Mükerrer Analiz Satırı Koruması
# ============================================================
class TestDuplicateProtection(unittest.TestCase):
    def _check_no_duplicates(self, csv_path, desc):
        if not os.path.exists(csv_path):
            self.skipTest(f"{desc} henüz oluşturulmadı.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        keys = [(r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
                for r in rows]
        self.assertEqual(len(keys), len(set(keys)), f"{desc}: {len(keys)-len(set(keys))} mükerrer anahtar var.")

    def test_no_duplicate_in_error_csv(self):
        self._check_no_duplicates(
            os.path.join(_pkg_root, "outputs", "metrics", "error_analysis_results.csv"),
            "error_analysis_results.csv"
        )

    def test_no_duplicate_in_hallucination_csv(self):
        self._check_no_duplicates(
            os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_results.csv"),
            "hallucination_analysis_results.csv"
        )


# ============================================================
# 4. Video/Saha Grup Ayrımı
# ============================================================
class TestGroupSeparation(unittest.TestCase):
    def _group(self, bn):
        return "video" if bn.startswith("video_") else ("saha" if bn.startswith("saha_") else "diger")

    def test_video_basenames(self):
        for bn in ["video_01_genel_anamnez", "video_02_psikiyatrik_anamnez", "video_06_gogus_hastaliklari_oyku_alma"]:
            self.assertEqual(self._group(bn), "video")

    def test_saha_basenames(self):
        for bn in ["saha_01_img_3463", "saha_02_img_3492", "saha_04_vid_20220913_142911"]:
            self.assertEqual(self._group(bn), "saha")

    def test_benchmark_csv_group_counts(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        main = [r for r in rows if r["run_purpose"] == "main_benchmark"
                and r["protocol_version"] == "pilot-v2-antiloop"
                and r["model_id"] == "faster-whisper-large-v3-int8"]
        basenames = set(r["basename"] for r in main)
        videos = [bn for bn in basenames if bn.startswith("video_")]
        sahas = [bn for bn in basenames if bn.startswith("saha_")]
        self.assertEqual(len(videos), 6, f"Video sayısı 6 olmalı: {videos}")
        self.assertEqual(len(sahas), 4, f"Saha sayısı 4 olmalı: {sahas}")


# ============================================================
# 5. Sabit Konfigürasyon Filtresi (A/B/C)
# ============================================================
class TestFixedConfigurationFilter(unittest.TestCase):
    CONFIGS = {
        "A": ("faster-whisper-large-v3-int8", "none"),
        "B": ("faster-whisper-large-v3-turbo-int8", "none"),
        "C": ("faster-whisper-large-v3-turbo-int8", "mild"),
    }

    def test_fixed_config_results_exist(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        main = [r for r in rows if r["run_purpose"] == "main_benchmark"
                and r["protocol_version"] == "pilot-v2-antiloop"]
        for label, (model, cond) in self.CONFIGS.items():
            subset = [r for r in main if r["model_id"] == model and r["condition"] == cond]
            self.assertEqual(len(subset), 10, f"Konfig {label}: 10 kayıt bekleniyor, {len(subset)} var.")


# ============================================================
# 6. Tekrar N-gram Analizi
# ============================================================
class TestNgramRepeat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("hallucination_analysis", "hallucination_analysis.py")

    def test_no_repeat(self):
        count, reps = self.mod.count_ngram_repeats(["a", "b", "c", "d"], 2)
        self.assertEqual(count, 0)

    def test_bigram_repeat(self):
        count, reps = self.mod.count_ngram_repeats(["a", "b", "a", "b", "c"], 2)
        self.assertGreater(count, 0)
        self.assertEqual(reps[0][0], ("a", "b"))

    def test_consecutive_repeat(self):
        total, max_chain, chains = self.mod.count_consecutive_repeats(["hasta", "hasta", "hasta", "geliyor"])
        self.assertEqual(total, 2)
        self.assertEqual(max_chain, 3)


# ============================================================
# 7. Tıbbi Terim Eşleştirme
# ============================================================
class TestMedicalTermMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("medical_term_analysis", "medical_term_analysis.py")

    def _words(self, text):
        norm = normalize_turkish_asr(text)
        return norm.split(), set(norm.split())

    def test_exact_match(self):
        words, word_set = self._words("hasta dispne nedeniyle acile geldi")
        self.assertTrue(self.mod.term_in_text("dispne", word_set, words))

    def test_prefix_match_with_suffix(self):
        """Ek almış kelime - minimum 5 karakter prefix."""
        words, word_set = self._words("antidepresandan kesmemesi gerekiyor")
        self.assertTrue(self.mod.term_in_text("antidepresan", word_set, words))

    def test_no_false_substring_positive(self):
        """'psi' 'psikiyatrik' için eşleşmemeli."""
        words, word_set = self._words("psi soyadli bir hasta")
        self.assertFalse(self.mod.term_in_text("psikiyatrik", word_set, words))

    def test_multiword_term(self):
        words, word_set = self._words("vagal manevra denendi ve ritim normale dondü")
        self.assertTrue(self.mod.term_in_text("vagal manevra", word_set, words))


# ============================================================
# 8. Loudnorm Config Okuma
# ============================================================
class TestLoudnormConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module("prepare_audio", "prepare_audio.py")

    def test_loudnorm_config_keys(self):
        cfg = self.mod.load_loudnorm_config()
        for key in ["integrated_loudness", "loudness_range", "true_peak", "sample_rate", "channels"]:
            self.assertIn(key, cfg, f"Eksik loudnorm config anahtarı: {key}")
        self.assertEqual(cfg["sample_rate"], 16000)
        self.assertEqual(cfg["channels"], 1)
        self.assertEqual(cfg["integrated_loudness"], -23.0)

    def test_loudnorm_in_benchmark_condition_map(self):
        with open(os.path.join(_scripts_dir, "benchmark.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"loudnorm"', src)
        self.assertIn("audio_normalized_loudnorm", src)


# ============================================================
# 9. Saha Kayıtlarının Harici API Kapsamı Dışında Kalma Kuralı
# ============================================================
class TestSahaAPIScope(unittest.TestCase):
    def test_saha_not_in_external_api_scope(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["basename"].startswith("saha_"):
                self.assertNotIn("openai", r.get("model_id", "").lower())
                self.assertNotIn("cloud", r.get("engine", "").lower())


# ============================================================
# 10. CSV Şema ve Tarihsel Veri Koruması
# ============================================================
class TestCSVSchemaIntegrity(unittest.TestCase):
    def test_benchmark_csv_required_fields(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        for field in ["model_id", "basename", "condition", "protocol_version", "run_purpose",
                      "WER", "CER", "RTF", "transcription", "reference_text"]:
            self.assertIn(field, fields, f"Gerekli alan eksik: {field}")

    def test_error_analysis_csv_schema(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "error_analysis_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("error_analysis_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        for field in ["model_id", "basename", "condition", "substitutions", "deletions",
                      "insertions", "reference_word_count", "dominant_error_type"]:
            self.assertIn(field, fields, f"error_analysis_results.csv'de alan eksik: {field}")

    def test_historical_rows_preserved(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        turbo_rows = [r for r in rows if r["model_id"] == "faster-whisper-large-v3-turbo-int8"
                      and r["run_purpose"] == "main_benchmark"]
        large_rows = [r for r in rows if r["model_id"] == "faster-whisper-large-v3-int8"
                      and r["run_purpose"] == "main_benchmark"]
        self.assertGreaterEqual(len(turbo_rows), 30)
        self.assertGreaterEqual(len(large_rows), 30)

    def test_medium_30_unique_rows(self):
        csv_path = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            self.skipTest("benchmark_results.csv yok.")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        medium = [r for r in rows
                  if r["model_id"] == "faster-whisper-medium-int8"
                  and r["run_purpose"] == "main_benchmark"
                  and r["protocol_version"] == "pilot-v2-antiloop"]
        keys = set((r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
                   for r in medium)
        self.assertGreaterEqual(len(keys), 30, f"Medium benzersiz satır en az 30 olmalı, {len(keys)} bulundu.")



if __name__ == "__main__":
    unittest.main(verbosity=2)
