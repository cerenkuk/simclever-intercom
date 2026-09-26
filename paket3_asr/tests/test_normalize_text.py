"""
Birim Test: paket3_asr/scripts/normalize_text.py
Türkçe Metin Normalizasyonu Testleri
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from normalize_text import (
    lowercase_tr,
    expand_numbers_in_text,
    strip_punctuation_and_formatting,
    normalize_turkish_asr
)

class TestNormalizeText(unittest.TestCase):

    def test_lowercase_tr(self):
        text = "İSTANBUL ve ILGIN ÇOK GÜZEL"
        expected = "istanbul ve ılgın çok güzel"
        self.assertEqual(lowercase_tr(text), expected)

    def test_expand_numbers_in_text(self):
        self.assertEqual(expand_numbers_in_text("120"), "yüz yirmi")
        self.assertEqual(expand_numbers_in_text("31"), "otuz bir")

    def test_strip_punctuation_and_formatting(self):
        text = "Merhaba, nasılsınız? Doktor Fatih Bey!"
        expected = "Merhaba nasılsınız Doktor Fatih Bey"
        self.assertEqual(strip_punctuation_and_formatting(text), expected)

    def test_normalize_turkish_asr_full(self):
        raw = "Bütün öğrencilerime merhaba! 120 hastaya baktım, yaşım 31."
        expected = "bütün öğrencilerime merhaba yüz yirmi hastaya baktım yaşım otuz bir"
        self.assertEqual(normalize_turkish_asr(raw), expected)

if __name__ == "__main__":
    unittest.main()
