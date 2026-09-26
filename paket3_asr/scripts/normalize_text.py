"""
Paket 3 - Türkçe ASR Metin Normalizasyonu Engine (turkish_asr_verbatim_v2)
SimClever Intercom - Türkçe ASR Sürümlü Normalizör
"""

import re
import sys
import os

TR_LOWER_MAP = {
    ord('İ'): 'i',
    ord('I'): 'ı',
    ord('Ğ'): 'ğ',
    ord('Ü'): 'ü',
    ord('Ş'): 'ş',
    ord('Ö'): 'ö',
    ord('Ç'): 'ç'
}

UNITS = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]

def number_to_words_tr(n: int) -> str:
    if n == 0:
        return "sıfır"
    parts = []
    thousands = n // 1000
    remainder = n % 1000
    if thousands > 0:
        if thousands == 1:
            parts.append("bin")
        else:
            parts.append(number_to_words_tr(thousands) + " bin")
    if remainder > 0:
        hundreds = remainder // 100
        rem_tens = remainder % 100
        if hundreds > 0:
            if hundreds == 1:
                parts.append("yüz")
            else:
                parts.append(f"{UNITS[hundreds]} yüz")
        if rem_tens > 0:
            tens = rem_tens // 10
            units = rem_tens % 10
            if tens > 0:
                parts.append(TENS[tens])
            if units > 0:
                parts.append(UNITS[units])
    return " ".join(parts)

def expand_numbers_in_text(text: str) -> str:
    def replace_num(match):
        num_str = match.group(0)
        try:
            val = int(num_str)
            return number_to_words_tr(val)
        except ValueError:
            return num_str
    return re.sub(r'\b\d+\b', replace_num, text)

def lowercase_tr(text: str) -> str:
    return text.translate(TR_LOWER_MAP).lower()

def strip_punctuation_and_formatting(text: str) -> str:
    cleaned = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Legacy v1 Normalization Function (Preserved)
def normalize_turkish_asr(text: str, expand_numbers: bool = True) -> str:
    if not text:
        return ""
    text = lowercase_tr(text)
    if expand_numbers:
        text = expand_numbers_in_text(text)
    text = strip_punctuation_and_formatting(text)
    return text

# New v2 Verbatim Normalization Function (turkish_asr_verbatim_v2)
def normalize_turkish_asr_v2(text: str, expand_numbers: bool = True) -> str:
    """
    Sürümlü Normalizör: turkish_asr_verbatim_v2
    - dr / dr. / doktor -> doktor kanonik dönüşümü
    - Non-lexical dolgu sesleri kanonikleştirmesi (ııı -> ıı, eee -> ee, hım/hmm -> hım)
    - Sözcüksel dolgu kelimeleri (şey, yani, hani, işte) ve medikal terimler (svt vb.) birebir koruma
    """
    if not text:
        return ""

    # 1. Küçük harf dönüşümü
    text = lowercase_tr(text)

    # 2. Özel Kısaltma Ön-Eşleme (Noktalama temizlenmeden önce dr. için)
    # dr. kelimesini doktor yap
    text = re.sub(r'\bdr\.\b', 'doktor', text)

    # 3. Noktalama ve kesme temizliği
    text = strip_punctuation_and_formatting(text)

    # 4. Rakam genişletme
    if expand_numbers:
        text = expand_numbers_in_text(text)

    # 5. Kelime Seviyesi Mimarisi ve Kanonikleştirme
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        # dr -> doktor
        if tok == "dr":
            tok = "doktor"
        # Non-lexical dolgu kanonikleştirmesi (Yalnızca açık uzatmalar: len > 1)
        # ıı, ııı, ıııı... -> ıı (tek 'ı' korunur)
        elif re.fullmatch(r'ı+', tok) and len(tok) > 1:
            tok = "ıı"
        # ee, eee, eeee... -> ee (tek 'e' korunur)
        elif re.fullmatch(r'e+', tok) and len(tok) > 1:
            tok = "ee"
        # hmm, hmmm -> hım (hı ve hım aynen korunur)
        elif tok in ["hmm", "hmmm"]:
            tok = "hım"

        normalized_tokens.append(tok)


    return " ".join(normalized_tokens)

if __name__ == "__main__":
    sample = "Dr. Ahmet Bey hasta için dr önerdi. Iıı... hasta ee şey dedi, hmmm."
    print("v1 norm:", normalize_turkish_asr(sample))
    print("v2 norm:", normalize_turkish_asr_v2(sample))
