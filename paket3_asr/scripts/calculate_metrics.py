"""
Paket 3 - WER (Word Error Rate) & CER (Character Error Rate) Ölçüm Engine (Bölüm 3 & 9)
SimClever Intercom - Türkçe ASR Kıyas Metrikleri
"""

import sys
import os

# Normalizasyon modülünü içe aktarma
sys.path.append(os.path.dirname(__file__))
from normalize_text import normalize_turkish_asr

def levenshtein_distance(ref_tokens, hyp_tokens):
    """Token veya Karakter dizileri için Levenshtein düzenleme mesafesi hesaplar."""
    n = len(ref_tokens)
    m = len(hyp_tokens)

    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return dp[n][m]

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Kelime Hata Oranı (Word Error Rate - WER) hesaplar."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Karakter Hata Oranı (Character Error Rate - CER) hesaplar."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return round(dist / len(ref_chars), 4)

def compute_sdi_counts(reference: str, hypothesis: str):
    """Ref ve Hyp kelime dizileri arasında S, D, I sayılarını hesaplar."""
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0, 0, m
    if m == 0:
        return 0, n, 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    i, j = n, m
    sub, mole, ins = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            sub += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            mole += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, mole, ins

def calculate_wer_cer(reference: str, hypothesis: str):
    """WER, CER ve S, D, I sayılarını döndüren birleşik hesaplayıcı."""
    wer = calculate_wer(reference, hypothesis)
    cer = calculate_cer(reference, hypothesis)
    sub, mole, ins = compute_sdi_counts(reference, hypothesis)
    return wer, cer, sub, mole, ins


def evaluate_asr_pair(reference: str, hypothesis: str, normalize: bool = True) -> dict:
    """Referans ve Model Çıktısı Çiftini Paket 3 Protokolüne Göre Değerlendirir."""
    if normalize:
        ref_norm = normalize_turkish_asr(reference)
        hyp_norm = normalize_turkish_asr(hypothesis)
    else:
        ref_norm = reference.strip()
        hyp_norm = hypothesis.strip()

    wer = calculate_wer(ref_norm, hyp_norm)
    cer = calculate_cer(ref_norm, hyp_norm)

    return {
        "wer": wer,
        "cer": cer,
        "ref_length_words": len(ref_norm.split()),
        "hyp_length_words": len(hyp_norm.split()),
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm
    }

if __name__ == "__main__":
    ref = "Hasta solunum sıkıntısı ve öksürük şikayetiyle başvurdu."
    hyp = "Hasta solunumu sıkıntısı ve öksürük şikayeti ile başvurdu."

    result = evaluate_asr_pair(ref, hyp)
    print("--- WER & CER Ölçüm Testi ---")
    print(f"Referans Norm  : {result['reference_normalized']}")
    print(f"Model Norm     : {result['hypothesis_normalized']}")
    print(f"WER (Kelime)   : %{result['wer'] * 100:.2f}")
    print(f"CER (Karakter) : %{result['cer'] * 100:.2f}")
