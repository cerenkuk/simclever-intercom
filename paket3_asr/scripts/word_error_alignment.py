"""
Paket 3 - Aşama 3: video_01 ve video_06 İlk 20 Hata Analizi
Model: faster-whisper-large-v3-int8 + none
Model: faster-whisper-large-v3-turbo-int8 + none
Koşul: none (ham kayıt)
"""

import csv
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_script_dir)
from normalize_text import normalize_turkish_asr

_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
CSV_IN = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")

TARGET_MODELS = [
    "faster-whisper-large-v3-int8",
    "faster-whisper-large-v3-turbo-int8",
]
TARGET_BASENAMES = [
    "video_01_genel_anamnez",
    "video_06_gogus_hastaliklari_oyku_alma",
]
TARGET_CONDITION = "none"


def compute_alignment(ref_tokens, hyp_tokens):
    """Levenshtein DP + geri izleme ile S/D/I ve hizalamayı çıkarır."""
    n = len(ref_tokens)
    m = len(hyp_tokens)
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

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            aligned.append(("H", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            aligned.append(("S", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append(("D", ref_tokens[i - 1], None))
            i -= 1
        else:
            aligned.append(("I", None, hyp_tokens[j - 1]))
            j -= 1
    aligned.reverse()
    return aligned


def classify_error(ref_w, hyp_w, aligned, idx):
    """
    Basit kural tabanlı muhtemel neden sınıflandırması.
    Sınıflar: gerçek_ASR_hatası / referans_transkript_şüphesi /
              normalizasyon_şüphesi / tıbbi_terim_veya_özel_isim /
              VAD_veya_ses_başı_sonu / belirsiz
    """
    # Tıbbi/özel terim aday listesi (kısmi - referans metinden çıkarılır)
    MEDICAL_HINTS = {
        "psikiyatrik", "antidepresan", "hipotiroidi", "euthyrox",
        "paroksetin", "fluoksetin", "sertralin", "okzasepin",
        "epilepsi", "nöbet", "troid", "tiroid",
        "internist", "kardiyoloji", "gastroenteroloji",
        "tansiyon", "hipertansiyon", "diyabet", "insülin",
        "hemoglobin", "hematoloji", "onkoloji",
        "patoloji", "biyopsi", "tomografi", "ultrason",
        "ekg", "ekg'de", "ekg'si",
        "anamnez", "muayene", "auskültasyon",
        "dispne", "taşikardi", "bradikardi",
        "röntgen", "mri", "manyetik",
        "poliklinik", "acil", "klinik",
        "göğüs", "akciğer", "bronş", "trakea",
        "plevra", "efüzyon", "pnömoni",
        "astım", "koah", "amfizem",
        "satürasyon", "oksijen", "spo2",
        "ateş", "nabız", "solunum",
    }

    check_word = (hyp_w or ref_w or "").lower().strip(".,;:!?")
    ref_check = (ref_w or "").lower().strip(".,;:!?")

    # VAD / ses başı sonu - ilk veya son pozisyon
    if idx < 3 or idx >= len(aligned) - 3:
        if ref_w is None or hyp_w is None:
            return "VAD_veya_ses_başı_sonu"

    # Tıbbi/özel isim kontrolü
    if ref_check in MEDICAL_HINTS or check_word in MEDICAL_HINTS:
        return "tıbbi_terim_veya_özel_isim"

    # Büyük harf (özel isim işareti - normalize öncesi)
    if ref_w and ref_w[0].isupper() and idx > 0:
        return "tıbbi_terim_veya_özel_isim"

    # Sadece ek farkı (normalizasyon)
    if ref_w and hyp_w:
        if ref_w[:4] == hyp_w[:4] and len(ref_w) > 4:
            return "normalizasyon_şüphesi"

    # Insertion - halüsinasyon şüphesi
    if ref_w is None:
        return "gerçek_ASR_hatası"

    # Deletion
    if hyp_w is None:
        return "gerçek_ASR_hatası"

    return "gerçek_ASR_hatası"


def get_context(aligned, idx, window=2):
    """Sol ve sağ bağlam kelimelerini döndür."""
    left = []
    for k in range(max(0, idx - window), idx):
        op, r, h = aligned[k]
        w = h if op in ("H", "I", "S") else f"[DEL:{r}]"
        left.append(w)
    right = []
    for k in range(idx + 1, min(len(aligned), idx + window + 1)):
        op, r, h = aligned[k]
        w = h if op in ("H", "I", "S") else f"[DEL:{r}]"
        right.append(w)
    return " ".join(left), " ".join(right)


def analyze_pair(row):
    """Bir satır için hizalama ve ilk 20 hata tablosunu üretir."""
    ref_raw = row["reference_text"].strip()
    hyp_raw = row["transcription"].strip()

    ref_norm = normalize_turkish_asr(ref_raw)
    hyp_norm = normalize_turkish_asr(hyp_raw)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    N = len(ref_words)
    M = len(hyp_words)
    aligned = compute_alignment(ref_words, hyp_words)

    hits = sum(1 for op, _, _ in aligned if op == "H")
    subs = sum(1 for op, _, _ in aligned if op == "S")
    dels = sum(1 for op, _, _ in aligned if op == "D")
    ins  = sum(1 for op, _, _ in aligned if op == "I")

    # Hata listesi (S/D/I pozisyonları)
    errors = [(i, op, r, h) for i, (op, r, h) in enumerate(aligned) if op != "H"]

    first20 = errors[:20]
    table_rows = []
    for rank, (idx, op, ref_w, hyp_w) in enumerate(first20, 1):
        sol, sag = get_context(aligned, idx)
        reason = classify_error(ref_w, hyp_w, aligned, idx)
        table_rows.append({
            "sira": rank,
            "hata_tipi": op,
            "referans_kelime": ref_w or "–",
            "model_kelime": hyp_w or "–",
            "sol_bagam": sol,
            "sag_bagam": sag,
            "muhtemel_neden": reason,
        })

    return {
        "model_id": row["model_id"],
        "basename": row["basename"],
        "condition": row["condition"],
        "N_ref": N,
        "M_hyp": M,
        "hits": hits,
        "subs": subs,
        "dels": dels,
        "ins": ins,
        "wer_csv": float(row.get("WER", 0) or 0),
        "wer_computed": round((subs + dels + ins) / N, 4) if N else 0,
        "first20_errors": table_rows,
    }


def load_rows():
    with open(CSV_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        r for r in rows
        if r["model_id"] in TARGET_MODELS
        and r["basename"] in TARGET_BASENAMES
        and r["condition"] == TARGET_CONDITION
        and r["run_purpose"] == "main_benchmark"
        and r["protocol_version"] == "pilot-v2-antiloop"
        and r.get("transcription", "").strip()
        and r.get("reference_text", "").strip()
    ]


def print_report(analysis):
    print(f"\n{'='*90}")
    print(f"MODEL: {analysis['model_id']}")
    print(f"KAYIT: {analysis['basename']}  KOŞUL: {analysis['condition']}")
    print(f"Ref kelime: {analysis['N_ref']}  Hyp kelime: {analysis['M_hyp']}")
    print(f"WER(CSV)={analysis['wer_csv']*100:.2f}%  WER(hesap)={analysis['wer_computed']*100:.2f}%")
    print(f"Hits={analysis['hits']}  S={analysis['subs']}  D={analysis['dels']}  I={analysis['ins']}")
    print(f"\n{'İLK 20 HATA':^90}")
    print(f"{'#':>3} {'Tip':>4} {'Referans':>22} {'Model':>22} {'Sol bağlam':<20} {'Sağ bağlam':<20} {'Neden'}")
    print("-" * 110)
    for r in analysis["first20_errors"]:
        print(f"{r['sira']:>3} {r['hata_tipi']:>4} {r['referans_kelime']:>22} {r['model_kelime']:>22} {r['sol_bagam']:<20} {r['sag_bagam']:<20} {r['muhtemel_neden']}")


def compare_two(a1, a2):
    """video_01 vs video_06 karşılaştırma - aynı model."""
    print(f"\n{'='*90}")
    print(f"KARŞILAŞTIRMA — MODEL: {a1['model_id']}")
    print(f"{'Metrik':<35} {'video_01':>15} {'video_06':>15}")
    print("-" * 70)
    metrics = [
        ("WER (CSV)", f"{a1['wer_csv']*100:.2f}%", f"{a2['wer_csv']*100:.2f}%"),
        ("Referans kelime sayısı", str(a1["N_ref"]), str(a2["N_ref"])),
        ("Hipotez kelime sayısı", str(a1["M_hyp"]), str(a2["M_hyp"])),
        ("Substitutions", str(a1["subs"]), str(a2["subs"])),
        ("Deletions", str(a1["dels"]), str(a2["dels"])),
        ("Insertions", str(a1["ins"]), str(a2["ins"])),
        ("Hits", str(a1["hits"]), str(a2["hits"])),
        ("S_rate", f"{a1['subs']/a1['N_ref']:.3f}", f"{a2['subs']/a2['N_ref']:.3f}"),
        ("D_rate", f"{a1['dels']/a1['N_ref']:.3f}", f"{a2['dels']/a2['N_ref']:.3f}"),
        ("I_rate", f"{a1['ins']/a1['N_ref']:.3f}", f"{a2['ins']/a2['N_ref']:.3f}"),
        ("Hyp/Ref oranı", f"{a1['M_hyp']/a1['N_ref']:.3f}", f"{a2['M_hyp']/a2['N_ref']:.3f}"),
    ]
    for label, v1, v2 in metrics:
        print(f"  {label:<33} {v1:>15} {v2:>15}")


def main():
    rows = load_rows()
    print(f"Analiz edilecek satır: {len(rows)}")
    if not rows:
        print("[HATA] Uygun satır bulunamadı.")
        return

    # Her kombinasyonu analiz et
    analyses = {}
    for r in rows:
        key = (r["model_id"], r["basename"])
        analyses[key] = analyze_pair(r)

    # Raporları yazdır
    for key in sorted(analyses.keys()):
        print_report(analyses[key])

    # video_01 vs video_06 karşılaştırma - her model için
    for model in TARGET_MODELS:
        k1 = (model, "video_01_genel_anamnez")
        k2 = (model, "video_06_gogus_hastaliklari_oyku_alma")
        if k1 in analyses and k2 in analyses:
            compare_two(analyses[k1], analyses[k2])


if __name__ == "__main__":
    main()
