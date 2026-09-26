"""
Paket 3 - Aşama 5: Halüsinasyon Kesit Analizi
Mevcut transkripsiyonlardan, model yeniden çalıştırılmadan:
- insertion sayısı ve oranı
- art arda tekrar eden kelime sayısı
- tekrar eden 2-gram, 3-gram, 4-gram
- en uzun art arda tekrar zinciri
- hipotez/referans kelime oranı
- şüpheli tekrar örnekleri

Çıktı: paket3_asr/outputs/metrics/hallucination_analysis_results.csv

NOT: Zaman damgası olmadığı için gerçek sessizlik kesiti tespiti yapılamaz.
Yalnız metin tabanlı tekrar/halüsinasyon analizi yapılmaktadır.
"""

import csv
import os
import sys
from collections import Counter

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_script_dir)
from normalize_text import normalize_turkish_asr

_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
CSV_IN  = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
CSV_OUT = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_results.csv")


def count_consecutive_repeats(words):
    """
    Art arda aynı kelimenin tekrar sayısını ve en uzun zinciri bulur.
    """
    if not words:
        return 0, 0, []
    total_repeats = 0
    max_chain = 1
    chains = []
    current_word = words[0]
    current_count = 1
    for w in words[1:]:
        if w == current_word:
            current_count += 1
        else:
            if current_count > 1:
                total_repeats += current_count - 1
                chains.append((current_word, current_count))
                if current_count > max_chain:
                    max_chain = current_count
            current_word = w
            current_count = 1
    if current_count > 1:
        total_repeats += current_count - 1
        chains.append((current_word, current_count))
        if current_count > max_chain:
            max_chain = current_count
    if max_chain == 1:
        max_chain = 0
    return total_repeats, max_chain, chains[:5]  # en fazla 5 örnek


def count_ngram_repeats(words, n):
    """n-gram tekrar sayısını hesaplar (aynı n-gram 2+ defa geçiyorsa)."""
    if len(words) < n:
        return 0, []
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = [(ng, cnt) for ng, cnt in counts.items() if cnt > 1]
    repeated.sort(key=lambda x: -x[1])
    return len(repeated), repeated[:3]


def compute_alignment_sdi(ref_words, hyp_words):
    """S/D/I hesaplamak için basit Levenshtein."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    # Sadece sayılar isteniyor; geri izleme yapmıyoruz
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


def analyze_row(row):
    ref_raw = row.get("reference_text", "").strip()
    hyp_raw = row.get("transcription", "").strip()
    if not hyp_raw:
        return None

    ref_norm = normalize_turkish_asr(ref_raw) if ref_raw else ""
    hyp_norm = normalize_turkish_asr(hyp_raw)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split()

    N = len(ref_words)
    M = len(hyp_words)

    # S/D/I
    if N > 0:
        subs, dels, ins = compute_alignment_sdi(ref_words, hyp_words)
        insertion_rate = round(ins / N, 4)
    else:
        subs = dels = ins = 0
        insertion_rate = 0.0

    # Art arda tekrar
    consec_count, max_chain_len, chain_examples = count_consecutive_repeats(hyp_words)

    # N-gram tekrarlar
    bigram_rep_count, bigram_ex = count_ngram_repeats(hyp_words, 2)
    trigram_rep_count, trigram_ex = count_ngram_repeats(hyp_words, 3)
    fourgram_rep_count, fourgram_ex = count_ngram_repeats(hyp_words, 4)

    # Hyp/Ref oranı
    hyp_ref_ratio = round(M / N, 4) if N > 0 else 0.0

    # Şüpheli tekrar örnekleri (string)
    suspicious_examples = "; ".join(
        [f"'{' '.join(ng)}' x{cnt}" for ng, cnt in trigram_ex[:2]] +
        [f"consec:'{w}'x{c}" for w, c in chain_examples[:2]]
    )

    return {
        "model_id": row["model_id"],
        "basename": row["basename"],
        "condition": row["condition"],
        "protocol_version": row["protocol_version"],
        "run_purpose": row["run_purpose"],
        "reference_word_count": N,
        "hypothesis_word_count": M,
        "insertions": ins,
        "insertion_rate": insertion_rate,
        "consecutive_repeat_count": consec_count,
        "max_consecutive_chain": max_chain_len,
        "bigram_repeat_types": bigram_rep_count,
        "trigram_repeat_types": trigram_rep_count,
        "fourgram_repeat_types": fourgram_rep_count,
        "hyp_ref_ratio": hyp_ref_ratio,
        "suspicious_examples": suspicious_examples,
        "note": ""
    }


def main():
    with open(CSV_IN, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # main_benchmark, pilot-v2-antiloop ve transcription mevcut
    target_rows = [
        r for r in all_rows
        if r["run_purpose"] == "main_benchmark"
        and r["protocol_version"] == "pilot-v2-antiloop"
        and r.get("transcription", "").strip()
    ]

    # Mükerrer koruma
    existing_keys = set()
    existing_results = []
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
                existing_keys.add(key)
                existing_results.append(r)

    results = []
    for row in target_rows:
        key = (row["model_id"], row["basename"], row["condition"], row["protocol_version"], row["run_purpose"])
        if key in existing_keys:
            continue
        result = analyze_row(row)
        if result:
            results.append(result)

    all_results = existing_results + results

    fieldnames = [
        "model_id", "basename", "condition", "protocol_version", "run_purpose",
        "reference_word_count", "hypothesis_word_count",
        "insertions", "insertion_rate",
        "consecutive_repeat_count", "max_consecutive_chain",
        "bigram_repeat_types", "trigram_repeat_types", "fourgram_repeat_types",
        "hyp_ref_ratio", "suspicious_examples", "note"
    ]

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"=== HALÜSİNASYON ANALİZİ TAMAMLANDI ===")
    print(f"Yeni satır: {len(results)}")
    print(f"Toplam: {len(all_results)}")
    print(f"Çıktı: {CSV_OUT}")
    print()

    # Özet tablo
    print(f"{'model_id':<45} {'basename':<45} {'cond':<12} {'ins_rate':>9} {'consec':>7} {'tri_rep':>8} {'h/r_ratio':>10}")
    print("-" * 145)
    for r in sorted(all_results, key=lambda x: (x["model_id"], x["basename"], x["condition"])):
        print(f"{r['model_id']:<45} {r['basename']:<45} {r['condition']:<12} {float(r.get('insertion_rate',0)):>9.3f} {r.get('consecutive_repeat_count','?'):>7} {r.get('trigram_repeat_types','?'):>8} {float(r.get('hyp_ref_ratio',0)):>10.3f}")

    # Özellikle small + video_01 karşılaştırması
    print("\n=== SMALL + VIDEO_01 PILOT-V2-ANTİLOOP ===")
    small_v01 = [r for r in all_results if r["model_id"] == "faster-whisper-small-int8" and r["basename"] == "video_01_genel_anamnez"]
    if small_v01:
        for r in small_v01:
            print(f"  condition={r['condition']}  insertion_rate={r.get('insertion_rate')}  consec={r.get('consecutive_repeat_count')}  tri_rep={r.get('trigram_repeat_types')}  hyp/ref={r.get('hyp_ref_ratio')}")
    else:
        print("  [BİLGİ] small + video_01 pilot-v2-antiloop satırı bulunamadı.")
        print("  [NOT] pilot-v1 ham kayıtları bu CSV'de yer almamaktadır.")

    # Özellikle tıbbi cihaz sesi içeren kayıtlar
    print("\n=== TIBBI CİHAZ / KISIK KONUŞMA KAYITLARI ===")
    special = ["video_02_psikiyatrik_anamnez", "saha_01_img_3463", "saha_02_img_3492", "saha_03_img_3602", "saha_04_vid_20220913_142911"]
    for r in sorted(all_results, key=lambda x: (x["basename"], x["model_id"], x["condition"])):
        if r["basename"] in special:
            print(f"  {r['basename']:<45} {r['model_id']:<42} {r['condition']:<12} ins_rate={float(r.get('insertion_rate',0)):.3f}  consec={r.get('consecutive_repeat_count')}  hyp/ref={r.get('hyp_ref_ratio')}")


if __name__ == "__main__":
    main()
