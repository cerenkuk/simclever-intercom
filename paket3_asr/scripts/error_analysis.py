"""
Paket 3 - Aşama 2: S/D/I Hata Analizi
Tüm main_benchmark transcription/reference çiftlerinden geriye dönük S/D/I hesaplar.
ASR modeli yeniden çalıştırılmaz.
Çıktı: paket3_asr/outputs/metrics/error_analysis_results.csv
"""

import csv
import os
import sys

# normalize_text modülü scripts/ klasöründe
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_script_dir)
from normalize_text import normalize_turkish_asr

_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
CSV_IN  = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
CSV_OUT = os.path.join(_pkg_root, "outputs", "metrics", "error_analysis_results.csv")

# --- Levenshtein alignment (S/D/I) ---
def compute_alignment(ref_tokens, hyp_tokens):
    """
    Standart Levenshtein DP ile S/D/I ve hit sayısını hesaplar.
    Geri izleme (backtrace) ile gerçek hizalamayı çıkarır.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    # DP tablosu: (cost, ops)
    INF = 10**9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],   # substitution
                    dp[i - 1][j],        # deletion
                    dp[i][j - 1]         # insertion
                )

    # Geri izleme
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

    hits = sum(1 for op, _, _ in aligned if op == "H")
    subs = sum(1 for op, _, _ in aligned if op == "S")
    dels = sum(1 for op, _, _ in aligned if op == "D")
    ins  = sum(1 for op, _, _ in aligned if op == "I")
    return hits, subs, dels, ins, aligned


def dominant_error_type(subs, dels, ins):
    counts = {"substitution": subs, "deletion": dels, "insertion": ins}
    max_type = max(counts, key=counts.get)
    max_val = counts[max_type]
    if max_val == 0:
        return "none"
    return max_type


def wer_check(subs, dels, ins, ref_n):
    if ref_n == 0:
        return 0.0
    return round((subs + dels + ins) / ref_n, 4)


def load_main_benchmark():
    with open(CSV_IN, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    main = [
        r for r in rows
        if r["run_purpose"] == "main_benchmark"
        and r["protocol_version"] == "pilot-v2-antiloop"
        and r.get("transcription", "").strip()
        and r.get("reference_text", "").strip()
    ]
    # Mükerrer kontrolü
    seen = set()
    unique = []
    for r in main:
        key = (r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def run_error_analysis():
    rows = load_main_benchmark()
    print(f"Analiz edilecek benzersiz satır sayısı: {len(rows)}")

    # Mevcut çıktıyı oku (mükerrer koruma)
    existing_keys = set()
    existing_rows = []
    if os.path.exists(CSV_OUT):
        with open(CSV_OUT, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
                existing_keys.add(key)
                existing_rows.append(r)
        print(f"Mevcut çıktı satırı: {len(existing_rows)}")

    results = []
    wer_mismatches = []

    for r in rows:
        key = (r["model_id"], r["basename"], r["condition"], r["protocol_version"], r["run_purpose"])
        if key in existing_keys:
            print(f"  [ATLANDI - mevcut] {key}")
            continue

        ref_raw = r["reference_text"].strip()
        hyp_raw = r["transcription"].strip()

        ref_norm = normalize_turkish_asr(ref_raw)
        hyp_norm = normalize_turkish_asr(hyp_raw)

        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        N = len(ref_words)
        M = len(hyp_words)

        if N == 0:
            print(f"  [UYARI] Referans boş: {key}")
            continue

        hits, subs, dels, ins, aligned = compute_alignment(ref_words, hyp_words)

        computed_wer = wer_check(subs, dels, ins, N)
        csv_wer = float(r.get("WER", 0) or 0)
        wer_diff = abs(computed_wer - csv_wer)
        if wer_diff > 0.01:
            wer_mismatches.append({
                "key": key,
                "csv_wer": csv_wer,
                "computed_wer": computed_wer,
                "diff": wer_diff
            })
            print(f"  [WER UYUMSUZLUK] {key}: CSV={csv_wer:.4f} Hesaplanan={computed_wer:.4f} Fark={wer_diff:.4f}")

        dom_type = dominant_error_type(subs, dels, ins)

        results.append({
            "model_id": r["model_id"],
            "basename": r["basename"],
            "condition": r["condition"],
            "protocol_version": r["protocol_version"],
            "run_purpose": r["run_purpose"],
            "hits": hits,
            "substitutions": subs,
            "deletions": dels,
            "insertions": ins,
            "reference_word_count": N,
            "hypothesis_word_count": M,
            "substitution_rate": round(subs / N, 4),
            "deletion_rate": round(dels / N, 4),
            "insertion_rate": round(ins / N, 4),
            "dominant_error_type": dom_type,
            "computed_wer": computed_wer,
            "csv_wer": csv_wer,
            "wer_match": "OK" if wer_diff <= 0.01 else f"MISMATCH_diff={wer_diff:.4f}",
        })

    all_results = existing_rows + results
    fieldnames = [
        "model_id", "basename", "condition", "protocol_version", "run_purpose",
        "hits", "substitutions", "deletions", "insertions",
        "reference_word_count", "hypothesis_word_count",
        "substitution_rate", "deletion_rate", "insertion_rate",
        "dominant_error_type", "computed_wer", "csv_wer", "wer_match"
    ]
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n=== S/D/I ANALİZ TAMAMLANDI ===")
    print(f"Yeni satır: {len(results)}")
    print(f"Toplam çıktı: {len(all_results)}")
    print(f"WER uyumsuzluk: {len(wer_mismatches)}")
    if wer_mismatches:
        print("  UYUMSUZ SATIRLAR:")
        for m in wer_mismatches:
            print(f"    {m}")
    print(f"Çıktı: {CSV_OUT}")

    # Özet - model bazlı dominant error
    print("\n=== MODEL BAZLI BASKILILIK HATASİ ===")
    from collections import Counter
    for model_id in sorted(set(r.get("model_id","") for r in all_results)):
        model_rows = [r for r in all_results if r.get("model_id") == model_id]
        dom_count = Counter(r.get("dominant_error_type","") for r in model_rows)
        dom_s = sum(int(r.get("substitutions",0) or 0) for r in model_rows)
        dom_d = sum(int(r.get("deletions",0) or 0) for r in model_rows)
        dom_i = sum(int(r.get("insertions",0) or 0) for r in model_rows)
        dom_n = sum(int(r.get("reference_word_count",0) or 0) for r in model_rows)
        micro_wer = (dom_s + dom_d + dom_i) / dom_n if dom_n else 0
        print(f"  {model_id}: S={dom_s} D={dom_d} I={dom_i} N={dom_n} micro_WER={micro_wer:.3f} dom={dom_count.most_common(1)[0][0] if dom_count else 'N/A'}")


if __name__ == "__main__":
    run_error_analysis()
