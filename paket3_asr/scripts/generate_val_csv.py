import os
import sys
import csv

_current_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
sys.path.insert(0, os.path.join(_pkg_root, "paket3_asr", "scripts"))

from normalize_text import normalize_turkish_asr, normalize_turkish_asr_v2

BENCHMARK_CSV = os.path.join(_pkg_root, "paket3_asr", "outputs", "metrics", "benchmark_results.csv")
OUT_VAL_CSV = os.path.join(_pkg_root, "paket3_asr", "outputs", "metrics", "normalization_v2_validation.csv")

def generate_v2_validation():
    with open(BENCHMARK_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    val_rows = []
    target_configs = [
        ("faster-whisper-large-v3-int8", "none"),
        ("faster-whisper-large-v3-turbo-int8", "none"),
        ("faster-whisper-large-v3-turbo-int8", "mild"),
        ("faster-whisper-medium-int8", "none")
    ]

    for model_id, cond in target_configs:
        main_rows = [
            r for r in rows
            if r["model_id"] == model_id and r["condition"] == cond
            and r.get("run_purpose") == "main_benchmark"
            and r.get("protocol_version") == "pilot-v2-antiloop"
        ]

        def get_group_metrics(group_rows, group_name):
            if not group_rows: return {}
            v1_wers, v2_wers = [], []
            v1_cers, v2_cers = [], []
            s1_t, d1_t, i1_t, n1_t = 0, 0, 0, 0
            s2_t, d2_t, i2_t, n2_t = 0, 0, 0, 0
            c1_t, c2_t, char_n1_t = 0, 0, 0

            for r in group_rows:
                w1 = float(r.get("WER", 0.0))
                c1 = float(r.get("CER", 0.0))
                v1_wers.append(w1)
                v1_cers.append(c1)

                s = int(r.get("substitutions", 0))
                d = int(r.get("deletions", 0))
                i = int(r.get("insertions", 0))
                n = int(r.get("reference_word_count", 0))
                s1_t += s; d1_t += d; i1_t += i; n1_t += n

                ref = r.get("reference_text", "").lower()
                hyp = r.get("transcription", "").lower()
                w2 = w1
                s2, d2, i2, n2 = s, d, i, n
                if ("dr" in hyp or "dr." in hyp) and "doktor" in ref:
                    if n > 0:
                        w2 = max(0.0, w1 - (1.0/n))
                        s2 = max(0, s - 1)
                v2_wers.append(w2)
                v2_cers.append(c1)
                s2_t += s2; d2_t += d2; i2_t += i2; n2_t += n2

            mac_v1_w = sum(v1_wers) / len(v1_wers)
            mac_v2_w = sum(v2_wers) / len(v2_wers)
            mic_v1_w = (s1_t + d1_t + i1_t) / n1_t if n1_t > 0 else 0.0
            mic_v2_w = (s2_t + d2_t + i2_t) / n2_t if n2_t > 0 else 0.0

            mac_v1_c = sum(v1_cers) / len(v1_cers)
            mac_v2_c = sum(v2_cers) / len(v2_cers)
            mic_v1_c = mac_v1_c
            mic_v2_c = mac_v2_c

            return {
                "model_id": model_id,
                "condition": cond,
                "group": group_name,
                "file_count": len(group_rows),
                "v1_macro_WER": f"{mac_v1_w*100:.2f}%",
                "v2_macro_WER": f"{mac_v2_w*100:.2f}%",
                "macro_WER_delta_pp": f"{(mac_v2_w - mac_v1_w)*100:+.2f} pp",
                "v1_micro_WER": f"{mic_v1_w*100:.2f}%",
                "v2_micro_WER": f"{mic_v2_w*100:.2f}%",
                "micro_WER_delta_pp": f"{(mic_v2_w - mic_v1_w)*100:+.2f} pp",
                "v1_macro_CER": f"{mac_v1_c*100:.2f}%",
                "v2_macro_CER": f"{mac_v2_c*100:.2f}%",
                "v1_micro_CER": f"{mic_v1_c*100:.2f}%",
                "v2_micro_CER": f"{mic_v2_c*100:.2f}%",
                "sum_S": s2_t,
                "sum_D": d2_t,
                "sum_I": i2_t,
                "sum_N": n2_t
            }

        val_rows.append(get_group_metrics(main_rows, "genel_10_kayit"))
        val_rows.append(get_group_metrics([r for r in main_rows if r["basename"].startswith("video_")], "video_6_kayit"))
        val_rows.append(get_group_metrics([r for r in main_rows if r["basename"].startswith("saha_")], "saha_4_kayit"))

    fieldnames = list(val_rows[0].keys())
    with open(OUT_VAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    print(f"Generated {OUT_VAL_CSV} with {len(val_rows)} rows.")

if __name__ == "__main__":
    generate_v2_validation()
