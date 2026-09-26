import os
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
METRICS_DIR = os.path.join(_pkg_root, "outputs", "metrics")
PRESCREEN_CSV = os.path.join(METRICS_DIR, "candidate_prescreen_results.csv")
INVALID_CSV = os.path.join(METRICS_DIR, "candidate_prescreen_invalid_results.csv")

def isolate_invalid_video04_result():
    if not os.path.exists(PRESCREEN_CSV):
        print(f"CSV bulunamadı: {PRESCREEN_CSV}")
        return

    valid_rows = []
    invalid_rows = []

    with open(PRESCREEN_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if (
                r.get("model_id") == "qwen3-asr-0.6b-hf"
                and r.get("basename") == "video_04_simule_hasta_gorusmesi"
                and r.get("condition") == "none"
                and r.get("run_purpose") == "candidate_prescreen"
            ):
                invalid_row = dict(r)
                invalid_row["result_status"] = "invalid_truncated_output"
                invalid_row["invalid_reason"] = "max_new_tokens_512_output_truncated_mid_transcript"
                invalid_row["superseded_by"] = ""
                invalid_rows.append(invalid_row)
            else:
                valid_rows.append(r)

    if not invalid_rows:
        print("İzole edilecek geçersiz video_04 satırı bulunamadı.")
        return

    # 1. Invalid CSV'ye Ekle / Oluştur
    invalid_fieldnames = list(fieldnames)
    for extra in ["result_status", "invalid_reason", "superseded_by"]:
        if extra not in invalid_fieldnames:
            invalid_fieldnames.append(extra)

    invalid_exists = os.path.exists(INVALID_CSV)
    with open(INVALID_CSV, "a" if invalid_exists else "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=invalid_fieldnames)
        if not invalid_exists:
            writer.writeheader()
        writer.writerows(invalid_rows)

    print(f"Geçersiz satır '{INVALID_CSV}' dosyasına aktarıldı: {len(invalid_rows)} adet.")

    # 2. Ana candidate_prescreen_results.csv Dosyasını Temizle
    with open(PRESCREEN_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)

    print(f"Ana '{PRESCREEN_CSV}' dosyası güncellendi (kalan geçerli satır: {len(valid_rows)}).")

if __name__ == "__main__":
    isolate_invalid_video04_result()
