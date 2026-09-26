import os
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
MANUAL_CSV = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_manual_review.csv")
V3_CSV = os.path.join(_pkg_root, "outputs", "metrics", "hallucination_analysis_v3_results.csv")

def create_manual_review_csv():
    reviews = [
        {
            "review_id": "REV-001",
            "basename": "video_02_psikiyatrik_anamnez",
            "model_id": "faster-whisper-large-v3-turbo-int8",
            "condition": "none",
            "suspected_token": "hayır",
            "actual_audio_occurrence_count": 5,
            "hypothesis_consecutive_repeat_count": 112,
            "manual_verdict": "CONFIRMED_PATHOLOGICAL_ASR_LOOP",
            "reviewed_by": "human_reviewer",
            "review_method": "full_audio_manual_listening",
            "review_scope": "full_record",
            "notes": "Model çıktısındaki aşırı tekrar gerçek konuşmada bulunmamaktadır."
        },
        {
            "review_id": "REV-002",
            "basename": "video_04_simule_hasta_gorusmesi",
            "model_id": "faster-whisper-large-v3-turbo-int8",
            "condition": "none",
            "suspected_token": "herhalde",
            "actual_audio_occurrence_count": 1,
            "hypothesis_consecutive_repeat_count": 73,
            "manual_verdict": "CONFIRMED_PATHOLOGICAL_ASR_LOOP",
            "reviewed_by": "human_reviewer",
            "review_method": "full_audio_manual_listening",
            "review_scope": "full_record",
            "notes": "Gerçek tek kullanım, model tarafından patolojik ardışık tekrar dizisine dönüştürülmüştür."
        },
        {
            "review_id": "REV-003",
            "basename": "saha_03_img_3602",
            "model_id": "faster-whisper-large-v3-turbo-int8",
            "condition": "none",
            "suspected_token": "çek",
            "actual_audio_occurrence_count": 0,
            "hypothesis_consecutive_repeat_count": 74,
            "manual_verdict": "CONFIRMED_HALLUCINATED_TOKEN_LOOP",
            "reviewed_by": "human_reviewer",
            "review_method": "full_audio_manual_listening",
            "review_scope": "full_record",
            "notes": "çek kelimesi kaynak seste hiç bulunmadığı halde model tarafından tekrar döngüsü olarak üretilmiştir."
        }
    ]

    fieldnames = [
        "review_id", "basename", "model_id", "condition", "suspected_token",
        "actual_audio_occurrence_count", "hypothesis_consecutive_repeat_count",
        "manual_verdict", "reviewed_by", "review_method", "review_scope", "notes"
    ]

    with open(MANUAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Created manual review CSV: {MANUAL_CSV} with 3 entries.")

def update_v3_csv_with_manual_verdicts():
    if not os.path.exists(V3_CSV):
        print("V3 CSV not found!")
        return

    with open(V3_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Read manual review verdicts
    manual_map = {
        ("video_02_psikiyatrik_anamnez", "faster-whisper-large-v3-turbo-int8", "none"): ("completed", "confirmed_pathological_loop"),
        ("video_04_simule_hasta_gorusmesi", "faster-whisper-large-v3-turbo-int8", "none"): ("completed", "confirmed_pathological_loop"),
        ("saha_03_img_3602", "faster-whisper-large-v3-turbo-int8", "none"): ("completed", "confirmed_pathological_loop")
    }

    for r in rows:
        key = (r["basename"], r["model_id"], r["condition"])
        if key in manual_map:
            r["manual_review_status"] = manual_map[key][0]
            r["manual_verdict"] = manual_map[key][1]
        else:
            if r.get("loop_detected") == "EVET" and r.get("loop_severity") == "pathological":
                r["manual_review_status"] = "pending_human_review"
                r["manual_verdict"] = "unverified_pathological_candidate"
            elif r.get("loop_detected") == "EVET" and r.get("loop_severity") == "warning":
                r["manual_review_status"] = "pending_human_review"
                r["manual_verdict"] = "unverified_warning_candidate"
            else:
                r["manual_review_status"] = "not_required"
                r["manual_verdict"] = "normal_transcription"

    fieldnames = list(rows[0].keys())
    if "manual_verdict" not in fieldnames:
        fieldnames.append("manual_verdict")

    with open(V3_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {V3_CSV} with manual_review_status and manual_verdict fields.")

if __name__ == "__main__":
    create_manual_review_csv()
    update_v3_csv_with_manual_verdicts()
