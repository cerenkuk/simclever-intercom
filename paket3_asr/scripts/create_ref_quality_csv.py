import os
import sys
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
REF_QUALITY_CSV = os.path.join(_pkg_root, "outputs", "metrics", "reference_quality_review.csv")

def create_reference_quality_review_csv():
    records = [
        {
            "basename": "saha_03_img_3602",
            "suspected_text": "çek",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 0,
            "issue_type": "missing_token_in_ground_truth_reference",
            "review_status": "pending_reference_correction",
            "proposed_correction": "Referans TXT içine duyulan 2 adet 'çek' kelimesinin eklenmesi değerlendirilsin.",
            "approved_by": "",
            "applied": "false",
            "notes": "Seste 2 kez çek duyuluyor, model 74 üretti, referansta 0 kelime var."
        },
        {
            "basename": "saha_02_img_3492",
            "suspected_text": "SVT",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 2,
            "issue_type": "medical_acronym_format_verification",
            "review_status": "pending_reference_correction",
            "proposed_correction": "Referanstaki 'SVT' yazım biçiminin model çıktısındaki 'S R C' ile karşılaştırması doğrulansın.",
            "approved_by": "",
            "applied": "false",
            "notes": "Seste SVT duyuldu, model S R C harflemiştir."
        }
    ]

    fieldnames = [
        "basename", "suspected_text", "audio_occurrence_count", "reference_occurrence_count",
        "issue_type", "review_status", "proposed_correction", "approved_by", "applied", "notes"
    ]

    with open(REF_QUALITY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Created reference quality review CSV: {REF_QUALITY_CSV} with 2 records.")

if __name__ == "__main__":
    create_reference_quality_review_csv()
