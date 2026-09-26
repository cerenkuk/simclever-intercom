import os
import sys
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
REF_QUALITY_CSV = os.path.join(_pkg_root, "outputs", "metrics", "reference_quality_review.csv")

def update_ref_quality_status():
    records = [
        {
            "basename": "saha_03_img_3602",
            "suspected_text": "çek",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 2, # Note: Current reference already has 'Çek,çek.' at line 1!
            "issue_type": "reference_token_presence_verified",
            "review_status": "proposal_prepared",
            "proposed_correction": "Referans TXT başındaki 'Çek,çek.' biçimi 'Çek çek.' olarak noktalama temizliğiyle korunabilir. Ek token eklenmesine gerek yoktur.",
            "approved_by": "",
            "applied": "false",
            "notes": "Referans TXT dosyasında 'Çek,çek.' kelimeleri 1. satırda bulunmaktadır. Referansta eksik olmadığı doğrulanmıştır."
        },
        {
            "basename": "saha_02_img_3492",
            "suspected_text": "SVT",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 2,
            "issue_type": "medical_acronym_format_verification",
            "review_status": "proposal_prepared",
            "proposed_correction": "Referanstaki 1. satırda geçen 'SVT' ve 'SVT taşikardi' biçimleri doğru kanoniktir. Modelin 'S R C' üretmesi Substitution hatası olarak kalmalıdır.",
            "approved_by": "",
            "applied": "false",
            "notes": "Referans TXT dosyasında SVT 2 kez mevcuttur. Model harf harf S R C üretmiştir."
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

    print(f"Updated {REF_QUALITY_CSV} with review_status=proposal_prepared.")

if __name__ == "__main__":
    update_ref_quality_status()
