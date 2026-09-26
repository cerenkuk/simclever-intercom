import os
import sys
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
REF_QUALITY_CSV = os.path.join(_pkg_root, "outputs", "metrics", "reference_quality_review.csv")

def close_ref_quality_review():
    records = [
        {
            "basename": "saha_03_img_3602",
            "suspected_text": "çek",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 2,
            "issue_type": "reference_token_presence_verified",
            "review_status": "resolved_no_reference_change",
            "resolution": "existing_reference_contains_two_verified_tokens",
            "reference_review_required": "false",
            "medical_term_review_required": "false",
            "proposed_correction": "Değişiklik yapılmadı. Mevcut referans 1. satırda 'Çek,çek.' olarak 2 kelimeyi içermektedir.",
            "approved_by": "human_reviewer",
            "applied": "false",
            "notes": "İnsan dinlemesiyle referans kelime içeriğinin %100 uyumlu olduğu doğrulandı."
        },
        {
            "basename": "saha_02_img_3492",
            "suspected_text": "SVT",
            "audio_occurrence_count": 2,
            "reference_occurrence_count": 2,
            "issue_type": "medical_acronym_format_verification",
            "review_status": "resolved_no_reference_change",
            "resolution": "existing_reference_contains_two_verified_svt_tokens",
            "reference_review_required": "false",
            "medical_term_review_required": "true",
            "proposed_correction": "Değişiklik yapılmadı. Referansta SVT 2 kez mevcuttur. Modelin S R C üretmesi substitution hatası olarak kalacaktır.",
            "approved_by": "human_reviewer",
            "applied": "false",
            "notes": "SVT -> S R C ikamesi tıbbi terim analizinde izlenmek üzere medical_term_review_required=true olarak bırakıldı."
        }
    ]

    fieldnames = [
        "basename", "suspected_text", "audio_occurrence_count", "reference_occurrence_count",
        "issue_type", "review_status", "resolution", "reference_review_required", "medical_term_review_required",
        "proposed_correction", "approved_by", "applied", "notes"
    ]

    with open(REF_QUALITY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Closed {REF_QUALITY_CSV} with review_status=resolved_no_reference_change.")

if __name__ == "__main__":
    close_ref_quality_review()
