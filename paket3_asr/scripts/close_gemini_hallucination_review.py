import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
METRICS_DIR = os.path.join(PKG_ROOT, "outputs", "metrics")

ANALYSIS_CSV = os.path.join(
    METRICS_DIR, "gemini_hallucination_analysis_v3_results.csv"
)
MANUAL_CSV = os.path.join(
    METRICS_DIR, "gemini_hallucination_manual_review.csv"
)

DECISIONS = {
    "video_05_poliklinik_anamnez_fizik_muayene": {
        "detected_candidate": "'kırk kırk bir' (x8)",
        "manual_verdict": "confirmed_real_speech_repetition",
        "final_loop_detected": "HAYIR",
        "final_loop_severity": "none",
        "evidence_source": "human_reference_and_transcript_context",
        "reference_evidence": (
            "İnsan referansında 'kırk kırk bir' ifadesi sekiz kez "
            "ardışık yer almaktadır."
        ),
        "context": (
            "Vibrasyon torasik muayenesi sırasında hastadan "
            "'kırk kırk bir' ifadesini tekrarlaması istenmektedir."
        ),
        "decision_reason": (
            "Model döngüsü değildir; ses ve insan referansındaki "
            "gerçek klinik muayene tekrarıdır."
        ),
    },
    "video_06_gogus_hastaliklari_oyku_alma": {
        "detected_candidate": "'böyle yapınca ağrıyor' (x4)",
        "manual_verdict": "confirmed_real_speech_repetition",
        "final_loop_detected": "HAYIR",
        "final_loop_severity": "none",
        "evidence_source": "human_reference_and_transcript_context",
        "reference_evidence": (
            "İnsan referansında 'böyle yapınca ağrıyor' ifadesi "
            "dört kez ardışık yer almaktadır."
        ),
        "context": (
            "Konuşmacının kırık parmakla ilgili klinik fıkrayı "
            "aktarırken söylediği gerçek tekrardır."
        ),
        "decision_reason": (
            "Model döngüsü değildir; ses ve insan referansındaki "
            "gerçek anlatı tekrarıdır."
        ),
    },
}

MANUAL_FIELDS = [
    "model_id",
    "basename",
    "detected_candidate",
    "automatic_severity",
    "manual_review_status",
    "manual_verdict",
    "final_loop_detected",
    "final_loop_severity",
    "evidence_source",
    "reference_evidence",
    "context",
    "decision_reason",
]

def main():
    if not os.path.isfile(ANALYSIS_CSV):
        raise FileNotFoundError(ANALYSIS_CSV)

    with open(
        ANALYSIS_CSV, "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if len(rows) != 6:
        raise RuntimeError(
            f"GEMINI_V3_ROW_COUNT_ERROR: expected=6 actual={len(rows)}"
        )

    basenames = {row["basename"] for row in rows}
    missing = set(DECISIONS) - basenames
    if missing:
        raise RuntimeError(
            f"MANUAL_REVIEW_TARGET_MISSING: {sorted(missing)}"
        )

    manual_rows = []

    for row in rows:
        basename = row["basename"]
        if basename not in DECISIONS:
            continue

        decision = DECISIONS[basename]

        if row.get("manual_review_status") != "pending":
            raise RuntimeError(
                f"EXPECTED_PENDING_REVIEW: {basename}"
            )

        manual_rows.append({
            "model_id": row["model_id"],
            "basename": basename,
            "detected_candidate":
                decision["detected_candidate"],
            "automatic_severity": row["loop_severity"],
            "manual_review_status": "completed",
            "manual_verdict": decision["manual_verdict"],
            "final_loop_detected":
                decision["final_loop_detected"],
            "final_loop_severity":
                decision["final_loop_severity"],
            "evidence_source": decision["evidence_source"],
            "reference_evidence":
                decision["reference_evidence"],
            "context": decision["context"],
            "decision_reason": decision["decision_reason"],
        })

        row["loop_detected"] = decision["final_loop_detected"]
        row["loop_severity"] = decision["final_loop_severity"]
        row["loop_detection_reason"] = (
            "CONFIRMED_REAL_SPEECH_REPETITION: "
            + decision["decision_reason"]
        )
        row["manual_review_status"] = "completed"
        row["manual_verdict"] = decision["manual_verdict"]

    if len(manual_rows) != 2:
        raise RuntimeError(
            f"MANUAL_REVIEW_COUNT_ERROR: expected=2 "
            f"actual={len(manual_rows)}"
        )

    manual_tmp = MANUAL_CSV + ".tmp"
    with open(
        manual_tmp, "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANUAL_FIELDS
        )
        writer.writeheader()
        writer.writerows(manual_rows)
    os.replace(manual_tmp, MANUAL_CSV)

    analysis_tmp = ANALYSIS_CSV + ".tmp"
    with open(
        analysis_tmp, "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(analysis_tmp, ANALYSIS_CSV)

    pathological = sum(
        row["loop_severity"] == "pathological"
        for row in rows
    )
    warnings = sum(
        row["loop_severity"] == "warning"
        for row in rows
    )
    pending = sum(
        row["manual_review_status"] == "pending"
        for row in rows
    )
    loops = sum(
        row["loop_detected"] == "EVET"
        for row in rows
    )

    if pathological or warnings or pending or loops:
        raise RuntimeError(
            "GEMINI_V3_CLOSURE_FAILED: "
            f"loops={loops}, pathological={pathological}, "
            f"warnings={warnings}, pending={pending}"
        )

    print("GEMINI V3 KAPANIS: 6 / 6")
    print("INSAN INCELEMESI TAMAMLANAN: 2 / 2")
    print("DOGRULANMIS PATOLOJIK DONGU: 0")
    print("WARNING: 0")
    print("BEKLEYEN MANUEL INCELEME: 0")
    print(f"MANUEL KANIT: {MANUAL_CSV}")
    print(f"NIHAI ANALIZ: {ANALYSIS_CSV}")

if __name__ == "__main__":
    main()
