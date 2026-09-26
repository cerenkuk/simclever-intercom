import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from hallucination_analysis_v3 import analyze_repetition_v3, classify_loop_v3
from normalize_text import normalize_turkish_asr

INPUT_CSV = os.path.join(
    PKG_ROOT, "outputs", "metrics", "commercial_api_results.csv"
)
OUTPUT_CSV = os.path.join(
    PKG_ROOT, "outputs", "metrics",
    "gemini_hallucination_analysis_v3_results.csv"
)
REFERENCE_DIR = os.path.join(PKG_ROOT, "data", "references")

EXPECTED_BASENAMES = {
    "video_01_genel_anamnez",
    "video_02_psikiyatrik_anamnez",
    "video_03_pediatrik_anamnez",
    "video_04_simule_hasta_gorusmesi",
    "video_05_poliklinik_anamnez_fizik_muayene",
    "video_06_gogus_hastaliklari_oyku_alma",
}

FIELDNAMES = [
    "model_id", "basename", "condition", "protocol_version", "run_purpose",
    "reference_word_count", "hypothesis_word_count", "hyp_ref_ratio",
    "substitutions", "deletions", "insertions", "insertion_rate",
    "consecutive_token_repeat_count", "longest_consecutive_token_run",
    "consecutive_phrase_repeat_count", "longest_consecutive_phrase_run",
    "distributed_occurrence_count", "repeated_token_or_phrase",
    "loop_detected", "loop_severity", "loop_detection_reason",
    "manual_review_status", "manual_verdict",
    "WER", "CER", "RTF",
]

def read_reference(basename):
    path = os.path.join(REFERENCE_DIR, basename + ".txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"REFERENCE_NOT_FOUND: {basename} -> {path}"
        )
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()

def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"COMMERCIAL_RESULTS_NOT_FOUND: {INPUT_CSV}"
        )

    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    rows = [
        row for row in all_rows
        if row.get("model_id") == "gemini-3.6-flash"
        and row.get("provider") == "gemini"
        and row.get("result_status") == "valid"
        and row.get("protocol_version") == "pilot-v2-antiloop"
        and row.get("basename") in EXPECTED_BASENAMES
    ]

    keys = [
        (
            row.get("model_id"),
            row.get("basename"),
            row.get("condition"),
            row.get("run_purpose"),
            row.get("protocol_version"),
        )
        for row in rows
    ]

    if len(rows) != 6:
        raise RuntimeError(
            f"GEMINI_RESULT_COUNT_ERROR: expected=6 actual={len(rows)}"
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("GEMINI_DUPLICATE_RESULT_ERROR")
    if {row["basename"] for row in rows} != EXPECTED_BASENAMES:
        missing = sorted(
            EXPECTED_BASENAMES - {row["basename"] for row in rows}
        )
        raise RuntimeError(f"GEMINI_MISSING_BASENAMES: {missing}")

    analyzed = []

    for row in sorted(rows, key=lambda item: item["basename"]):
        basename = row["basename"]
        reference_raw = read_reference(basename)
        hypothesis_raw = row.get("transcription", "")

        reference_words = normalize_turkish_asr(reference_raw).split()
        hypothesis_words = normalize_turkish_asr(hypothesis_raw).split()

        if not reference_words:
            raise RuntimeError(f"EMPTY_REFERENCE: {basename}")
        if not hypothesis_words:
            raise RuntimeError(f"EMPTY_TRANSCRIPTION: {basename}")

        reference_count = len(reference_words)
        hypothesis_count = len(hypothesis_words)
        insertions = int(float(row.get("I") or 0))

        insertion_rate = round(insertions / reference_count, 4)
        hyp_ref_ratio = round(
            hypothesis_count / reference_count, 4
        )

        (
            consecutive_token_count,
            longest_token_run,
            consecutive_phrase_count,
            longest_phrase_run,
            distributed_count,
            repetition_summary,
        ) = analyze_repetition_v3(hypothesis_words)

        (
            loop_detected,
            loop_severity,
            loop_reason,
        ) = classify_loop_v3(
            longest_token_run,
            consecutive_token_count,
            longest_phrase_run,
            insertion_rate,
            hyp_ref_ratio,
        )

        if loop_detected == "EVET":
            manual_status = "pending"
            manual_verdict = ""
        else:
            manual_status = "not_required"
            manual_verdict = "normal_transcription"

        analyzed.append({
            "model_id": row["model_id"],
            "basename": basename,
            "condition": row.get("condition", "none"),
            "protocol_version": row["protocol_version"],
            "run_purpose": row.get(
                "run_purpose", "commercial_reference"
            ),
            "reference_word_count": reference_count,
            "hypothesis_word_count": hypothesis_count,
            "hyp_ref_ratio": hyp_ref_ratio,
            "substitutions": row.get("S", 0),
            "deletions": row.get("D", 0),
            "insertions": insertions,
            "insertion_rate": insertion_rate,
            "consecutive_token_repeat_count":
                consecutive_token_count,
            "longest_consecutive_token_run":
                longest_token_run,
            "consecutive_phrase_repeat_count":
                consecutive_phrase_count,
            "longest_consecutive_phrase_run":
                longest_phrase_run,
            "distributed_occurrence_count":
                distributed_count,
            "repeated_token_or_phrase": repetition_summary,
            "loop_detected": loop_detected,
            "loop_severity": loop_severity,
            "loop_detection_reason": loop_reason,
            "manual_review_status": manual_status,
            "manual_verdict": manual_verdict,
            "WER": row.get("WER", ""),
            "CER": row.get("CER", ""),
            "RTF": row.get("RTF", ""),
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    temp_output = OUTPUT_CSV + ".tmp"

    with open(
        temp_output, "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(analyzed)

    os.replace(temp_output, OUTPUT_CSV)

    pathological = sum(
        row["loop_severity"] == "pathological"
        for row in analyzed
    )
    warnings = sum(
        row["loop_severity"] == "warning"
        for row in analyzed
    )
    pending = sum(
        row["manual_review_status"] == "pending"
        for row in analyzed
    )

    print(f"GEMINI V3 ANALIZ: {len(analyzed)} / 6")
    print(f"PATHOLOGICAL ADAY: {pathological}")
    print(f"WARNING ADAY: {warnings}")
    print(f"MANUEL INCELEME BEKLEYEN: {pending}")
    print(f"CIKTI: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
