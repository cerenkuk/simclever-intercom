import os
import csv
import json
import time
import wave
import argparse
from cloud_security import validate_cloud_upload_permission, CloudUploadDeniedError
from normalize_text import normalize_turkish_asr_v2
from calculate_metrics import calculate_wer_cer

PROMPT_VERSION_GEMINI = "gemini_verbatim_tr_v1"
NORMALIZATION_VERSION_V2 = "turkish_asr_verbatim_v2"
MODEL_ID_GEMINI = "gemini-3.6-flash"

VERBATIM_SYSTEM_PROMPT = (
    "Bu T\u00fcrk\u00e7e ses kayd\u0131n\u0131 birebir, kelimesi kelimesine yaz. "
    "Dilbilgisi veya anlat\u0131m d\u00fczeltmesi yapma. "
    "Konu\u015fmac\u0131n\u0131n s\u00f6yledi\u011fi kelimeleri daha d\u00fczg\u00fcn ifadelerle de\u011fi\u015ftirme. "
    "Dolgu seslerini ve dolgu kelimelerini koru. Tekrarlar\u0131 koru. "
    "C\u00fcmleleri \u00f6zetleme, k\u0131saltma veya yeniden yazma. "
    "Duyulmayan bir kelimeyi tahmin ederek tamamlama. "
    "Yaln\u0131zca transkripti d\u00f6nd\u00fcr; a\u00e7\u0131klama, ba\u015fl\u0131k veya \u00f6zet ekleme."
)

def run_commercial_gemini_transcription(
    basename: str,
    audio_path: str,
    metadata: dict = None,
    api_client=None,
    output_csv_path: str = None,
    run_purpose: str = "commercial_reference",
    dry_run: bool = False
) -> dict:
    """
    Executes commercial Gemini ASR API transcription with double-layer fail-closed security.
    Rules:
    1. Security check (validate_cloud_upload_permission) MUST run FIRST before any client init or file read.
    2. Missing audio file raises FileNotFoundError.
    3. --dry-run performs security and file checks, then returns early without API call or CSV write.
    4. Reference text is NEVER included in API payload; only used for local WER/CER calculation post-response.
    5. Empty or missing API response text is rejected (raises ValueError and does not write to CSV).
    6. Results are saved EXCLUSIVELY to commercial_api_results.csv.
    """
    if metadata is None:
        metadata = {}

    # 1. GÃ¶nderim Ã–ncesi Ä°kinci Katman GÃ¼venlik KontrolÃ¼ (Fail-Closed)
    validate_cloud_upload_permission(basename=basename, audio_path=audio_path, metadata=metadata)

    # 2. Dosya VarlÄ±k KontrolÃ¼
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError(f"AUDIO_FILE_NOT_FOUND: Ses dosyasÄ± bulunamadÄ± ({audio_path}).")

    # 3. Dry-Run KontrolÃ¼
    if dry_run:
        print(f"[DRY-RUN] GÃ¼venlik ve dosya kontrolleri baÅŸarÄ±lÄ± ({basename}). Ä°stemci/API/CSV Ã§aÄŸrÄ±sÄ± yapÄ±lmadÄ±.")
        return {
            "status": "dry_run_success",
            "basename": basename,
            "audio_path": audio_path
        }

    # 4. Client Ä°nisiyalleÅŸtirme
    if api_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY_MISSING: GEMINI_API_KEY Ã§evre deÄŸiÅŸkeni tanÄ±mlanmamÄ±ÅŸ.")
        try:
            from google import genai
            api_client = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError("GOOGLE_GENAI_SDK_MISSING: google-genai kÃ¼tÃ¼phanesi yÃ¼klÃ¼ deÄŸil.")

    # 5. Ses DosyasÄ± Boyut KontrolÃ¼ ve YÃ¼kleme/Ä°Ã§erik HazÄ±rlÄ±ÄŸÄ±
    file_size_bytes = os.path.getsize(audio_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    uploaded_file = None

    t0 = time.time()
    try:
        if file_size_mb > 20:
            if hasattr(api_client, "files") and hasattr(api_client.files, "upload"):
                uploaded_file = api_client.files.upload(file=audio_path)
                content_payload = [VERBATIM_SYSTEM_PROMPT, uploaded_file]
            else:
                raise RuntimeError("API_CLIENT_FILES_UNSUPPORTED: Client files.upload desteklemiyor.")
        else:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            content_payload = [
                VERBATIM_SYSTEM_PROMPT,
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_bytes
                    }
                }
            ]

        # 6. API Ã‡aÄŸrÄ±sÄ±
        if hasattr(api_client, "models") and hasattr(api_client.models, "generate_content"):
            raw_response = api_client.models.generate_content(
                model=MODEL_ID_GEMINI,
                contents=content_payload
            )
        elif hasattr(api_client, "generate_content"):
            raw_response = api_client.generate_content(content_payload)
        else:
            raise RuntimeError("API_CLIENT_METHOD_UNSUPPORTED: Client generate_content metoduna sahip deÄŸil.")

    finally:
        # Uzak dosyayÄ± silme garantisi (finally)
        if uploaded_file is not None:
            try:
                if hasattr(api_client, "files") and hasattr(api_client.files, "delete"):
                    api_client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"[WARN] Remote uploaded file delete error: {e}")

    elapsed_seconds = round(time.time() - t0, 2)

    # 7. YanÄ±t DoÄŸrulama (BoÅŸ Ã§Ä±ktÄ± CSV'ye yazÄ±lmaz)
    response_text = getattr(raw_response, "text", None)
    raw_text = response_text.strip() if isinstance(response_text, str) else ""
    if not raw_text:
        raise ValueError(f"EMPTY_API_RESPONSE: Gemini API'sinden boÅŸ veya geÃ§ersiz yanÄ±t alÄ±ndÄ± ({basename}). CSV'ye yazÄ±lmadÄ±.")


    # 8. Yerel Metrik HesabÄ± (Referans metin yalnÄ±z burada kullanÄ±lÄ±r)
    ref_raw = metadata.get("reference_text", "")
    ref_norm = normalize_turkish_asr_v2(ref_raw)
    hyp_norm = normalize_turkish_asr_v2(raw_text)

    wer, cer, sub, mole, ins = calculate_wer_cer(ref_norm, hyp_norm)
    duration_seconds = float(metadata.get("duration_seconds", 0.0))
    rtf = round(elapsed_seconds / duration_seconds, 4) if duration_seconds > 0 else 0.0

    result_row = {
        "protocol_version": "pilot-v2-antiloop",
        "run_purpose": run_purpose,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": "gemini",
        "model_id": MODEL_ID_GEMINI,
        "prompt_version": PROMPT_VERSION_GEMINI,
        "normalization_version": NORMALIZATION_VERSION_V2,
        "basename": basename,
        "condition": "none",
        "WER": wer,
        "CER": cer,
        "S": sub,
        "D": mole,
        "I": ins,
        "elapsed_seconds": elapsed_seconds,
        "duration_seconds": duration_seconds,
        "RTF": rtf,
        "transcription": raw_text,
        "result_status": "valid"
    }

    # 9. Ä°zole Ã‡Ä±ktÄ± DosyasÄ±na Yazma
    if output_csv_path is None:
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_csv_path = os.path.join(pkg_root, "outputs", "metrics", "commercial_api_results.csv")

    save_commercial_api_result(result_row, output_csv_path)
    return result_row


def save_commercial_api_result(result_row: dict, output_csv_path: str):
    """Saves result row exclusively to commercial_api_results.csv with unique key deduplication."""
    fieldnames = [
        "protocol_version", "run_purpose", "timestamp", "provider", "model_id",
        "prompt_version", "normalization_version", "basename", "condition",
        "WER", "CER", "S", "D", "I", "elapsed_seconds", "duration_seconds", "RTF",
        "transcription", "result_status"
    ]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    existing_rows = []
    if os.path.exists(output_csv_path):
        with open(output_csv_path, "r", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    new_key = (
        result_row["provider"],
        result_row["model_id"],
        result_row["basename"],
        result_row["prompt_version"],
        result_row["normalization_version"],
        result_row["run_purpose"]
    )

    filtered_rows = [
        r for r in existing_rows
        if (r.get("provider"), r.get("model_id"), r.get("basename"), r.get("prompt_version"), r.get("normalization_version"), r.get("run_purpose")) != new_key
    ]
    filtered_rows.append(result_row)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Commercial Cloud Gemini ASR Transcription with double-layer fail-closed security enforcement."
    )
    parser.add_argument("--basename", type=str, required=True, help="Basename of the audio file to transcribe.")
    parser.add_argument("--audio-path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--output-csv", type=str, help="Path to output isolated CSV results file.")
    parser.add_argument("--dry-run", action="store_true", help="Perform security and file validation without API calls or CSV writes.")
    args = parser.parse_args()

    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    reference_path = os.path.join(
        package_root,
        "data",
        "references",
        f"{args.basename}.txt"
    )

    if not os.path.isfile(reference_path):
        raise FileNotFoundError(
            f"REFERENCE_FILE_NOT_FOUND: {reference_path}"
        )

    with open(reference_path, "r", encoding="utf-8-sig") as ref_file:
        reference_text = ref_file.read()

    with wave.open(args.audio_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        duration_seconds = (
            wav_file.getnframes() / frame_rate if frame_rate else 0.0
        )

    metadata = {
        "source_group": (
            "video" if args.basename.startswith("video_") else "saha"
        ),
        "reference_text": reference_text,
        "duration_seconds": duration_seconds
    }
    result = run_commercial_gemini_transcription(
        basename=args.basename,
        audio_path=args.audio_path,
        metadata=metadata,
        output_csv_path=args.output_csv,
        dry_run=args.dry_run
    )
    print("Execution Result:", json.dumps(result, ensure_ascii=False, indent=2))





