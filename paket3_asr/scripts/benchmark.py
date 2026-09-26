"""
Paket 3 - Ana Deney: Ham vs Denoise (A/B) & Model Kıyaslama Orkestratörü (Bölüm 8, 12, 13)
SimClever Intercom - Türkçe ASR Seçimi ve Optimizasyonu Benchmark Runner
Protokol Sürümü: pilot-v2-antiloop (Tepe RAM Ölçümü & Ayrıştırılmış Kaynak Doğrulama Desteği)
"""

import os
import sys
import argparse
import datetime
import csv
import json
import shutil
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(__file__))
from transcribe import (
    transcribe_audio_faster_whisper,
    load_faster_whisper_model,
    transcribe_audio_transformers,
    load_transformers_model_and_processor
)
from calculate_metrics import evaluate_asr_pair
from medical_term_score import calculate_medical_term_recall

METADATA_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "metadata.csv"))
REF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "references"))
MODELS_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml"))
EXPERIMENT_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "experiment.yaml"))
OUTPUT_METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))

OUTPUT_CSV = os.path.join(OUTPUT_METRICS_DIR, "benchmark_results.csv")
RESOURCE_OUTPUT_CSV = os.path.join(OUTPUT_METRICS_DIR, "resource_benchmark_results.csv")
CANDIDATE_PRESCREEN_OUTPUT_CSV = os.path.join(OUTPUT_METRICS_DIR, "candidate_prescreen_results.csv")
PRODUCT_CANDIDATE_OUTPUT_CSV = os.path.join(OUTPUT_METRICS_DIR, "product_candidate_validation_results.csv")
BACKUP_CSV = os.path.join(OUTPUT_METRICS_DIR, "benchmark_results_pilot_v1_backup.csv")



# Koşul ve Dizin Eşleşmeleri (Paket 2 ComboBox Seçenekleri + EBU R128 Loudnorm)
CONDITION_MAP = {
    "none": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_16k")),
        "denoiser_method": "none"
    },
    "mild": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_denoised_mild")),
        "denoiser_method": "webrtc_mild"
    },
    "aggressive": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_denoised_aggressive")),
        "denoiser_method": "webrtc_aggressive"
    },
    "loudnorm": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_normalized_loudnorm")),
        "denoiser_method": "ffmpeg_ebu_r128_loudnorm"
    },
    "max": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_denoised_max")),
        "denoiser_method": "webrtc_max"
    },
    "extreme": {
        "dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_denoised_extreme")),
        "denoiser_method": "webrtc_extreme"
    }
}

# Y1 — Observation Addition kosullari experiment.yaml'dan genisletilerek eklenir.
# Kosul kimlikleri ve dizinleri koda gomulmez; tek kaynak experiment.yaml'dir.
from oa_conditions import oa_condition_map, expand_oa_conditions, extra_condition_map  # noqa: E402
from spectral_alpha import sa_condition_map, expand_sa_conditions  # noqa: E402
from adaptive_alpha import aa_condition_map, expand_aa_conditions  # noqa: E402

OA_CONDITION_IDS = [c["id"] for c in expand_oa_conditions()]
SA_CONDITION_IDS = [c["id"] for c in expand_sa_conditions()]
AA_CONDITION_IDS = [c["id"] for c in expand_aa_conditions()]
for _oa_id, _oa_cfg in oa_condition_map().items():
    if _oa_id in CONDITION_MAP:
        raise ValueError(
            f"OA kosul kimligi '{_oa_id}' temel CONDITION_MAP ile cakisiyor; "
            f"experiment.yaml id_template'ini degistirin."
        )
    CONDITION_MAP[_oa_id] = _oa_cfg

# Y1b — spektral alfa kosullari (experiment.yaml -> spectral_alpha)
for _sa_id, _sa_cfg in sa_condition_map().items():
    if _sa_id in CONDITION_MAP:
        raise ValueError(
            f"Y1b kosul kimligi '{_sa_id}' mevcut CONDITION_MAP ile cakisiyor; "
            f"experiment.yaml id_template'ini degistirin."
        )
    CONDITION_MAP[_sa_id] = _sa_cfg

# Y2 — uyarlamali alfa kosullari (experiment.yaml -> adaptive_alpha)
for _aa_id, _aa_cfg in aa_condition_map().items():
    if _aa_id in CONDITION_MAP:
        raise ValueError(
            f"Y2 kosul kimligi '{_aa_id}' mevcut CONDITION_MAP ile cakisiyor; "
            f"experiment.yaml id_template'ini degistirin."
        )
    CONDITION_MAP[_aa_id] = _aa_cfg

# Y3 ve diger tekil kosullar (experiment.yaml -> extra_conditions)
EXTRA_CONDITION_IDS = []
for _ex_id, _ex_cfg in extra_condition_map().items():
    if _ex_id in CONDITION_MAP:
        raise ValueError(f"Ek kosul kimligi '{_ex_id}' mevcut CONDITION_MAP ile cakisiyor.")
    CONDITION_MAP[_ex_id] = _ex_cfg
    EXTRA_CONDITION_IDS.append(_ex_id)

REQUIRED_FIELDNAMES = [
    "protocol_version", "run_purpose", "timestamp", "basename", "condition", "denoiser_method", "model_id",
    "engine", "transcription_mode", "model_name", "device", "compute_type", "language", "temperature",
    "transcription", "WER", "CER", "medical_term_recall", "load_seconds",
    "elapsed_seconds", "duration_seconds", "RTF", "total_seconds",
    "load_peak_ram_mb", "load_ram_delta_mb", "inference_peak_ram_mb", "inference_ram_delta_mb",
    "peak_vram_mb", "vram_status", "memory_measurement_method",
    "decoding_config", "unsupported_decoding_params",
    "found_terms", "missed_terms", "reference_text"
]

def migrate_csv_schema_if_needed(target_csv_path: str = OUTPUT_CSV, default_purpose: str = "main_benchmark"):
    """Eski CSV şemasını kayıpsız koruyarak yedekler ve yeni run_purpose ile tepe RAM alanları ekler."""
    if not os.path.exists(target_csv_path):
        return

    with open(target_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if header and header != REQUIRED_FIELDNAMES:
        print(f"[ŞEMA GÖÇÜ] Eski CSV şeması algılandı: {target_csv_path}")
        if target_csv_path == OUTPUT_CSV:
            shutil.copy2(target_csv_path, BACKUP_CSV)
            print(f"[YEDEK OLUŞTURULDU] Eski sonuçlar yedeklendi: {BACKUP_CSV}")

        migrated_rows = []
        with open(target_csv_path, "r", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            for r in dict_reader:
                eng = r.get("engine", "")
                mode = r.get("transcription_mode")
                if not mode:
                    mode = "faster_whisper_native" if eng == "faster-whisper" else "whisper_generate_longform"

                new_row = {
                    "protocol_version": r.get("protocol_version", "pilot-v1"),
                    "run_purpose": r.get("run_purpose", default_purpose),
                    "timestamp": r.get("timestamp", ""),
                    "basename": r.get("basename", ""),
                    "condition": r.get("condition", ""),
                    "denoiser_method": r.get("denoiser_method", ""),
                    "model_id": r.get("model_id", ""),
                    "engine": eng,
                    "transcription_mode": mode,
                    "model_name": r.get("model_name", ""),
                    "device": r.get("device", ""),
                    "compute_type": r.get("compute_type", ""),
                    "language": r.get("language", "tr"),
                    "temperature": r.get("temperature", "0.0"),
                    "transcription": r.get("transcription", ""),
                    "WER": r.get("WER", "0.0"),
                    "CER": r.get("CER", "0.0"),
                    "medical_term_recall": r.get("medical_term_recall", "0.0"),
                    "load_seconds": r.get("load_seconds", "0.0"),
                    "elapsed_seconds": r.get("elapsed_seconds", "0.0"),
                    "duration_seconds": r.get("duration_seconds", "0.0"),
                    "RTF": r.get("RTF", "0.0"),
                    "total_seconds": r.get("total_seconds", "0.0"),
                    "load_peak_ram_mb": r.get("load_peak_ram_mb", ""),
                    "load_ram_delta_mb": r.get("load_ram_delta_mb", ""),
                    "inference_peak_ram_mb": r.get("inference_peak_ram_mb", ""),
                    "inference_ram_delta_mb": r.get("inference_ram_delta_mb", ""),
                    "peak_vram_mb": r.get("peak_vram_mb", ""),
                    "vram_status": r.get("vram_status", ""),
                    "memory_measurement_method": r.get("memory_measurement_method", ""),
                    "decoding_config": r.get("decoding_config", "{}"),
                    "unsupported_decoding_params": r.get("unsupported_decoding_params", "[]"),
                    "found_terms": r.get("found_terms", ""),
                    "missed_terms": r.get("missed_terms", ""),
                    "reference_text": r.get("reference_text", "")
                }
                migrated_rows.append(new_row)

        with open(target_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDNAMES)
            writer.writeheader()
            writer.writerows(migrated_rows)

        print(f"[GÖÇ TAMAMLANDI] {len(migrated_rows)} kayıt yeni şemaya ({target_csv_path}) aktarıldı.")

def load_models_config():
    if not os.path.exists(MODELS_YAML):
        raise FileNotFoundError(f"Model yapılandırma dosyası bulunamadı: {MODELS_YAML}")
    with open(MODELS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", [])

def get_model_spec(model_id: str):
    models = load_models_config()
    for m in models:
        if m.get("id") == model_id:
            if m.get("deprecated", False):
                raise ValueError(
                    f"[KULLANIMI SONLANDIRILMIŞ MODEL] '{model_id}' ID'li model kullanımı sonlandırılmıştır (deprecated)! "
                    f"Lütfen resmî finalist model ID'sini kullanın."
                )
            if not m.get("enabled", False):
                raise ValueError(
                    f"[PASİF MODEL HATA] '{model_id}' ID'li model 'models.yaml' içinde pasiftir (enabled: false)! "
                    f"Benchmark çalıştırmak için enabled: true yapılmalıdır."
                )
            return m
    raise ValueError(f"'{model_id}' ID'li model '{MODELS_YAML}' dosyasında bulunamadı!")

def run_single_condition(
    basename: str,
    condition: str,
    model_spec: dict,
    loaded_model_or_pipe,
    load_sec: float,
    load_mem_stats: dict = None,
    run_purpose: str = "main_benchmark",
    speech_pad_ms: int = None
):

    if condition not in CONDITION_MAP:
        raise ValueError(f"Geçersiz koşul '{condition}'. Geçerli seçenekler: {list(CONDITION_MAP.keys())}")

    cond_info = CONDITION_MAP[condition]
    target_dir = cond_info["dir"]
    denoiser_method = cond_info["denoiser_method"]

    audio_path = os.path.join(target_dir, f"{basename}.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[{condition.upper()}] Ses dosyası bulunamadı: '{audio_path}'")

    ref_path = os.path.join(REF_DIR, f"{basename}.txt")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Referans metin (Ground Truth) dosyası bulunamadı: '{ref_path}'")

    with open(ref_path, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()

    if not reference_text:
        raise ValueError(f"Referans metin dosyası boş: '{ref_path}'")

    engine = model_spec.get("engine", "faster-whisper").lower().strip()
    model_name = model_spec.get("model_name", "small")
    device = model_spec.get("device", "cpu")
    compute_type = model_spec.get("compute_type", "int8")

    # Hedef CSV Belirleme
    if run_purpose == "resource_validation":
        target_csv = RESOURCE_OUTPUT_CSV
    elif run_purpose == "candidate_prescreen":
        target_csv = CANDIDATE_PRESCREEN_OUTPUT_CSV
    elif run_purpose == "product_candidate_validation":
        target_csv = PRODUCT_CANDIDATE_OUTPUT_CSV
    else:
        target_csv = OUTPUT_CSV

    # Fail-fast kontrolü for product_candidate_validation
    if run_purpose == "product_candidate_validation":
        if model_spec["id"] != "faster-whisper-large-v3-int8" or condition != "none":
            raise ValueError(f"[FAIL-FAST ERROR] product_candidate_validation yalnızca Large-v3 + none koşulunu kabul eder! Alınan: {model_spec['id']} + {condition}")

    os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)
    migrate_csv_schema_if_needed(target_csv, default_purpose=run_purpose)

    # Mükerrer Kontrolü (Aynı model_id + basename + condition + protocol_version + run_purpose)
    if os.path.exists(target_csv):
        with open(target_csv, "r", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            for r in dict_reader:
                if (
                    r.get("model_id") == model_spec["id"]
                    and r.get("basename") == basename
                    and r.get("condition") == condition
                    and r.get("run_purpose") == run_purpose
                ):
                    # Check speech_pad_ms matching if product_candidate_validation
                    dec_cfg = json.loads(r.get("decoding_config", "{}")) if isinstance(r.get("decoding_config"), str) else {}
                    r_pad = dec_cfg.get("speech_pad_ms")
                    if run_purpose != "product_candidate_validation" or r_pad == speech_pad_ms:
                        print(f"\n[UYARI MÜKERRER TEST] '{model_spec['id']}' | '{basename}' | '{condition}' | '{run_purpose}' (speech_pad_ms={r_pad}) sonucu '{target_csv}' dosyasında zaten var!")
                        print(f"[ATLANDI] Mükerrer kaynak/benchmark sonucu tekrar eklenmedi.")
                        return r

    print(f"\n--- [BENCHMARK RUN ({run_purpose.upper()})] Basename: '{basename}' | Koşul: '{condition}' ({denoiser_method}) | Engine: '{engine}' | Model: '{model_spec['id']}' | speech_pad_ms: {speech_pad_ms} ---")
    print(f"Ses Dosyası     : {audio_path}")
    print(f"Referans Metin  : {ref_path}")
    print(f"Hedef Çıktı CSV : {target_csv}")

    if engine == "faster-whisper":
        res = transcribe_audio_faster_whisper(
            audio_path=audio_path,
            model_size_or_path=model_name,
            language="tr",
            temperature=0.0,
            device=device,
            compute_type=compute_type,
            model=loaded_model_or_pipe,
            load_seconds=load_sec,
            load_mem_stats=load_mem_stats,
            speech_pad_ms=speech_pad_ms
        )

    elif engine == "transformers":
        res = transcribe_audio_transformers(
            audio_path=audio_path,
            model_size_or_path=model_name,
            language="tr",
            temperature=0.0,
            device=device,
            compute_type=compute_type,
            model_and_processor=loaded_model_or_pipe,
            load_seconds=load_sec,
            load_mem_stats=load_mem_stats
        )
    else:
        raise NotImplementedError(f"Desteklenmeyen motor türü: '{engine}'")

    hyp_text = res["text"]

    eval_res = evaluate_asr_pair(reference_text, hyp_text, normalize=True)
    med_res = calculate_medical_term_recall(reference_text, hyp_text)

    row = {
        "protocol_version": res.get("protocol_version", "pilot-v2-antiloop"),
        "run_purpose": run_purpose,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "basename": basename,
        "condition": condition,
        "denoiser_method": denoiser_method,
        "model_id": model_spec["id"],
        "engine": engine,
        "transcription_mode": res.get("transcription_mode", "faster_whisper_native"),
        "model_name": model_name,
        "device": res["device"],
        "compute_type": res["compute_type"],
        "language": res["language"],
        "temperature": res["temperature"],
        "transcription": hyp_text,
        "WER": eval_res["wer"],
        "CER": eval_res["cer"],
        "medical_term_recall": med_res["medical_term_accuracy"],
        "load_seconds": res["load_seconds"],
        "elapsed_seconds": res["elapsed_seconds"],
        "duration_seconds": res["duration_seconds"],
        "RTF": res["rtf"],
        "total_seconds": res["total_seconds"],
        "load_peak_ram_mb": res.get("load_peak_ram_mb", ""),
        "load_ram_delta_mb": res.get("load_ram_delta_mb", ""),
        "inference_peak_ram_mb": res.get("inference_peak_ram_mb", ""),
        "inference_ram_delta_mb": res.get("inference_ram_delta_mb", ""),
        "peak_vram_mb": res.get("peak_vram_mb", ""),
        "vram_status": res.get("vram_status", "N/A - CUDA unavailable"),
        "memory_measurement_method": res.get("memory_measurement_method", "psutil_rss_sampling_100ms_process_tree"),
        "decoding_config": res.get("decoding_config", "{}"),
        "unsupported_decoding_params": res.get("unsupported_decoding_params", "[]"),
        "found_terms": ";".join(med_res.get("found_terms", [])),
        "missed_terms": ";".join(med_res.get("missed_terms", [])),
        "reference_text": reference_text
    }

    file_exists = os.path.exists(target_csv)
    with open(target_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[SONUÇ] WER: %{row['WER']*100:.2f} | CER: %{row['CER']*100:.2f} | RTF: {row['RTF']} | Yükleme: {row['load_seconds']}s | Çıkarım: {row['elapsed_seconds']}s")
    print(f"[RAM & BELLEK] Yükleme Tepe RAM: {row['load_peak_ram_mb']} MB (Artış: +{row['load_ram_delta_mb']} MB) | Çıkarım Tepe RAM: {row['inference_peak_ram_mb']} MB (Artış: +{row['inference_ram_delta_mb']} MB) | VRAM: {row['vram_status']}")
    print(f"[KAYDEDİLDİ] Protokol '{row['protocol_version']}' / Amaç '{row['run_purpose']}' sonucu CSV'ye eklendi: {target_csv}")
    return row

def run_benchmark_pipeline(basename: str, model_id: str, condition: str, run_purpose: str = "main_benchmark", speech_pad_ms: int = None):
    model_spec = get_model_spec(model_id)
    engine = model_spec.get("engine", "faster-whisper").lower().strip()
    model_name = model_spec.get("model_name", "small")
    device = model_spec.get("device", "cpu")
    compute_type = model_spec.get("compute_type", "int8")

    print(f"\n[MODEL YÜKLEME] Motor: '{engine}' | Model: '{model_name}' | Cihaz: '{device}' | Compute: '{compute_type}' | Amaç: '{run_purpose}'")
    if engine == "faster-whisper":
        loaded_obj, load_sec, _, load_mem_stats = load_faster_whisper_model(
            model_size_or_path=model_name, device=device, compute_type=compute_type
        )
    elif engine == "transformers":
        loaded_obj, load_sec, _, _, load_mem_stats = load_transformers_model_and_processor(
            model_name=model_name, device=device, compute_type=compute_type
        )
    else:
        raise NotImplementedError(f"Desteklenmeyen motor türü: '{engine}'")

    print(f"[MODEL YÜKLENDİ] Süre: {load_sec} saniye | Tepe RAM: {load_mem_stats['peak_ram_mb']} MB (Artış: +{load_mem_stats['ram_delta_mb']} MB).")

    if condition == "all":
        # all: 4 ana koşul kapsasın (none, mild, aggressive, loudnorm)
        conditions_to_run = ["none", "mild", "aggressive", "loudnorm"]
    elif condition == "oa":
        # oa: Y1 Observation Addition koşullarının tamamı (experiment.yaml kaynaklı)
        conditions_to_run = list(OA_CONDITION_IDS)
    elif condition == "sa":
        # sa: Y1b spektral alfa koşullarının tamamı
        conditions_to_run = list(SA_CONDITION_IDS)
    elif condition == "aa":
        # aa: Y2 uyarlamalı alfa koşullarının tamamı
        conditions_to_run = list(AA_CONDITION_IDS)
    else:
        conditions_to_run = [condition]

    for cond in conditions_to_run:
        run_single_condition(basename, cond, model_spec, loaded_obj, load_sec, load_mem_stats, run_purpose=run_purpose, speech_pad_ms=speech_pad_ms)


def main():
    parser = argparse.ArgumentParser(
        description="SimClever Intercom - Paket 3 ASR A/B Test ve Benchmark Orkestratörü",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="İşlenecek ses ve referans dosyasının kök adı (Varsayılan: main_benchmark için video_01_genel_anamnez, resource_validation için saha_03_img_3602)"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="faster-whisper-small-int8",
        help="models.yaml dosyasındaki model ID'si (Varsayılan: faster-whisper-small-int8)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["none", "mild", "aggressive", "loudnorm", "max", "extreme", "all", "oa", "sa", "aa"] + OA_CONDITION_IDS + SA_CONDITION_IDS + AA_CONDITION_IDS + EXTRA_CONDITION_IDS,
        metavar="KOSUL",
        default=None,
        help=(
            "Test edilecek ses koşulu:\n"
            "  none       - Ham 16kHz Mono PCM WAV\n"
            "  mild       - WebRTC NS Mild Denoise\n"
            "  aggressive - WebRTC NS Aggressive Denoise\n"
            "  loudnorm   - EBU R128 iki geçişli FFmpeg loudnorm (bağımsız 4. koşul)\n"
            "  all        - Dört koşul: none+mild+aggressive+loudnorm\n"
            "  oa         - Tüm Y1 Observation Addition koşulları\n"
            + (f"  {', '.join(OA_CONDITION_IDS)}\n             - Tek bir OA koşulu\n" if OA_CONDITION_IDS else "")
            + "  (Varsayılan: main_benchmark için all, resource_validation için none)"
        )
    )
    parser.add_argument(
        "--run-purpose",
        type=str,
        choices=["main_benchmark", "resource_validation", "candidate_prescreen", "product_candidate_validation"],
        default="main_benchmark",
        help="Çalıştırma amacı: main_benchmark, resource_validation, candidate_prescreen veya product_candidate_validation"
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=None,
        help="VAD konuşma kenar boşluğu (ms). Varsayılan: experiment.yaml değerleri (400ms)"
    )

    args = parser.parse_args()

    # Varsayılan dosya ve koşul belirleme
    if args.run_purpose == "resource_validation":
        basename = args.basename if args.basename is not None else "saha_03_img_3602"
        condition = args.condition if args.condition is not None else "none"
    elif args.run_purpose == "product_candidate_validation":
        basename = args.basename if args.basename is not None else "video_01_genel_anamnez"
        condition = args.condition if args.condition is not None else "none"
    else:
        basename = args.basename if args.basename is not None else "video_01_genel_anamnez"
        condition = args.condition if args.condition is not None else "all"

    run_benchmark_pipeline(basename, args.model_id, condition, run_purpose=args.run_purpose, speech_pad_ms=args.speech_pad_ms)


if __name__ == "__main__":
    main()
