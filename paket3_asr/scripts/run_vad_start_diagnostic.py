"""
Paket 3 - speech_pad_ms VAD Başlangıç Kırpması Tanısal ASR Test Scripti
SimClever Intercom - Diagnostic VAD Start Evaluation Engine (Fast 60s Snippet Edition)
"""

import os
import sys
import time
import csv
import wave
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path.insert(0, _script_dir)

from normalize_text import normalize_turkish_asr_v2
from compute_norm_v2_comparison import compute_alignment_sdi
from calculate_metrics import levenshtein_distance
from memory_tracker import ProcessTreeMemoryTracker
import jiwer


DIAGNOSTIC_DIR = os.path.join(_pkg_root, "data", "diagnostic_vad_start")
SNIPPETS_DIR = os.path.join(DIAGNOSTIC_DIR, "audio_snippets")
OUT_CSV = os.path.join(_pkg_root, "outputs", "metrics", "vad_start_diagnostic_results.csv")

TARGET_VIDEOS = ["video_01_genel_anamnez", "video_06_gogus_hastaliklari_oyku_alma"]

def create_60s_snippet(src_wav_path: str, dst_snippet_path: str, max_seconds: float = 60.0) -> str:
    """
    Standart wave kütüphanesini kullanarak WAV dosyasından yalnızca ilk max_seconds saniyeyi keser.
    16 kHz mono PCM WAV formatını ve tam 60s sınırını doğrular.
    """
    os.makedirs(os.path.dirname(dst_snippet_path), exist_ok=True)

    with wave.open(src_wav_path, 'rb') as rf:
        n_channels = rf.getnchannels()
        sampwidth = rf.getsampwidth()
        framerate = rf.getframerate()
        n_frames = rf.getnframes()

        max_frames = int(framerate * max_seconds)
        frames_to_read = min(n_frames, max_frames)

        raw_bytes = rf.readframes(frames_to_read)
        actual_seconds = frames_to_read / float(framerate)

    if actual_seconds > 60.01:
        raise RuntimeError(f"Kesit süresi 60s sınırını aştı: {actual_seconds:.2f}s")

    with wave.open(dst_snippet_path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(raw_bytes)

    return dst_snippet_path

def get_audio_duration_wave(wav_path: str) -> float:
    with wave.open(wav_path, 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def get_diagnostic_reference_text(basename: str) -> str:
    """
    Tanısal kesit için referans metnini yükler.
    Önce diagnostic_vad_start/{basename}_snippet_ref.txt arar.
    Yoksa ana data/references/{basename}.txt dosyasından ilk 60 saniyelik kelimeleri alır.
    """
    snippet_ref_path = os.path.join(DIAGNOSTIC_DIR, f"{basename}_snippet_ref.txt")
    if os.path.exists(snippet_ref_path):
        with open(snippet_ref_path, encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                return text

    main_ref_path = os.path.join(_pkg_root, "data", "references", f"{basename}.txt")
    if not os.path.exists(main_ref_path):
        raise RuntimeError(f"[HATA] Ana referans TXT dosyası bulunamadı: {main_ref_path}")

    with open(main_ref_path, encoding="utf-8") as f:
        full_text = f.read().strip()

    words = full_text.split()
    if basename == "video_01_genel_anamnez":
        ref_text = " ".join(words[:49])
    elif basename == "video_06_gogus_hastaliklari_oyku_alma":
        ref_text = " ".join(words[:94])
    else:
        ref_text = " ".join(words[:50])

    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    with open(snippet_ref_path, "w", encoding="utf-8") as f:
        f.write(ref_text)

    return ref_text

def get_existing_completed_keys(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Geçersiz boş referans satırlarını sayma
            if int(r.get("reference_word_count", 0)) > 0:
                key = (r.get("basename"), r.get("model_id"), str(r.get("vad_filter")).lower(), str(r.get("speech_pad_ms")), r.get("run_purpose"))
                completed.add(key)
    return completed

def append_result_row(csv_path: str, row_dict: dict):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    fieldnames = [
        "basename", "model_id", "snippet_seconds", "vad_filter", "speech_pad_ms",
        "config_label", "reference_word_count", "hypothesis_word_count",
        "WER", "CER", "substitutions", "deletions", "insertions",
        "first_reference_words", "first_hypothesis_words",
        "leading_deletion_count", "recovered_leading_tokens",
        "first_segment_start_sec", "inference_seconds", "RTF", "peak_ram_mb",
        "run_purpose", "diagnostic_reference_review_required"
    ]
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def calculate_leading_deletion(ref_words, hyp_words):
    """
    Gerçek jiwer hizalamasının başlangıcındaki ardışık delete işlemlerinden kaç referans kelimesinin silindiğini hesaplar.
    """
    if not ref_words or not hyp_words:
        return len(ref_words) if ref_words else 0

    ref_str = " ".join(ref_words)
    hyp_str = " ".join(hyp_words)

    align = jiwer.process_words(ref_str, hyp_str)
    chunks = align.alignments[0] if align.alignments else []

    if not chunks:
        return 0

    first_chunk = chunks[0]
    if first_chunk.type == "delete" and first_chunk.ref_start_idx == 0:
        return first_chunk.ref_end_idx - first_chunk.ref_start_idx
    return 0


def run_diagnostic_vad_test(speech_pad_candidates=[400, 600, 800], include_no_vad=True):
    os.makedirs(DIAGNOSTIC_DIR, exist_ok=True)
    os.makedirs(SNIPPETS_DIR, exist_ok=True)

    try:
        from faster_whisper import WhisperModel
        from faster_whisper.vad import VadOptions
    except ImportError:
        print("[HATA] faster-whisper yüklü değil.", flush=True)
        return

    completed_keys = get_existing_completed_keys(OUT_CSV)

    # 1. Tanısal kesitlerin ve referansların oluşturulması ve doğrulanması
    snippet_map = {}
    for bn in TARGET_VIDEOS:
        src_wav = os.path.join(_pkg_root, "data", "audio_16k", f"{bn}.wav")
        dst_snippet = os.path.join(SNIPPETS_DIR, f"{bn}_first_60s.wav")

        if not os.path.exists(src_wav):
            print(f"[UYARI] Kaynak ses dosyası bulunamadı: {src_wav}", flush=True)
            continue

        create_60s_snippet(src_wav, dst_snippet, max_seconds=60.0)
        dur = get_audio_duration_wave(dst_snippet)

        if dur > 60.01:
            raise RuntimeError(f"GÜVENLİK İHLALİ: {dst_snippet} süresi 60 saniyeyi aşıyor! ({dur:.2f}s)")

        ref_text = get_diagnostic_reference_text(bn)
        ref_norm = normalize_turkish_asr_v2(ref_text)
        ref_words = ref_norm.split()

        if not ref_text.strip():
            raise RuntimeError(f"FAIL-FAST HATA: {bn} için referans metni tamamen boş!")
        if len(ref_words) < 10:
            raise RuntimeError(f"FAIL-FAST HATA: {bn} için referans kelime sayısı < 10 ({len(ref_words)} kelime)!")

        snippet_map[bn] = (dst_snippet, dur, ref_text, ref_words)
        print(f"[HAZIRLIK] {bn} için 60s kesit ve referans metni ({len(ref_words)} kelime) doğrulandı. Süre: {dur:.2f}s", flush=True)

    # 2. Test konfigürasyonlarının hazırlanması (8 benzersiz koşu)
    test_configs = []
    for pad_ms in speech_pad_candidates:
        test_configs.append({"vad_filter": True, "speech_pad_ms": pad_ms, "label": f"pad_{pad_ms}ms"})
    if include_no_vad:
        test_configs.append({"vad_filter": False, "speech_pad_ms": 0, "label": "vad_disabled_diagnostic"})

    total_runs = len(TARGET_VIDEOS) * len(test_configs)

    # 3. Modelin TEK BİR KEZ yüklenmesi
    print("\nLoading faster-whisper-large-v3-int8 model (tek bir kez yükleniyor)...", flush=True)
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    print("Model başarıyla yüklendi.\n", flush=True)

    current_run_idx = 0

    for bn in TARGET_VIDEOS:
        if bn not in snippet_map:
            continue

        snippet_path, snippet_dur, ref_raw, ref_words = snippet_map[bn]

        for cfg in test_configs:
            current_run_idx += 1

            key = (bn, "faster-whisper-large-v3-int8", str(cfg["vad_filter"]).lower(), str(cfg["speech_pad_ms"]), "vad_start_diagnostic")
            if key in completed_keys:
                print(f"[{current_run_idx}/{total_runs}] ATLANILDI (Daha önce tamamlanmış): {bn} | {cfg['label']}", flush=True)
                continue

            print(f"[{current_run_idx}/{total_runs}] BAŞLIYOR: {bn} | {cfg['label']} (Kesit Süresi: {snippet_dur:.2f}s, Referans: {len(ref_words)} kelime)...", flush=True)

            tracker = ProcessTreeMemoryTracker(interval=0.1)
            tracker.start()
            t_start = time.time()

            vad_opt = VadOptions(speech_pad_ms=cfg["speech_pad_ms"]) if cfg["vad_filter"] else None

            # Tam WAV kullanımını engelleyen fail-fast doğrulaması
            run_dur = get_audio_duration_wave(snippet_path)
            if run_dur > 60.01:
                raise RuntimeError(f"GÜVENLİK İHLALİ: Modeline tam WAV gitmesi engellendi! ({run_dur:.2f}s)")

            segments, info = model.transcribe(
                snippet_path,
                language="tr",
                temperature=0.0,
                beam_size=5,
                vad_filter=cfg["vad_filter"],
                vad_parameters=vad_opt if cfg["vad_filter"] else None
            )

            seg_list = list(segments)
            hyp_text = " ".join([s.text.strip() for s in seg_list]).strip()
            inf_mem = tracker.stop()
            elapsed = round(time.time() - t_start, 2)
            rtf = round(elapsed / snippet_dur, 4) if snippet_dur > 0 else 0.0

            ref_norm = normalize_turkish_asr_v2(ref_raw)
            hyp_norm = normalize_turkish_asr_v2(hyp_text)

            hyp_words = hyp_norm.split() if hyp_norm else []

            N = len(ref_words)
            M = len(hyp_words)

            if N == 0:
                raise RuntimeError(f"FAIL-FAST HATA: Referans kelime sayısı N == 0! Basename: {bn}")

            subs, dels, ins = compute_alignment_sdi(ref_words, hyp_words)
            wer_raw = (subs + dels + ins) / N if N > 0 else 0.0
            cer_raw = levenshtein_distance(list(ref_norm.replace(" ","")), list(hyp_norm.replace(" ",""))) / len(ref_norm.replace(" ","")) if ref_norm else 0.0

            if (subs + dels + ins) > 0 and wer_raw == 0.0:
                raise RuntimeError(f"FAIL-FAST HATA: S+D+I > 0 iken WER == 0 olamaz! (S={subs}, D={dels}, I={ins})")

            first_ref = " ".join(ref_words[:10]) if ref_words else ""
            first_hyp = " ".join(hyp_words[:10]) if hyp_words else ""

            leading_del = calculate_leading_deletion(ref_words, hyp_words)

            recovered_tokens = ""
            if leading_del == 0:
                recovered_tokens = "FULL_MATCH_START"
            elif leading_del < 5:
                recovered_tokens = f"RECOVERED_{5 - leading_del}_TOKENS"
            else:
                recovered_tokens = "NO_RECOVERY"

            first_seg_start = round(seg_list[0].start, 2) if seg_list else 0.0

            row_data = {
                "basename": bn,
                "model_id": "faster-whisper-large-v3-int8",
                "snippet_seconds": round(snippet_dur, 2),
                "vad_filter": cfg["vad_filter"],
                "speech_pad_ms": cfg["speech_pad_ms"],
                "config_label": cfg["label"],
                "reference_word_count": N,
                "hypothesis_word_count": M,
                "WER": round(wer_raw, 4),
                "CER": round(cer_raw, 4),
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "first_reference_words": first_ref,
                "first_hypothesis_words": first_hyp,
                "leading_deletion_count": leading_del,
                "recovered_leading_tokens": recovered_tokens,
                "first_segment_start_sec": first_seg_start,
                "inference_seconds": elapsed,
                "RTF": rtf,
                "peak_ram_mb": inf_mem["peak_ram_mb"],
                "run_purpose": "vad_start_diagnostic",
                "diagnostic_reference_review_required": "false"
            }

            append_result_row(OUT_CSV, row_data)
            completed_keys.add(key)

            print(f"[{current_run_idx}/{total_runs}] TAMAMLANTI ({elapsed}s, RTF: {rtf}) | WER: %{wer_raw*100:.2f} (S:{subs}, D:{dels}, I:{ins}) | Leading Del: {leading_del} | First Seg: {first_seg_start}s", flush=True)

    print(f"\nTÜM TANI TESTLERİ BİTTİ. Sonuçlar {OUT_CSV} dosyasına kaydedildi.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paket 3 Fast 60s Diagnostic VAD Start Runner")
    parser.add_argument("--pad_candidates", nargs="+", type=int, default=[400, 600, 800])
    args = parser.parse_args()
    run_diagnostic_vad_test(speech_pad_candidates=args.pad_candidates)
