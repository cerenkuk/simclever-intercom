"""
Paket 3 - Deterministik ASR Transkripsiyon Modülü (Bölüm 5, 6, 7)
SimClever Intercom - Anti-Loop (pilot-v2-antiloop) Kıyas Protokolfrom transformers import AutoProcessor,AutoModel,...
Resmî HuggingFace Whisper Long-Form model.generate() Desteği & Tepe RAM Ölçümü
"""

import os
import sys
import time
import json
import yaml

import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(__file__))
from memory_tracker import ProcessTreeMemoryTracker

EXPERIMENT_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "experiment.yaml"))
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def clean_qwen_transcript(text: str) -> str:
    """
    Qwen3-ASR modelinin ürettiği başlangıç kontrol/dil etiketlerini ('language Turkish<asr_text>' vb.) temizler.
    Metin içerisindeki normal kelimelere dokunmaz.
    """
    if not text:
        return ""

    # Yalnızca başlangıca sabitlenmiş Qwen kontrol formatı deseni
    pattern = r"^(?:language\s+[A-Za-z0-9_-]+\s*)?(?:<asr_text>|<transcribe>|<speech>)?\s*"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return cleaned


def get_experiment_decoding_config() -> tuple:
    """experiment.yaml dosyasından decoding parametrelerini ve protocol_version okur."""
    default_version = "pilot-v2-antiloop"
    default_decoding = {
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "vad_filter": True,
        "beam_size": 5
    }

    if not os.path.exists(EXPERIMENT_YAML):
        return default_version, default_decoding

    try:
        with open(EXPERIMENT_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        version = cfg.get("protocol_version", default_version)
        decoding = cfg.get("decoding", default_decoding)
        return version, decoding
    except Exception as e:
        print(f"[UYARI] experiment.yaml okunamadı ({e}), varsayılan decoding ayarları kullanılıyor.")
        return default_version, default_decoding

def torch_has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ==============================================================================
# 1. FASTER-WHISPER ENGINE (CTranslate2)
# ==============================================================================

def load_faster_whisper_model(
    model_size_or_path: str = "small",
    device: str = "cpu",
    compute_type: str = "int8"
):
    """faster-whisper modelini yükler; yükleme süresini ve tepe RAM miktarını ölçer."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("'faster-whisper' kütüphanesi yüklü değil. Lütfen 'pip install faster-whisper' çalıştırın.")

    target_device = "cuda" if (device.lower().startswith("cuda") and torch_has_cuda()) else "cpu"

    tracker = ProcessTreeMemoryTracker(interval=0.1)
    tracker.start()
    t_start = time.time()
    try:
        model = WhisperModel(model_size_or_path, device=target_device, compute_type=compute_type)
    finally:
        load_mem = tracker.stop()

    load_seconds = round(time.time() - t_start, 2)
    return model, load_seconds, target_device, load_mem

def transcribe_audio_faster_whisper(
    audio_path: str,
    model_size_or_path: str = "small",
    language: str = "tr",
    temperature: float = 0.0,
    device: str = "cpu",
    compute_type: str = "int8",
    model=None,
    load_seconds: float = 0.0,
    load_mem_stats: dict = None,
    speech_pad_ms: int = None
) -> dict:
    """
    faster-whisper (CTranslate2) motoru ile anti-loop deterministik ASR transkripsiyonu ve RAM ölçümü.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    if model is None:
        model, load_seconds, target_device, load_mem_stats = load_faster_whisper_model(
            model_size_or_path=model_size_or_path, device=device, compute_type=compute_type
        )
    else:
        target_device = "cuda" if (device.lower().startswith("cuda") and torch_has_cuda()) else "cpu"
        if load_mem_stats is None:
            # Önceden yüklenmiş model tekrar kullanılıyorsa anlık RSS başlangıç alınır
            current_rss_mb = round(ProcessTreeMemoryTracker().get_current_rss_bytes() / (1024 * 1024), 2)
            load_mem_stats = {
                "initial_ram_mb": current_rss_mb,
                "peak_ram_mb": current_rss_mb,
                "ram_delta_mb": 0.0
            }

    protocol_version, decoding_cfg = get_experiment_decoding_config()

    condition_on_prev = decoding_cfg.get("condition_on_previous_text", False)
    comp_ratio_thresh = decoding_cfg.get("compression_ratio_threshold", 2.4)
    log_prob_thresh = decoding_cfg.get("log_prob_threshold", -1.0)
    no_speech_thresh = decoding_cfg.get("no_speech_threshold", 0.6)
    vad_filter = decoding_cfg.get("vad_filter", True)
    beam_size = decoding_cfg.get("beam_size", 5)

    effective_speech_pad_ms = speech_pad_ms if speech_pad_ms is not None else decoding_cfg.get("speech_pad_ms", 400)

    applied_decoding_config = {
        "engine": "faster-whisper",
        "transcription_mode": "faster_whisper_native",
        "condition_on_previous_text": condition_on_prev,
        "compression_ratio_threshold": comp_ratio_thresh,
        "log_prob_threshold": log_prob_thresh,
        "no_speech_threshold": no_speech_thresh,
        "vad_filter": vad_filter,
        "speech_pad_ms": effective_speech_pad_ms,
        "beam_size": beam_size,
        "language": language,
        "temperature": temperature
    }
    unsupported_params = []

    vad_kwargs = {}
    if vad_filter:
        vad_kwargs["speech_pad_ms"] = effective_speech_pad_ms

    inf_tracker = ProcessTreeMemoryTracker(interval=0.1)
    inf_tracker.start()
    t_start = time.time()
    try:
        segments, info = model.transcribe(
            audio_path,
            language=language,
            temperature=temperature,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_kwargs if vad_kwargs else None,
            condition_on_previous_text=condition_on_prev,
            compression_ratio_threshold=comp_ratio_thresh,
            log_prob_threshold=log_prob_thresh,
            no_speech_threshold=no_speech_thresh
        )
        full_text = " ".join([seg.text.strip() for seg in segments]).strip()

    finally:
        inf_mem_stats = inf_tracker.stop()

    elapsed_seconds = round(time.time() - t_start, 2)
    duration_seconds = round(info.duration, 2)
    rtf_val = round(elapsed_seconds / info.duration, 4) if info.duration > 0 else 0.0
    total_seconds = round(load_seconds + elapsed_seconds, 2)

    return {
        "protocol_version": protocol_version,
        "model_name": model_size_or_path,
        "engine": "faster-whisper",
        "transcription_mode": "faster_whisper_native",
        "device": target_device,
        "compute_type": compute_type,
        "language": language,
        "temperature": temperature,
        "text": full_text,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed_seconds,
        "duration_seconds": duration_seconds,
        "rtf": rtf_val,
        "total_seconds": total_seconds,
        "load_peak_ram_mb": load_mem_stats["peak_ram_mb"],
        "load_ram_delta_mb": load_mem_stats["ram_delta_mb"],
        "inference_peak_ram_mb": inf_mem_stats["peak_ram_mb"],
        "inference_ram_delta_mb": inf_mem_stats["ram_delta_mb"],
        "peak_vram_mb": "",
        "vram_status": "N/A - CUDA unavailable",
        "memory_measurement_method": "psutil_rss_sampling_100ms_process_tree",
        "decoding_config": json.dumps(applied_decoding_config, ensure_ascii=False),
        "unsupported_decoding_params": json.dumps(unsupported_params)
    }

# ==============================================================================
# 2. TRANSFORMERS ENGINE (Official HuggingFace Whisper Long-Form model.generate)
# ==============================================================================

def load_transformers_model_and_processor(
    model_name: str = "Sercan/distil-whisper-large-v3-tr",
    device: str = "cpu",
    compute_type: str = "float32"
):
    """HuggingFace nesnelerini yükler; yükleme süresini ve tepe RAM miktarını ölçer."""
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForMultimodalLM, AutoModelForCTC, AutoModelForRNNT, AutoModel, pipeline

    except ImportError as e:
        raise ImportError(f"'transformers' veya 'torch' kütüphaneleri eksik: {e}")

    target_device = "cuda:0" if (device.lower().startswith("cuda") and torch_has_cuda()) else "cpu"

    if target_device.startswith("cuda"):
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tracker = ProcessTreeMemoryTracker(interval=0.1)
    tracker.start()
    t_start = time.time()
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if "qwen3-asr" in model_name.lower() or "qwen" in model_name.lower():
            model = AutoModelForMultimodalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        elif "nemotron" in model_name.lower():
            try:
                model = AutoModelForRNNT.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            model.to(target_device)
            model.eval()


        else:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(target_device)
            model.eval()
    finally:
        load_mem = tracker.stop()

    load_seconds = round(time.time() - t_start, 2)
    return (model, processor), load_seconds, target_device, str(torch_dtype), load_mem


# Eski fonksiyon adı uyumluluğu için alias
load_transformers_pipeline = load_transformers_model_and_processor

def transcribe_audio_transformers(
    audio_path: str,
    model_size_or_path: str = "Sercan/distil-whisper-large-v3-tr",
    language: str = "tr",
    temperature: float = 0.0,
    device: str = "cpu",
    compute_type: str = "float32",
    pipe=None,
    model_and_processor=None,
    load_seconds: float = 0.0,
    load_mem_stats: dict = None
) -> dict:
    """
    HuggingFace Resmî Long-Form / Multimodal / Pipeline yöntemi ile ASR transkripsiyonu ve RAM ölçümü.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    model_proc_obj = model_and_processor if model_and_processor is not None else pipe

    if model_proc_obj is None:
        (model, processor), load_seconds, target_device, dtype_str, load_mem_stats = load_transformers_model_and_processor(
            model_name=model_size_or_path, device=device, compute_type=compute_type
        )
    else:
        if isinstance(model_proc_obj, tuple):
            (model, processor) = model_proc_obj
        else:
            model = getattr(model_proc_obj, "model", model_proc_obj)
            processor = getattr(model_proc_obj, "processor", None)

        target_device = "cuda:0" if (device.lower().startswith("cuda") and torch_has_cuda()) else "cpu"
        dtype_str = compute_type
        if load_mem_stats is None:
            current_rss_mb = round(ProcessTreeMemoryTracker().get_current_rss_bytes() / (1024 * 1024), 2)
            load_mem_stats = {
                "initial_ram_mb": current_rss_mb,
                "peak_ram_mb": current_rss_mb,
                "ram_delta_mb": 0.0
            }

    protocol_version, decoding_cfg = get_experiment_decoding_config()

    try:
        import soundfile as sf
        speech_array, sampling_rate = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Ses dosyası okunurken hata oluştu: {e}")

    if len(speech_array.shape) > 1:
        speech_array = speech_array.mean(axis=1)

    duration_seconds = round(len(speech_array) / sampling_rate, 2)

    unsupported_params = [
        "compression_ratio_threshold",
        "log_prob_threshold",
        "no_speech_threshold",
        "vad_filter (native CTranslate2)"
    ]

    inf_tracker = ProcessTreeMemoryTracker(interval=0.1)
    inf_tracker.start()
    t_start = time.time()
    try:
        import torch
        model_name_lower = model_size_or_path.lower()

        # 1. Qwen3-ASR (AutoProcessor.apply_transcription_request + AutoModelForMultimodalLM with Adaptive 60s/30s/15s Chunking & Checkpointing)
        if "qwen3-asr" in model_name_lower or "qwen" in model_name_lower:
            import math
            import tempfile
            import wave
            import hashlib

            base_chunk_sec = 60.0
            min_chunk_sec = 15.0
            num_base_chunks = math.ceil(duration_seconds / base_chunk_sec) if duration_seconds > 0 else 1
            max_new_tokens_per_chunk = 512

            # Checkpoint Hazırlığı (Atomic JSON Cache)
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            audio_hash = hashlib.md5(f"{audio_path}_{os.path.getsize(audio_path)}".encode()).hexdigest()[:10]
            checkpoint_dir = os.path.join(_pkg_root, "data", "diagnostic_vad_start", "qwen_checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, f"qwen_chk_{audio_basename}_{audio_hash}.json")


            cached_chunks = {}
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        cached_chunks = json.load(f).get("completed_chunks", {})
                except Exception:
                    cached_chunks = {}

            applied_decoding_config = {
                "engine": "transformers",
                "transcription_mode": "qwen_adaptive_chunked_long_audio",
                "language": language,
                "temperature": temperature,
                "base_chunk_seconds": base_chunk_sec,
                "minimum_chunk_seconds": min_chunk_sec,
                "adaptive_split_enabled": True,
                "adaptive_split_count": 0,
                "chunking_enabled": True if num_base_chunks > 1 else False,
                "chunk_count": num_base_chunks,
                "max_new_tokens_per_chunk": max_new_tokens_per_chunk,
                "truncation_detected": False,
                "truncation_recovered": False,
                "skipped_silent_tail": False,
                "silent_tail_duration_seconds": 0.0,
                "silent_tail_dbfs": 0.0,
                "max_generated_tokens_observed": 0,
                "qwen_postprocessing_version": "v2_adaptive_chunking_and_silence_check"
            }
            unsupported_params.append("condition_on_previous_text")

            chunk_texts = []
            max_gen_observed = 0
            any_truncation_detected = False
            adaptive_split_count = 0
            skipped_silent_tail = False
            silent_tail_dur = 0.0
            silent_tail_dbfs = 0.0

            def compute_wav_dbfs(frames_bytes, sample_width):
                import numpy as np
                if not frames_bytes:
                    return -100.0
                if sample_width == 2:
                    samples = np.frombuffer(frames_bytes, dtype=np.int16).astype(np.float32)
                    max_possible = 32768.0
                elif sample_width == 4:
                    samples = np.frombuffer(frames_bytes, dtype=np.int32).astype(np.float32)
                    max_possible = 2147483648.0
                else:
                    samples = np.frombuffer(frames_bytes, dtype=np.int8).astype(np.float32)
                    max_possible = 128.0

                rms = np.sqrt(np.mean(samples ** 2))
                if rms <= 0:
                    return -100.0
                dbfs = 20 * np.log10(rms / max_possible)
                return round(float(dbfs), 2)

            def save_checkpoint_chunk(chunk_key, text_content):
                cached_chunks[chunk_key] = text_content
                try:
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump({"completed_chunks": cached_chunks}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            def process_audio_segment(start_s, end_s, current_chunk_idx, total_parent_chunks, tmp_dir, rf_handle, params):
                nonlocal max_gen_observed, any_truncation_detected, adaptive_split_count

                seg_dur = end_s - start_s
                chunk_key = f"{start_s:.2f}_{end_s:.2f}"

                # Checkpoint Resume
                if chunk_key in cached_chunks:
                    print(f"[QWEN CHECKPOINT RESUME] ({start_s:.1f}s - {end_s:.1f}s) Önbellekten okundu.", flush=True)
                    return [cached_chunks[chunk_key]]

                start_frame = int(start_s * params.framerate)
                num_frames = int((end_s - start_s) * params.framerate)

                seg_wav_path = os.path.join(tmp_dir, f"qwen_seg_{start_s:.1f}_{end_s:.1f}.wav")
                rf_handle.setpos(start_frame)
                frames_data = rf_handle.readframes(num_frames)

                with wave.open(seg_wav_path, "wb") as wf:
                    wf.setparams((params.nchannels, params.sampwidth, params.framerate, len(frames_data) // (params.nchannels * params.sampwidth), params.comptype, params.compname))
                    wf.writeframes(frames_data)

                print(f"[QWEN CHUNK {current_chunk_idx}/{total_parent_chunks}] ({start_s:.1f}s - {end_s:.1f}s, Süre: {seg_dur:.1f}s) Çıkarım Yapılıyor...", flush=True)
                c_t0 = time.time()

                inputs = processor.apply_transcription_request(audio=seg_wav_path)
                inputs = {k: v.to(device=target_device, dtype=torch.float32) if hasattr(v, 'to') and v.dtype.is_floating_point else (v.to(device=target_device) if hasattr(v, 'to') else v) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens_per_chunk)

                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                gen_count = generated_ids.shape[1]
                max_gen_observed = max(max_gen_observed, gen_count)

                eos_id = getattr(processor.tokenizer, "eos_token_id", None)
                has_eos = (eos_id in generated_ids[0].tolist()) if eos_id is not None else False

                # 1. Truncation Gözlemlendi (512 token & EOS yok) -> ADAPTİF İKİYE BÖLME (Checkpoint KAYDEDİLMEZ)
                if gen_count >= max_new_tokens_per_chunk and not has_eos:
                    any_truncation_detected = True
                    if seg_dur > min_chunk_sec:
                        adaptive_split_count += 1
                        mid_s = start_s + (seg_dur / 2.0)
                        print(f"[QWEN ADAPTIVE SPLIT] ({start_s:.1f}s - {end_s:.1f}s) chunk 512/EOS=False; {seg_dur/2.0:.1f}s alt parçalara bölünüyor.", flush=True)
                        sub1 = process_audio_segment(start_s, mid_s, current_chunk_idx, total_parent_chunks, tmp_dir, rf_handle, params)
                        sub2 = process_audio_segment(mid_s, end_s, current_chunk_idx, total_parent_chunks, tmp_dir, rf_handle, params)
                        return sub1 + sub2
                    else:
                        raise RuntimeError(f"[QWEN TRUNCATION ERROR] Minimum parça ({seg_dur:.1f}s) 512 token sınırına ulaştı ve EOS görünmedi (invalid_runtime_truncation)!")

                raw_text = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
                cleaned_text = clean_qwen_transcript(raw_text)

                c_elapsed = round(time.time() - c_t0, 2)
                print(f"[QWEN CHUNK {current_chunk_idx}/{total_parent_chunks}] Tamamlandı: {c_elapsed}s, token sayısı: {gen_count}, EOS: {has_eos}", flush=True)

                dbfs_val = compute_wav_dbfs(frames_data, params.sampwidth)

                # 2. Boş Çıktı Kontrolü & Sessiz Kuyruk Değerlendirmesi
                if not cleaned_text:
                    is_last_segment = (abs(end_s - duration_seconds) < 0.1)
                    eos_accepted = has_eos or (eos_id is None) or (gen_count < max_new_tokens_per_chunk)
                    if is_last_segment and seg_dur <= 5.0 and eos_accepted and dbfs_val < -50.0:
                        print(f"[QWEN SILENCE TAIL] Son {seg_dur:.1f}s parça sessiz olarak doğrulandı ({dbfs_val} dBFS) ve güvenle atlandı.", flush=True)
                        nonlocal skipped_silent_tail, silent_tail_dur, silent_tail_dbfs
                        skipped_silent_tail = True
                        silent_tail_dur = round(seg_dur, 2)
                        silent_tail_dbfs = dbfs_val
                        return []
                    else:
                        raise RuntimeError(f"[QWEN EMPTY CHUNK ERROR] Parça ({start_s:.1f}s - {end_s:.1f}s, dBFS: {dbfs_val}) konuşma enerjisi taşımasına veya orta konumda olmasına rağmen çıktı boş kaldı!")

                save_checkpoint_chunk(chunk_key, cleaned_text)
                return [cleaned_text]



            # Dynamic Chunk Bounds Generator (<15s Remainder Tail Merge Policy)
            base_ranges = []
            if duration_seconds <= base_chunk_sec:
                base_ranges.append((0.0, duration_seconds))
            else:
                curr_s = 0.0
                while curr_s < duration_seconds:
                    next_s = min(curr_s + base_chunk_sec, duration_seconds)
                    rem_s = duration_seconds - next_s
                    # Eğer kalan remainder süresi 0 < rem_s < 15.0 ise bu parça bağımsız yapılmaz, mevcut parçaya eklenir
                    if 0 < rem_s < min_chunk_sec:
                        next_s = duration_seconds
                    base_ranges.append((round(curr_s, 2), round(next_s, 2)))
                    curr_s = next_s

            num_base_chunks = len(base_ranges)

            applied_decoding_config["chunk_count"] = num_base_chunks
            applied_decoding_config["chunking_enabled"] = True if num_base_chunks > 1 else False
            applied_decoding_config["tail_merge_policy"] = "merge_remainder_less_than_15s"

            # FFmpeg / Wave PCM chunk bölme döngüsü
            with tempfile.TemporaryDirectory() as tmpdir:
                with wave.open(audio_path, "rb") as rf:
                    params = rf.getparams()
                    for i, (b_start, b_end) in enumerate(base_ranges):
                        res_texts = process_audio_segment(b_start, b_end, i + 1, num_base_chunks, tmpdir, rf, params)
                        chunk_texts.extend(res_texts)

            applied_decoding_config["max_generated_tokens_observed"] = max_gen_observed
            applied_decoding_config["truncation_detected"] = any_truncation_detected
            applied_decoding_config["truncation_recovered"] = any_truncation_detected and (adaptive_split_count > 0)
            applied_decoding_config["adaptive_split_count"] = adaptive_split_count
            applied_decoding_config["skipped_silent_tail"] = skipped_silent_tail
            applied_decoding_config["silent_tail_duration_seconds"] = silent_tail_dur
            applied_decoding_config["silent_tail_dbfs"] = silent_tail_dbfs

            full_text = " ".join(chunk_texts).strip()

            if not full_text:
                raise RuntimeError("[QWEN TRUNCATION ERROR] Son birleştirilmiş metin boş kaldı (invalid_runtime_truncation)!")






        # 2. NVIDIA Nemotron ASR (AutoModelForRNNT + AutoProcessor)
        elif "nemotron" in model_name_lower:
            # Language code mapping: tr -> tr-TR
            requested_lang = language if language else "tr"
            target_lang_code = "tr-TR" if requested_lang.lower().strip() in ["tr", "tr-tr", "turkish"] else requested_lang

            # Extract sampling rate dynamically from feature_extractor
            fe_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)

            # Processor call with dynamic sampling_rate and mapped locale (tr-TR)
            inputs = processor(
                audio=speech_array,
                sampling_rate=fe_sr,
                language=target_lang_code,
                return_tensors="pt"
            )

            inputs_on_device = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            prompt_ids = inputs_on_device.get("prompt_ids")
            num_lookahead_tokens = inputs_on_device.get("num_lookahead_tokens", 3)
            max_new_tokens = 512

            gen_kwargs = {
                "return_dict_in_generate": True,
                "max_new_tokens": max_new_tokens,
                "do_sample": False
            }
            if "input_features" in inputs_on_device:
                gen_kwargs["input_features"] = inputs_on_device["input_features"]
            elif "input_values" in inputs_on_device:
                gen_kwargs["input_values"] = inputs_on_device["input_values"]

            if "attention_mask" in inputs_on_device:
                gen_kwargs["attention_mask"] = inputs_on_device["attention_mask"]
            if prompt_ids is not None:
                gen_kwargs["prompt_ids"] = prompt_ids
            if num_lookahead_tokens is not None:
                gen_kwargs["num_lookahead_tokens"] = num_lookahead_tokens

            with torch.no_grad():
                res = model.generate(**gen_kwargs)

            seqs = res.sequences if hasattr(res, "sequences") else res
            seqs_shape = list(seqs.shape) if hasattr(seqs, "shape") else []
            seqs_dtype = str(seqs.dtype) if hasattr(seqs, "dtype") else ""

            # Token list extraction for diagnostics
            flat_tokens = seqs[0].tolist() if (hasattr(seqs, "ndim") and seqs.ndim > 1 and len(seqs[0]) > 0) else []
            first_30_tokens = flat_tokens[:30]
            last_30_tokens = flat_tokens[-30:] if len(flat_tokens) > 30 else flat_tokens

            eos_token_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
            pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)

            # Dual decoding: Raw (skip_special_tokens=False) vs Clean (skip_special_tokens=True)
            raw_decoded_list = processor.batch_decode(seqs, skip_special_tokens=False)
            clean_decoded_list = processor.batch_decode(seqs, skip_special_tokens=True)

            raw_text = raw_decoded_list[0].strip() if raw_decoded_list else ""
            full_text = clean_decoded_list[0].strip() if clean_decoded_list else ""

            is_empty_clean = (len(full_text) == 0)
            is_empty_raw = (len(raw_text) == 0)

            # Auto language fallback for diagnostic purposes only (not for production result)
            auto_fallback_result = None
            if is_empty_clean:
                try:
                    inputs_auto = processor(
                        audio=speech_array,
                        sampling_rate=fe_sr,
                        language="auto",
                        return_tensors="pt"
                    )
                    inputs_auto_dev = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs_auto.items()}
                    gen_kwargs_auto = dict(gen_kwargs)
                    if "input_features" in inputs_auto_dev: gen_kwargs_auto["input_features"] = inputs_auto_dev["input_features"]
                    if "attention_mask" in inputs_auto_dev: gen_kwargs_auto["attention_mask"] = inputs_auto_dev.get("attention_mask")
                    if "prompt_ids" in inputs_auto_dev: gen_kwargs_auto["prompt_ids"] = inputs_auto_dev["prompt_ids"]

                    with torch.no_grad():
                        res_auto = model.generate(**gen_kwargs_auto)
                    seqs_auto = res_auto.sequences if hasattr(res_auto, "sequences") else res_auto
                    raw_auto = processor.batch_decode(seqs_auto, skip_special_tokens=False)[0].strip()
                    clean_auto = processor.batch_decode(seqs_auto, skip_special_tokens=True)[0].strip()

                    auto_fallback_result = {
                        "language": "auto",
                        "raw_decoded": raw_auto,
                        "clean_decoded": clean_auto
                    }
                except Exception as ex_auto:
                    auto_fallback_result = {"error": str(ex_auto)}

            # Safe conversion helper for json serialization
            def _to_json_safe(v):
                if isinstance(v, (int, float, str, bool, type(None))):
                    return v
                if isinstance(v, (list, tuple)):
                    return [_to_json_safe(x) for x in v]
                return str(v)

            # Diagnostic metadata output object
            diagnostic_data = {
                "model_id": _to_json_safe(model_size_or_path),
                "language_requested": _to_json_safe(requested_lang),
                "language_mapped": _to_json_safe(target_lang_code),
                "sampling_rate_used": _to_json_safe(fe_sr),
                "output_type": str(type(res)),
                "sequences_type": str(type(seqs)),
                "sequences_shape": _to_json_safe(seqs_shape),
                "sequences_dtype": _to_json_safe(seqs_dtype),
                "first_30_tokens": _to_json_safe(first_30_tokens),
                "last_30_tokens": _to_json_safe(last_30_tokens),
                "eos_token_id": _to_json_safe(eos_token_id),
                "pad_token_id": _to_json_safe(pad_token_id),
                "raw_decoded_text": _to_json_safe(raw_text),
                "clean_decoded_text": _to_json_safe(full_text),
                "is_empty_raw": _to_json_safe(is_empty_raw),
                "is_empty_clean": _to_json_safe(is_empty_clean),
                "auto_language_fallback": _to_json_safe(auto_fallback_result)
            }

            # Save diagnostic output JSON
            diag_json_path = os.path.join(_pkg_root, "outputs", "metrics", "nemotron_smoke_diagnostic.json")
            os.makedirs(os.path.dirname(diag_json_path), exist_ok=True)
            with open(diag_json_path, "w", encoding="utf-8") as f_diag:
                json.dump(diagnostic_data, f_diag, ensure_ascii=False, indent=2)


            applied_decoding_config = {
                "engine": "transformers",
                "model_class": "AutoModelForRNNT",
                "decode_method": "processor_batch_decode_rnnt_direct",
                "empty_output_detected": is_empty_clean,
                "transcription_mode": "nemotron_rnnt_gen",
                "language_requested": requested_lang,
                "language_mapped": target_lang_code,
                "diagnostic_file": diag_json_path
            }
            unsupported_params.append("condition_on_previous_text")

            # FAIL-FAST RULES:
            # 1. Boş sequences veya yalnız özel token / boş clean text -> INVALID_EMPTY_OUTPUT
            if is_empty_clean:
                if not is_empty_raw:
                    applied_decoding_config["result_status"] = "INVALID_SPECIAL_TOKENS_SILENCED_TEXT"
                    raise RuntimeError(f"[NEMOTRON DİL HATASI] Ham decode dolu ('{raw_text[:50]}...') ancak temiz decode boş kaldı! Özel tokenlar metni silmektedir.")
                else:
                    applied_decoding_config["result_status"] = "INVALID_EMPTY_OUTPUT"
                    raise RuntimeError(f"[NEMOTRON BOŞ ÇIKTI HATASI] Model çıktısı hem ham hem de temiz haliyle boş kaldı (INVALID_EMPTY_OUTPUT)!")

            # 2. Non-Latin (Devanagari / Kiril) kontrolü -> INVALID_LANGUAGE_OUTPUT
            non_latin_chars = re.findall(r'[\u0400-\u04FF\u0900-\u097F]', full_text)
            if (len(non_latin_chars) / len(full_text)) > 0.3:
                applied_decoding_config["result_status"] = "INVALID_LANGUAGE_OUTPUT"
                raise RuntimeError(f"[NEMOTRON DİL HATASI] Çıktı beklenmeyen alfabe içermektedir (Kiril/Devanagari: {len(non_latin_chars)} karakter). Metin: '{full_text}'")

            applied_decoding_config["result_status"] = "valid"




        # 3. Whisper / Distil-Whisper Standard
        else:
            applied_decoding_config = {
                "engine": "transformers",
                "transcription_mode": "whisper_generate_longform",
                "return_timestamps": True,
                "condition_on_prev_tokens": False,
                "do_sample": False,
                "language": language,
                "task": "transcribe"
            }

            inputs = processor(
                speech_array,
                sampling_rate=16000,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(target_device)
            attention_mask = inputs.attention_mask.to(target_device) if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None else None

            generate_kwargs = {
                "return_timestamps": True,
                "language": language,
                "task": "transcribe",
                "do_sample": False,
                "condition_on_prev_tokens": False
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            with torch.no_grad():
                predicted_ids = model.generate(input_features, **generate_kwargs)

            full_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    except Exception as e:
        raise RuntimeError(f"[TRANSFORMERS ASR HATA] '{model_size_or_path}' transkripsiyonu sırasında hata: {e}")
    finally:
        inf_mem_stats = inf_tracker.stop()

    elapsed_seconds = round(time.time() - t_start, 2)
    rtf_val = round(elapsed_seconds / duration_seconds, 4) if duration_seconds > 0 else 0.0
    total_seconds = round(load_seconds + elapsed_seconds, 2)

    return {
        "protocol_version": protocol_version,
        "model_name": model_size_or_path,
        "engine": "transformers",
        "transcription_mode": applied_decoding_config.get("transcription_mode", "transformers_generic"),
        "device": target_device,
        "compute_type": dtype_str,
        "language": language,
        "temperature": temperature,
        "text": full_text,
        "load_seconds": load_seconds,
        "elapsed_seconds": elapsed_seconds,
        "duration_seconds": duration_seconds,
        "rtf": rtf_val,
        "total_seconds": total_seconds,
        "load_peak_ram_mb": load_mem_stats["peak_ram_mb"],
        "load_ram_delta_mb": load_mem_stats["ram_delta_mb"],
        "inference_peak_ram_mb": inf_mem_stats["peak_ram_mb"],
        "inference_ram_delta_mb": inf_mem_stats["ram_delta_mb"],
        "peak_vram_mb": "",
        "vram_status": "N/A - CUDA unavailable",
        "memory_measurement_method": "psutil_rss_sampling_100ms_process_tree",
        "decoding_config": json.dumps(applied_decoding_config, ensure_ascii=False),
        "unsupported_decoding_params": json.dumps(unsupported_params, ensure_ascii=False)
    }


if __name__ == "__main__":
    print("Anti-Loop ASR Transkripsiyon Modülü (Resmî HF Long-Form + Tepe RAM Ölçümü) Hazır.")
