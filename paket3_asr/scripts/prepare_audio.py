"""
Paket 3 - Audio Preparation & Denoise Pipeline (Bölüm 4 & 8)
SimClever Intercom - Raw audio (.wav, .mp3, .m4a) -> 16kHz Mono PCM WAV & Denoise Variants
Protokol Sürümü: pilot-v2-antiloop
Ana Deney Matrisi: none, mild, aggressive, loudnorm (4. bağımsız koşul)
Loudnorm parametreleri: paket3_asr/config/experiment.yaml -> loudnorm bölümünden okunur.
"""

import os
import sys
import glob
import json
import subprocess
import argparse
import numpy as np
import soundfile as sf

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Shared core denoiser import
shared_core_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared", "core"))
if shared_core_dir not in sys.path:
    sys.path.insert(0, shared_core_dir)

try:
    from denoiser import DenoiserProcessor
except ImportError:
    try:
        from shared.core.denoiser import DenoiserProcessor
    except ImportError as e:
        print(f"[HATA] 'DenoiserProcessor' modülü yüklenemedi: {e}", file=sys.stderr)
        raise e

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
EXPERIMENT_YAML = os.path.join(_PKG_ROOT, "config", "experiment.yaml")

if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Y1 — Observation Addition kosul genisletici (experiment.yaml kaynakli)
from oa_conditions import (  # noqa: E402
    expand_oa_conditions,
    get_mix_settings,
    load_experiment_config,
)

# Y1b — Spektral alanda alfa kontrolu (fazi koruyarak)
from spectral_alpha import (  # noqa: E402
    build_spectral_alpha,
    expand_sa_conditions,
    get_sa_settings,
    roundtrip_error,
)

# Y2 — Kesit bazli uyarlamali alfa (zamanla degisen zarf)
from adaptive_alpha import (  # noqa: E402
    build_adaptive_alpha,
    envelope_summary,
    expand_aa_conditions,
    get_aa_settings,
)


RAW_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data", "audio_raw"))
OUT_16K_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data", "audio_16k"))
OUT_LOUDNORM_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data", "audio_normalized_loudnorm"))

# Ana Deney Kapsamında Kullanılan Denoise Koşulları (none, mild, aggressive)
DENOISER_CONFIGS = {
    "mild": {
        "dir": os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data", "audio_denoised_mild")),
        "method": "webrtc_mild"
    },
    "aggressive": {
        "dir": os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "data", "audio_denoised_aggressive")),
        "method": "webrtc_aggressive"
    }
}

VALID_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def load_loudnorm_config():
    """
    experiment.yaml'dan EBU R128 loudnorm parametrelerini okur.
    YAML kütüphanesi yüklü değilse veya yaml yoksa güvenli varsayılanlar kullanılır.
    Parametreler kod içine sabit olarak gömülmez; her zaman config dosyasından okunur.
    """
    defaults = {
        "integrated_loudness": -23.0,
        "loudness_range": 7.0,
        "true_peak": -2.0,
        "sample_rate": 16000,
        "channels": 1,
        "linear": True,
        "print_format": "json",
    }
    if not _YAML_AVAILABLE or not os.path.exists(EXPERIMENT_YAML):
        print(f"[UYARI] experiment.yaml okunamadı, varsayılan loudnorm parametreleri kullanılıyor.", file=sys.stderr)
        return defaults
    try:
        with open(EXPERIMENT_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        ln = cfg.get("loudnorm", {})
        for k, v in defaults.items():
            if k not in ln:
                ln[k] = v
        return ln
    except Exception as e:
        print(f"[UYARI] experiment.yaml okuma hatası: {e}. Varsayılanlar kullanılıyor.", file=sys.stderr)
        return defaults

def verify_audio_file(file_path: str, expected_sr: int = 16000) -> tuple:
    """Ses dosyasının örnekleme oranı, kanal sayısı, örnek sayısı ve süresini doğrular."""
    info = sf.info(file_path)
    if info.samplerate != expected_sr:
        raise ValueError(f"Örnekleme oranı uyuşmazlığı ({file_path}): Beklenen {expected_sr} Hz, Alınan {info.samplerate} Hz")
    if info.channels != 1:
        raise ValueError(f"Kanal sayısı uyuşmazlığı ({file_path}): Beklenen 1 (Mono), Alınan {info.channels}")
    duration_sec = round(info.duration, 2)
    sample_count = info.frames
    return sample_count, duration_sec

def process_single_denoise(method: str, input_16k_path: str, output_wav_path: str, frame_size: int = 160) -> int:
    """
    Bir 16kHz mono WAV dosyasını belirtilen Paket 2 DenoiserProcessor yöntemiyle
    kare kare (frame-by-frame) işler ve örnek sayısının %100 eşitliğini doğrular.
    """
    try:
        denoiser = DenoiserProcessor(method=method, frame_size=frame_size, sample_rate=16000)
        if denoiser.processor is None and method not in ["none", "passthrough"]:
            raise RuntimeError(f"DenoiserProcessor '{method}' yöntemi sürücüsü başlatılamadı!")
    except Exception as e:
        print(f"[HATA] Denoise yöntemi '{method}' başlatılırken hata oluştu: {e}", file=sys.stderr)
        raise e

    audio_data, sr = sf.read(input_16k_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # Mono kanala indirge
        
    original_length = len(audio_data)
    if original_length == 0:
        raise ValueError(f"Ses dosyası boş: {input_16k_path}")
        
    cleaned_chunks = []
    num_frames = (original_length + frame_size - 1) // frame_size
    
    for i in range(num_frames):
        start_idx = i * frame_size
        end_idx = start_idx + frame_size
        chunk = audio_data[start_idx:end_idx]
        
        if len(chunk) < frame_size:
            pad_len = frame_size - len(chunk)
            padded_chunk = np.pad(chunk, (0, pad_len), mode="constant", constant_values=0.0)
            cleaned_chunk = denoiser.process(padded_chunk)
            if cleaned_chunk is None or len(cleaned_chunk) != frame_size:
                raise RuntimeError(f"Denoiser '{method}' geçersiz kare döndürdü (frame {i}).")
            cleaned_chunks.append(cleaned_chunk[:len(chunk)])
        else:
            cleaned_chunk = denoiser.process(chunk)
            if cleaned_chunk is None or len(cleaned_chunk) != frame_size:
                raise RuntimeError(f"Denoiser '{method}' geçersiz kare döndürdü (frame {i}).")
            cleaned_chunks.append(cleaned_chunk)
            
    cleaned_audio = np.concatenate(cleaned_chunks)
    
    # Giriş ve çıkış örnek sayısının 100% eşitliğini doğrula
    if len(cleaned_audio) != original_length:
        cleaned_audio = cleaned_audio[:original_length]
        
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    sf.write(output_wav_path, cleaned_audio, sr)

    # Çıktı dosyasını sf.info ile tekrar doğrula
    denoised_samples, _ = verify_audio_file(output_wav_path, expected_sr=16000)
    if denoised_samples != original_length:
        raise RuntimeError(f"Denoise çıktı örnek sayısı ({denoised_samples}) ham ses ile ({original_length}) eşleşmiyor!")

    return original_length

def convert_raw_to_16k_pcm_wav(file_path: str, out_16k_path: str, overwrite: bool = False) -> bool:
    """
    FFmpeg ile .m4a, .mp3 veya .wav sesini doğrudan 16kHz Mono 16-bit PCM WAV dosyasına dönüştürür.
    Geçici kayıplı dosya oluşturmaz.
    """
    if os.path.exists(out_16k_path) and not overwrite:
        print(f"[ATLANDI] {out_16k_path}")
        return False

    os.makedirs(os.path.dirname(out_16k_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le",
        out_16k_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        samples, duration = verify_audio_file(out_16k_path, expected_sr=16000)
        print(f"[BAŞARILI] 16kHz Mono PCM WAV (Raw/None): {os.path.basename(out_16k_path)} ({samples} örnek, {duration} sn)")
        return True
    except Exception as e:
        print(f"[HATA] FFmpeg dönüşüm hatası ({file_path}): {e}", file=sys.stderr)
        raise e


def apply_loudnorm_two_pass(input_16k_path: str, output_loudnorm_path: str, overwrite: bool = False) -> bool:
    """
    EBU R128 iki geçişli (two-pass) FFmpeg loudnorm uygular.
    Parametreler experiment.yaml -> loudnorm bölümünden okunur; kod içine gömülümez.

    Güçlü doğrulama:
    - sample_rate = 16000
    - channel_count = 1 (mono)
    - PCM WAV
    - Süre ham kayıtla uyumlu (±0.2 sn tolerans)
    - Clipping kontrolü (ne var ise bildirilir, işlem durdurulmaz)
    - --overwrite olmadan mevcut dosya değiştirilmez ([ATLANDI] davranışı)

    Loudnorm, mild veya aggressive ile birleştirilmez; bağımsız dördüncü koşul.
    """
    if os.path.exists(output_loudnorm_path) and not overwrite:
        print(f"[ATLANDI] {output_loudnorm_path}")
        return False

    ln = load_loudnorm_config()
    il  = ln["integrated_loudness"]
    lra = ln["loudness_range"]
    tp  = ln["true_peak"]
    sr  = ln["sample_rate"]
    ch  = ln["channels"]

    # Ham kaydın özelliklerini oku (süre kıyaslamak için)
    orig_samples, orig_duration = verify_audio_file(input_16k_path, expected_sr=16000)

    os.makedirs(os.path.dirname(output_loudnorm_path), exist_ok=True)

    # Geçici çıktı dosyası
    tmp_path = output_loudnorm_path + ".tmp.wav"

    # --- GEÇİŞ 1: Ölçüm ---
    loudnorm_filter_pass1 = (
        f"loudnorm=I={il}:LRA={lra}:TP={tp}:print_format=json"
    )
    cmd_pass1 = [
        "ffmpeg", "-y",
        "-i", input_16k_path,
        "-af", loudnorm_filter_pass1,
        "-f", "null", "-"
    ]
    try:
        result1 = subprocess.run(
            cmd_pass1,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        stderr_text = result1.stderr.decode("utf-8", errors="replace")
        # JSON ölçüm değerlerini ayrıştır
        json_start = stderr_text.rfind("{")
        json_end = stderr_text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise RuntimeError(f"loudnorm 1. geçiş JSON bulunamadı.\nstderr: {stderr_text[-500:]}")
        measured = json.loads(stderr_text[json_start:json_end])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[HATA] Loudnorm 1. geçiş başarısız: {e.stderr.decode('utf-8',errors='replace')[-300:]}")

    # --- GEÇİŞ 2: Uygulama ---
    loudnorm_filter_pass2 = (
        f"loudnorm=I={il}:LRA={lra}:TP={tp}"
        f":measured_I={measured['input_i']}"
        f":measured_LRA={measured['input_lra']}"
        f":measured_TP={measured['input_tp']}"
        f":measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}"
        f":linear={'true' if ln['linear'] else 'false'}"
        f":print_format=none"
    )
    cmd_pass2 = [
        "ffmpeg", "-y",
        "-i", input_16k_path,
        "-af", loudnorm_filter_pass2,
        "-ar", str(sr),
        "-ac", str(ch),
        "-c:a", "pcm_s16le",
        tmp_path
    ]
    try:
        subprocess.run(cmd_pass2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[HATA] Loudnorm 2. geçiş başarısız: {e}")

    # --- DOĞRULAMA ---
    out_samples, out_duration = verify_audio_file(tmp_path, expected_sr=16000)

    # Süre uyumluluk kontrolü (tolerans: ±0.2 sn)
    duration_diff = abs(out_duration - orig_duration)
    if duration_diff > 0.2:
        print(
            f"[UYARI] Loudnorm çıktı süresi ({out_duration:.3f}s) "
            f"ham ses süresinden ({orig_duration:.3f}s) {duration_diff:.3f}s farklı.",
            file=sys.stderr
        )
    if out_samples != orig_samples:
        print(
            f"[BİLGİ] Örnek sayısı farkı: ham={orig_samples} loudnorm={out_samples} "
            f"(fark={out_samples-orig_samples}; tolerans içinde kabul ediliyor)",
            file=sys.stderr
        )

    # Clipping kontrolü
    data, _ = sf.read(tmp_path, dtype="float32")
    clip_count = int((abs(data) >= 0.999).sum())
    if clip_count > 0:
        print(f"[UYARI] Loudnorm çıktısında {clip_count} örnekte clipping tespit edildi.", file=sys.stderr)

    # Geçici dosyayı kalıcı konuma taşı
    if os.path.exists(output_loudnorm_path):
        os.remove(output_loudnorm_path)
    os.rename(tmp_path, output_loudnorm_path)

    print(
        f"[BAŞARILI] Loudnorm (EBU R128 two-pass): {os.path.basename(output_loudnorm_path)} "
        f"({out_samples} örnek, {out_duration:.2f}sn, clipping={clip_count})"
    )
    return True


def apply_observation_addition(
    raw_16k_path: str,
    enhanced_path: str,
    output_oa_path: str,
    alpha: float,
    rms_align: bool = True,
    output_rms_match: bool = True,
    peak_guard: float = 0.999,
    overwrite: bool = False,
) -> bool:
    """
    Y1 — Observation Addition (alfa-karisimi) uygular:

        y[n] = (1 - alpha) * x_ham[n] + alpha * x_iyilestirilmis[n]

    rms_align=True ise karisimdan ONCE iyilestirilmis yol ham sesin RMS'ine
    olceklenir; boylece alfa gercek bir karisim agirligi olur.

    output_rms_match=True ise karisim SONRASI cikti ham sesin RMS'ine sabitlenir.
    Gerekcesi: iki yol kismen ilintisiz oldugundan ara alfalarda enerji yikici
    biçimde toplanir ve seviye ~3 dB duser; bu, alfa etkisini seviye etkisiyle
    karistirir. Iki duzeltme de tek skaler kazanc oldugu icin SNR'i DEGISTIRMEZ.

    alpha=0.0 durumunda her iki duzeltmenin de kazanci 1.0'dir; cikti ham ses
    ile bit-birebir aynidir (arama uzayi ham ses tabanini gercekten icerir).

    peak_guard asilirsa tum sinyal tek bir skalerle olceklenir (kirpma yerine).

    Guclu dogrulama:
    - iki girdinin ornek sayisi birebir esit olmali (Paket 2 denoise bunu garanti eder)
    - cikti 16 kHz mono PCM WAV
    - alpha=0.0 icin cikti ham ses ile bit-birebir ayni olmali
    """
    if os.path.exists(output_oa_path) and not overwrite:
        print(f"[ATLANDI] {output_oa_path}")
        return False

    raw, sr_raw = sf.read(raw_16k_path, dtype="float32")
    enh, sr_enh = sf.read(enhanced_path, dtype="float32")

    if raw.ndim > 1:
        raw = raw[:, 0]
    if enh.ndim > 1:
        enh = enh[:, 0]

    if sr_raw != 16000 or sr_enh != 16000:
        raise ValueError(
            f"OA girdileri 16 kHz olmali (ham={sr_raw} Hz, iyilestirilmis={sr_enh} Hz)"
        )
    if len(raw) != len(enh):
        raise ValueError(
            f"OA girdilerinin ornek sayisi esit degil: ham={len(raw)}, iyilestirilmis={len(enh)} "
            f"({os.path.basename(raw_16k_path)})"
        )
    if len(raw) == 0:
        raise ValueError(f"Ses dosyasi bos: {raw_16k_path}")

    raw64 = raw.astype(np.float64)
    enh64 = enh.astype(np.float64)
    rms_raw = float(np.sqrt(np.mean(raw64 ** 2)))

    applied_gain = 1.0
    if rms_align:
        rms_enh = float(np.sqrt(np.mean(enh64 ** 2)))
        if rms_enh > 1e-9:
            applied_gain = rms_raw / rms_enh
            enh64 = enh64 * applied_gain
        else:
            print(
                f"[UYARI] Iyilestirilmis yolun RMS'i sifira cok yakin ({rms_enh:.3e}); "
                f"RMS esitleme atlandi: {os.path.basename(enhanced_path)}",
                file=sys.stderr,
            )

    mixed = (1.0 - alpha) * raw64 + alpha * enh64

    # Cikti seviyesini ham sesin RMS'ine sabitle (alfa boyunca seviye sabit kalir)
    out_gain = 1.0
    if output_rms_match:
        rms_mix = float(np.sqrt(np.mean(mixed ** 2)))
        if rms_mix > 1e-9 and rms_raw > 1e-9:
            out_gain = rms_raw / rms_mix
            mixed = mixed * out_gain

    # Tepe korumasi: amaci KARISIMIN yarattigi kirpmayi onlemektir, girdide
    # zaten var olan tam olcegi duzeltmek DEGILDIR. Ham ses tepesi peak_guard'i
    # asiyorsa (orn. video_01 tam olcekte tepiyor: tepe=1.0), esik ham tepeye
    # yukseltilir. Aksi halde alpha=0'da bile sinyal olceklenir ve 'ham ses
    # arama uzayinin icinde' garantisi bozulur.
    raw_peak = float(np.max(np.abs(raw64))) if len(raw64) else 0.0
    guard_threshold = max(peak_guard, raw_peak)
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    peak_scale = 1.0
    if peak > guard_threshold:
        peak_scale = guard_threshold / peak
        mixed = mixed * peak_scale

    mixed = mixed.astype(np.float32)

    os.makedirs(os.path.dirname(output_oa_path), exist_ok=True)
    sf.write(output_oa_path, mixed, 16000, subtype="PCM_16")

    out_samples, out_duration = verify_audio_file(output_oa_path, expected_sr=16000)
    if out_samples != len(raw):
        raise RuntimeError(
            f"OA cikti ornek sayisi ({out_samples}) ham ses ile ({len(raw)}) eslesmiyor!"
        )

    print(
        f"[BASARILI] OA (alpha={alpha:.2f}): {os.path.basename(output_oa_path)} "
        f"({out_samples} ornek, {out_duration:.2f} sn, yol_kazanc={applied_gain:.3f}, "
        f"cikti_kazanc={out_gain:.3f}, tepe_olcek={peak_scale:.3f})"
    )
    return True


def verify_oa_endpoint(
    raw_16k_path: str,
    enhanced_path: str,
    rms_align: bool,
    peak_guard: float,
    output_rms_match: bool = True,
) -> None:
    """
    alpha=0.0 ozdesligini dogrular: karisim ham sesin ta kendisi olmalidir.
    Bu, OA arama uzayinin ham ses tabanini gercekten icerdigini kanitlar.
    Hata durumunda istisna firlatir.
    """
    raw, _ = sf.read(raw_16k_path, dtype="float32")
    enh, _ = sf.read(enhanced_path, dtype="float32")
    if raw.ndim > 1:
        raw = raw[:, 0]
    if enh.ndim > 1:
        enh = enh[:, 0]
    n = min(len(raw), len(enh))
    raw, enh = raw[:n], enh[:n]

    raw64 = raw.astype(np.float64)
    enh64 = enh.astype(np.float64)
    rms_raw = float(np.sqrt(np.mean(raw64 ** 2)))

    if rms_align:
        rms_enh = float(np.sqrt(np.mean(enh64 ** 2)))
        if rms_enh > 1e-9:
            enh64 = enh64 * (rms_raw / rms_enh)

    mixed = (1.0 - 0.0) * raw64 + 0.0 * enh64

    if output_rms_match:
        rms_mix = float(np.sqrt(np.mean(mixed ** 2)))
        if rms_mix > 1e-9 and rms_raw > 1e-9:
            mixed = mixed * (rms_raw / rms_mix)

    raw_peak = float(np.max(np.abs(raw64))) if len(raw64) else 0.0
    guard_threshold = max(peak_guard, raw_peak)
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > guard_threshold:
        mixed = mixed * (guard_threshold / peak)

    max_dev = float(np.max(np.abs(mixed.astype(np.float32) - raw)))
    if max_dev != 0.0:
        raise RuntimeError(
            f"OA uc dogrulamasi BASARISIZ: alpha=0.0 ciktisi ham sesten sapiyor "
            f"(maks sapma={max_dev:.3e}). Ham ses tabani arama uzayinda degil!"
        )
    print(f"[DOGRULANDI] OA uc noktasi: alpha=0.0 ciktisi ham ses ile bit-birebir ayni ({os.path.basename(raw_16k_path)})")


def _finalize_mix(mixed, raw64, output_rms_match: bool, peak_guard: float):
    """
    Karisim sonrasi seviye ve tepe islemleri (Y1b icin).
    Y1'in apply_observation_addition icindeki mantigin aynisidir; dogrulanmis
    kodu degistirmemek icin ayri tutulmustur.
    Doner: (mixed, cikti_kazanci, tepe_olcegi)
    """
    rms_raw = float(np.sqrt(np.mean(raw64 ** 2)))
    out_gain = 1.0
    if output_rms_match:
        rms_mix = float(np.sqrt(np.mean(mixed ** 2)))
        if rms_mix > 1e-9 and rms_raw > 1e-9:
            out_gain = rms_raw / rms_mix
            mixed = mixed * out_gain

    raw_peak = float(np.max(np.abs(raw64))) if len(raw64) else 0.0
    guard_threshold = max(peak_guard, raw_peak)
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    peak_scale = 1.0
    if peak > guard_threshold:
        peak_scale = guard_threshold / peak
        mixed = mixed * peak_scale
    return mixed, out_gain, peak_scale


def apply_spectral_alpha(
    raw_16k_path: str,
    enhanced_path: str,
    output_path: str,
    alpha: float,
    variant: str,
    settings: dict,
    overwrite: bool = False,
) -> bool:
    """
    Y1b — Whisper'in kendi STFT izgarasinda (n_fft=400, hop=160) spektral alanda
    alfa kontrollu karisim uretir. Faz her zaman ham sesin fazidir (gamma=0).

    variant='psm' -> guc spektrumu dogrusal karisimi (Mel alani karisiminin
                     matematiksel esdegeri; mel = DOGRUSAL matris @ guc)
    variant='mex' -> ortuk maske ussu A^alpha (arXiv 2607.11157 formulasyonu)
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"[ATLANDI] {output_path}")
        return False

    raw, sr_raw = sf.read(raw_16k_path, dtype="float64")
    enh, sr_enh = sf.read(enhanced_path, dtype="float64")
    if raw.ndim > 1:
        raw = raw[:, 0]
    if enh.ndim > 1:
        enh = enh[:, 0]
    if sr_raw != 16000 or sr_enh != 16000:
        raise ValueError(f"Y1b girdileri 16 kHz olmali (ham={sr_raw}, iyi={sr_enh})")
    if len(raw) != len(enh):
        raise ValueError(
            f"Y1b girdilerinin ornek sayisi esit degil: ham={len(raw)}, iyi={len(enh)}"
        )

    mixed = build_spectral_alpha(
        raw, enh, alpha=alpha, variant=variant,
        mask_clip_max=settings["mask_clip_max"],
        n_fft=settings["n_fft"], hop=settings["hop_length"],
    )
    mixed, out_gain, peak_scale = _finalize_mix(
        mixed, raw, settings["output_rms_match"], settings["peak_guard"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mixed.astype(np.float32), 16000, subtype="PCM_16")

    out_samples, out_duration = verify_audio_file(output_path, expected_sr=16000)
    if out_samples != len(raw):
        raise RuntimeError(
            f"Y1b cikti ornek sayisi ({out_samples}) ham ses ile ({len(raw)}) eslesmiyor!"
        )
    print(
        f"[BASARILI] Y1b {variant} (alpha={alpha:.2f}): {os.path.basename(output_path)} "
        f"({out_samples} ornek, {out_duration:.2f} sn, cikti_kazanc={out_gain:.3f}, "
        f"tepe_olcek={peak_scale:.3f})"
    )
    return True


def apply_adaptive_alpha(
    raw_16k_path: str,
    enhanced_path: str,
    output_path: str,
    rule: str,
    params: dict,
    settings: dict,
    smooth_ms: float,
    overwrite: bool = False,
) -> bool:
    """
    Y2 — Zamanla degisen alfa zarfiyla spektral karisim uretir.
    Sabit zarf verildiginde Y1b ile birebir ayni cikti verir (dogrulandi).
    """
    if os.path.exists(output_path) and not overwrite:
        print(f"[ATLANDI] {output_path}")
        return False

    raw, sr_raw = sf.read(raw_16k_path, dtype="float64")
    enh, sr_enh = sf.read(enhanced_path, dtype="float64")
    if raw.ndim > 1:
        raw = raw[:, 0]
    if enh.ndim > 1:
        enh = enh[:, 0]
    if sr_raw != 16000 or sr_enh != 16000:
        raise ValueError(f"Y2 girdileri 16 kHz olmali (ham={sr_raw}, iyi={sr_enh})")
    if len(raw) != len(enh):
        raise ValueError(
            f"Y2 girdilerinin ornek sayisi esit degil: ham={len(raw)}, iyi={len(enh)}"
        )

    mixed, info = build_adaptive_alpha(
        raw, enh, rule=rule, params=params,
        variant=settings["variant"], smooth_ms=smooth_ms,
        mask_clip_max=settings["mask_clip_max"], return_envelope=True,
    )
    mixed, out_gain, peak_scale = _finalize_mix(
        mixed, raw, settings["output_rms_match"], settings["peak_guard"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, mixed.astype(np.float32), 16000, subtype="PCM_16")

    out_samples, out_duration = verify_audio_file(output_path, expected_sr=16000)
    if out_samples != len(raw):
        raise RuntimeError(
            f"Y2 cikti ornek sayisi ({out_samples}) ham ses ile ({len(raw)}) eslesmiyor!"
        )
    es = envelope_summary(info["envelope"])
    print(
        f"[BASARILI] Y2 {rule} ({os.path.basename(os.path.dirname(output_path))}): "
        f"{os.path.basename(output_path)} ({out_duration:.2f} sn, "
        f"alfa_ort={es['alpha_mean']:.3f}, dokunulmayan=%{es['frac_untouched']*100:.1f}, "
        f"tam=%{es['frac_full']*100:.1f}, cikti_kazanc={out_gain:.3f}, "
        f"tepe_olcek={peak_scale:.3f})"
    )
    return True


def process_raw_audios(target_basename: str = None, condition: str = "all", overwrite: bool = False):
    """
    audio_raw içindeki ses dosyalarını 16kHz mono PCM WAV formatına dönüştürür ve
    seçilen koşullara göre Paket 2 gürültü bastırma yöntemlerini ve/veya EBU R128 loudnorm'u koşturur.

    Koşullar:
      none       - Ham 16kHz Mono PCM WAV
      mild       - WebRTC NS Mild Denoise
      aggressive - WebRTC NS Aggressive Denoise
      loudnorm   - EBU R128 iki geçişli FFmpeg loudnorm (bağımsız 4. koşul)
      all        - Dört koşulun tamamı: none, mild, aggressive, loudnorm
    """
    os.makedirs(OUT_16K_DIR, exist_ok=True)
    os.makedirs(OUT_LOUDNORM_DIR, exist_ok=True)
    
    all_raw_files = glob.glob(os.path.join(RAW_DIR, "*.*"))
    valid_files = [
        f for f in all_raw_files
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]
    
    if target_basename:
        target_basename_clean = target_basename.strip()
        valid_files = [
            f for f in valid_files
            if os.path.splitext(os.path.basename(f))[0] == target_basename_clean
        ]
        if not valid_files:
            raise FileNotFoundError(f"[HATA] '{RAW_DIR}' klasöründe '{target_basename}' kök adına sahip geçerli bir .m4a, .mp3 veya .wav ses dosyası bulunamadı!")

    if not valid_files:
        print(f"[UYARI] '{RAW_DIR}' klasöründe işlenecek geçerli ses dosyası bulunamadı.")
        return

    # Y1 OA kosullarini experiment.yaml'dan genislet
    _exp_cfg = load_experiment_config()
    oa_all = expand_oa_conditions(_exp_cfg)
    oa_by_id = {c["id"]: c for c in oa_all}
    mix_cfg = get_mix_settings(_exp_cfg)

    # Y1b spektral alfa kosullari
    sa_all = expand_sa_conditions(_exp_cfg)
    sa_by_id = {c["id"]: c for c in sa_all}
    sa_cfg = get_sa_settings(_exp_cfg)

    # Y2 uyarlamali alfa kosullari
    aa_all = expand_aa_conditions(_exp_cfg)
    aa_by_id = {c["id"]: c for c in aa_all}
    aa_cfg = get_aa_settings(_exp_cfg)

    # Iyilestirici kimliginden (webrtc_mild) uretim dizinine esleme
    enhancer_dir_by_method = {cfg["method"]: cfg["dir"] for cfg in DENOISER_CONFIGS.values()}
    enhancer_condkey_by_method = {cfg["method"]: key for key, cfg in DENOISER_CONFIGS.items()}

    # Koşul belirleme (temel 4 koşul + Y1 OA koşulları)
    if condition == "all":
        # all: temel dört koşulu kapsasın (OA hariç; OA ayrı istenir)
        conditions_to_run = ["none", "mild", "aggressive", "loudnorm"]
    elif condition == "oa":
        # oa: tüm OA koşulları (temel iyileştiriciler otomatik üretilir)
        conditions_to_run = [c["id"] for c in oa_all]
    elif condition == "sa":
        # sa: tüm Y1b spektral alfa koşulları
        conditions_to_run = [c["id"] for c in sa_all]
    elif condition == "aa":
        # aa: tüm Y2 uyarlamalı alfa koşulları
        conditions_to_run = [c["id"] for c in aa_all]
    else:
        conditions_to_run = [condition]

    aa_to_run = [aa_by_id[c] for c in conditions_to_run if c in aa_by_id]
    for _c in aa_to_run:
        _need = enhancer_condkey_by_method.get(_c["enhancer"])
        if _need and _need not in conditions_to_run:
            conditions_to_run.append(_need)
            print(f"[BILGI] '{_c['id']}' icin temel iyilestirici kosulu '{_need}' otomatik eklendi.")

    sa_to_run = [sa_by_id[c] for c in conditions_to_run if c in sa_by_id]
    for _c in sa_to_run:
        _need = enhancer_condkey_by_method.get(_c["enhancer"])
        if _need and _need not in conditions_to_run:
            conditions_to_run.append(_need)
            print(f"[BILGI] '{_c['id']}' icin temel iyilestirici kosulu '{_need}' otomatik eklendi.")

    oa_to_run = [oa_by_id[c] for c in conditions_to_run if c in oa_by_id]
    # OA icin gereken temel iyilestirici kosullarini otomatik ekle
    for c in oa_to_run:
        needed = enhancer_condkey_by_method.get(c["enhancer"])
        if needed and needed not in conditions_to_run:
            conditions_to_run.append(needed)
            print(f"[BILGI] '{c['id']}' icin temel iyilestirici kosulu '{needed}' otomatik eklendi.")

    print(f"\n--- [AUDIO PREPARE PIPELINE] Toplam {len(valid_files)} ses dosyası | Koşullar: {conditions_to_run} | Overwrite: {overwrite} ---")

    for file_path in valid_files:
        basename = os.path.splitext(os.path.basename(file_path))[0]
        out_16k_path = os.path.join(OUT_16K_DIR, f"{basename}.wav")

        # 1. Raw / None 16kHz PCM WAV Dönüşümü (loudnorm için de gerekli)
        if (
            "none" in conditions_to_run
            or "mild" in conditions_to_run
            or "aggressive" in conditions_to_run
            or "loudnorm" in conditions_to_run
        ) and not os.path.exists(out_16k_path):
            convert_raw_to_16k_pcm_wav(file_path, out_16k_path, overwrite=overwrite)
        elif "none" in conditions_to_run:
            convert_raw_to_16k_pcm_wav(file_path, out_16k_path, overwrite=overwrite)

        # 2. Denoise Koşulları (mild, aggressive)
        for cond_key, cfg in DENOISER_CONFIGS.items():
            if cond_key in conditions_to_run:
                target_dir = cfg["dir"]
                method_name = cfg["method"]
                output_denoised_path = os.path.join(target_dir, f"{basename}.wav")

                if os.path.exists(output_denoised_path) and not overwrite:
                    print(f"[ATLANDI] {output_denoised_path}")
                    continue

                if not os.path.exists(out_16k_path):
                    convert_raw_to_16k_pcm_wav(file_path, out_16k_path, overwrite=False)

                try:
                    sample_count = process_single_denoise(method_name, out_16k_path, output_denoised_path, frame_size=160)
                    duration_sec = round(sample_count / 16000.0, 2)
                    print(f"[BAŞARILI] Denoised ({cond_key} / {method_name}): {basename}.wav ({sample_count} örnek, {duration_sec} sn)")
                except Exception as e:
                    print(f"[HATA] Denoise '{cond_key}' işleme hatası ({basename}): {e}", file=sys.stderr)
                    raise e

        # 3. Loudnorm Koşulu (EBU R128 two-pass; none/mild/aggressive ile birleştirilmez)
        if "loudnorm" in conditions_to_run:
            output_loudnorm_path = os.path.join(OUT_LOUDNORM_DIR, f"{basename}.wav")
            if not os.path.exists(out_16k_path):
                convert_raw_to_16k_pcm_wav(file_path, out_16k_path, overwrite=False)
            try:
                apply_loudnorm_two_pass(out_16k_path, output_loudnorm_path, overwrite=overwrite)
            except Exception as e:
                print(f"[HATA] Loudnorm işleme hatası ({basename}): {e}", file=sys.stderr)
                raise e

        # 4. Y1 — Observation Addition Koşulları (alfa-karışımı)
        endpoint_verified_for_enhancer = set()
        for oa in oa_to_run:
            base_dir = enhancer_dir_by_method.get(oa["enhancer"])
            if base_dir is None:
                raise ValueError(
                    f"OA temel iyileştiricisi '{oa['enhancer']}' DENOISER_CONFIGS içinde tanımlı değil."
                )
            enhanced_path = os.path.join(base_dir, f"{basename}.wav")
            if not os.path.exists(enhanced_path):
                raise FileNotFoundError(
                    f"OA için temel iyileştirilmiş ses bulunamadı: '{enhanced_path}'. "
                    f"Önce --condition {enhancer_condkey_by_method.get(oa['enhancer'])} koşturun."
                )

            # alpha=0.0 özdeşliğini iyileştirici başına bir kez doğrula
            if mix_cfg["verify_endpoints"] and oa["enhancer"] not in endpoint_verified_for_enhancer:
                verify_oa_endpoint(
                    out_16k_path, enhanced_path,
                    rms_align=mix_cfg["rms_align"],
                    peak_guard=mix_cfg["peak_guard"],
                    output_rms_match=mix_cfg["output_rms_match"],
                )
                endpoint_verified_for_enhancer.add(oa["enhancer"])

            output_oa_path = os.path.join(oa["dir"], f"{basename}.wav")
            try:
                apply_observation_addition(
                    out_16k_path,
                    enhanced_path,
                    output_oa_path,
                    alpha=oa["alpha"],
                    rms_align=mix_cfg["rms_align"],
                    output_rms_match=mix_cfg["output_rms_match"],
                    peak_guard=mix_cfg["peak_guard"],
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[HATA] OA '{oa['id']}' işleme hatası ({basename}): {e}", file=sys.stderr)
                raise e

        # 5. Y1b — Spektral alanda alfa kontrolü
        if sa_to_run and sa_cfg["verify_endpoints"]:
            _probe, _ = sf.read(out_16k_path, dtype="float64")
            if _probe.ndim > 1:
                _probe = _probe[:, 0]
            _probe = _probe[: min(len(_probe), 16000 * 20)]
            _rt = roundtrip_error(_probe, n_fft=sa_cfg["n_fft"], hop=sa_cfg["hop_length"])
            if _rt > 1e-6:
                raise RuntimeError(
                    f"Y1b STFT gidis-donus hatasi cok buyuk ({_rt:.3e}); alpha=0 "
                    f"ozdesligi garanti edilemez: {basename}"
                )
            print(f"[DOGRULANDI] Y1b STFT gidis-donus hatasi = {_rt:.2e} ({basename})")

        for sa in sa_to_run:
            _bdir = enhancer_dir_by_method.get(sa["enhancer"])
            if _bdir is None:
                raise ValueError(
                    f"Y1b temel iyilestiricisi '{sa['enhancer']}' DENOISER_CONFIGS icinde tanimli degil."
                )
            _epath = os.path.join(_bdir, f"{basename}.wav")
            if not os.path.exists(_epath):
                raise FileNotFoundError(
                    f"Y1b icin temel iyilestirilmis ses bulunamadi: '{_epath}'"
                )
            try:
                apply_spectral_alpha(
                    out_16k_path, _epath,
                    os.path.join(sa["dir"], f"{basename}.wav"),
                    alpha=sa["alpha"], variant=sa["variant"],
                    settings=sa_cfg, overwrite=overwrite,
                )
            except Exception as e:
                print(f"[HATA] Y1b '{sa['id']}' isleme hatasi ({basename}): {e}", file=sys.stderr)
                raise e

        # 6. Y2 — Kesit bazli uyarlamali alfa
        for aa in aa_to_run:
            _bdir = enhancer_dir_by_method.get(aa["enhancer"])
            if _bdir is None:
                raise ValueError(
                    f"Y2 temel iyilestiricisi '{aa['enhancer']}' DENOISER_CONFIGS icinde tanimli degil."
                )
            _epath = os.path.join(_bdir, f"{basename}.wav")
            if not os.path.exists(_epath):
                raise FileNotFoundError(
                    f"Y2 icin temel iyilestirilmis ses bulunamadi: '{_epath}'"
                )
            try:
                apply_adaptive_alpha(
                    out_16k_path, _epath,
                    os.path.join(aa["dir"], f"{basename}.wav"),
                    rule=aa["rule"], params=aa["params"],
                    settings=aa_cfg, smooth_ms=aa["smooth_ms"],
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"[HATA] Y2 '{aa['id']}' isleme hatasi ({basename}): {e}", file=sys.stderr)
                raise e



def main():
    parser = argparse.ArgumentParser(
        description=(
            "SimClever Intercom - Paket 3 Audio Preparation & Denoise Pipeline\n"
            "Koşullar: none (ham), mild (WebRTC NS Mild), aggressive (WebRTC NS Aggressive),\n"
            "          loudnorm (EBU R128 iki geçişli FFmpeg; bağımsız 4. koşul),\n"
            "          all (dordünü kapsasın: none+mild+aggressive+loudnorm)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="İşlenecek ses dosyasının kök adı (örn: saha_01_img_3463). Belirtilmezse tüm raw sesler işlenir."
    )
    _oa_ids = [c["id"] for c in expand_oa_conditions()]
    _sa_ids = [c["id"] for c in expand_sa_conditions()]
    _aa_ids = [c["id"] for c in expand_aa_conditions()]
    _base_choices = ["none", "mild", "aggressive", "loudnorm", "all", "oa", "sa", "aa"]
    parser.add_argument(
        "--condition",
        type=str,
        choices=_base_choices + _oa_ids + _sa_ids + _aa_ids,
        metavar="KOSUL",
        default="all",
        help=(
            "Uretilecek ses koşulu:\n"
            "  none       - Ham 16kHz Mono PCM WAV\n"
            "  mild       - WebRTC NS Mild Denoise\n"
            "  aggressive - WebRTC NS Aggressive Denoise\n"
            "  loudnorm   - EBU R128 iki geçişli loudnorm (bağımsız 4. koşul)\n"
            "  all        - Temel dört koşul (Varsayılan: all)\n"
            "  oa         - Tüm Y1 Observation Addition koşulları\n"
            + (f"  {', '.join(_oa_ids)}\n             - Tek bir OA koşulu\n" if _oa_ids else "")
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Var olan çıktı dosyalarının üzerine tekrar yaz (Varsayılan: False)"
    )

    args = parser.parse_args()
    process_raw_audios(target_basename=args.basename, condition=args.condition, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
