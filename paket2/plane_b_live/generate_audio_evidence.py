"""
SimClever Intercom — Paket 2 Ses Kanıtı Üretimi (Teslimat 6)
=============================================================
Bu script, çevrimdışı korpus dosyaları üzerinden her gürültü tipi için
öncesi/sonrası WAV ses kanıtları üretir (canlı LAN testi yapılana kadar referans kanıt).
Her çıktı klasörün içine alınır: plane_b_live/audio_evidence/
"""
import os
import sys
import json
import numpy as np
import soundfile as sf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.denoiser import DenoiserProcessor

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plane_a_offline", "corpus")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_evidence")

NOISE_TYPES = [
    "device_alarm",
    "static_noise",
    "office_corridor",
    "babble",
    "impact_noise"
]

# Her gürültü tipi için en zorlu koşul: SNR=0 dB, reverb=300ms, mesafe=mid
TARGET_SUFFIX = "snr0dB_rev300ms_mid"

DENOISERS = [
    ("none",              "Ham Ses"),
    ("wiener",            "Adaptif Wiener Filtresi (SIFIR YUTULMA)"),
    ("pretrigger_gate",   "Pre-Trigger 50ms Gate (SIFIR KIRPILMA)"),
    ("baseline",          "Spektral Çıkarma"),
    ("webrtc_mild",       "WebRTC NS Mild (level=1)"),
    ("webrtc_aggressive", "WebRTC NS Aggressive (level=2)"),
]


def process_file(noisy_audio, method, frame_size=160, sample_rate=16000):
    denoiser = DenoiserProcessor(method=method, frame_size=frame_size, sample_rate=sample_rate)
    num_frames = len(noisy_audio) // frame_size
    out_frames = []
    for i in range(num_frames):
        frame = noisy_audio[i*frame_size:(i+1)*frame_size].astype(np.float32)
        cleaned = denoiser.process(frame)
        out_frames.append(cleaned)
    return np.concatenate(out_frames)

def generate_audio_evidence():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_rate = 16000
    frame_size = 160

    print("="*60, flush=True)
    print("  PAKET 2 — SES KANITI ÜRETİCİSİ (Teslimat 6)", flush=True)
    print("="*60, flush=True)

    evidence_log = []

    for noise_type in NOISE_TYPES:
        # En zorlu koşul dosyasını bul
        fname = f"noisy_{noise_type}_{TARGET_SUFFIX}.wav"
        fpath = os.path.join(CORPUS_DIR, fname)
        
        if not os.path.exists(fpath):
            print(f"[ATLANDI] Dosya bulunamadı: {fname}", flush=True)
            continue

        noisy_audio, sr = sf.read(fpath)
        noisy_audio = noisy_audio.astype(np.float32) / 32768.0  # int16 → float32

        noise_dir = os.path.join(OUTPUT_DIR, noise_type)
        os.makedirs(noise_dir, exist_ok=True)

        print(f"\n[{noise_type.upper()}] İşleniyor...", flush=True)

        for method, label in DENOISERS:
            processed = process_file(noisy_audio, method, frame_size, sample_rate)
            out_fname = f"{noise_type}_{method}.wav"
            out_path = os.path.join(noise_dir, out_fname)
            sf.write(out_path, (processed * 32767).astype(np.int16), sample_rate)
            print(f"  [OK] {label:<35} -> {out_fname}", flush=True)
            evidence_log.append({
                "noise_type": noise_type,
                "method": method,
                "label": label,
                "file": os.path.relpath(out_path, OUTPUT_DIR),
                "conditions": TARGET_SUFFIX
            })

    # Ses kanıtı manifest JSON
    manifest_path = os.path.join(OUTPUT_DIR, "audio_evidence_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(evidence_log, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"[OK] Ses kanitlari '{OUTPUT_DIR}' klasorune kaydedildi.", flush=True)
    print(f"[OK] Manifest: audio_evidence_manifest.json", flush=True)
    print(f"   {len(evidence_log)} adet WAV dosyasi uretildi.", flush=True)
    print(f"\nDinleme Kılavuzu:", flush=True)
    for noise_type in NOISE_TYPES:
        print(f"  {noise_type}/", flush=True)
        for method, label in DENOISERS:
                print(f"    {noise_type}_{method}.wav  <- {label}", flush=True)

if __name__ == "__main__":
    generate_audio_evidence()
