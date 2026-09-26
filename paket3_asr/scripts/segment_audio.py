"""
Paket 3 - VAD Audio Segmentation & Silence/Noise Extractor (Bölüm 9 & 11)
SimClever Intercom - Halüsinasyon Testi İçin Sessizlik Kesitleri Ayırma
"""

import os
import sys
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Shared VAD import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
try:
    from shared.core.vad import SileroVAD
except ImportError:
    SileroVAD = None

AUDIO_16K_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "audio_16k"))
SILENCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "silence_noise"))

def extract_silence_segments():
    """Ses kayıtlarındaki VAD ile tespit edilen konuşmasız (sessizlik/gürültü) kesitleri çıkartır."""
    os.makedirs(SILENCE_DIR, exist_ok=True)
    
    files = glob.glob(os.path.join(AUDIO_16K_DIR, "*.wav"))
    if not files:
        print(f"[BİLGİ] '{AUDIO_16K_DIR}' dizininde işlenecek 16kHz WAV ses dosyası bulunamadı.")
        return
        
    print(f"VAD ile {len(files)} dosya üzerinde sessizlik ve gürültü kesitleri taranıyor...")
    # Silence extraction placeholder / skeleton
    print("VAD segmentasyon altyapısı hazır.")

if __name__ == "__main__":
    extract_silence_segments()
