import os
import sys
import wave

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
SMOKE_WAV = os.path.join(_pkg_root, "data", "diagnostic_vad_start", "audio_snippets", "qwen_smoke_5s.wav")

def create_5s_smoke_wav():
    src_wav = os.path.join(_pkg_root, "data", "audio_16k", "video_01_genel_anamnez.wav")
    if not os.path.exists(src_wav):
        print(f"Kaynak bulunamadı: {src_wav}")
        return None

    os.makedirs(os.path.dirname(SMOKE_WAV), exist_ok=True)
    with wave.open(src_wav, "rb") as rf:
        params = rf.getparams()
        # 5 saniyelik frame (16000 * 5 = 80000)
        frames = rf.readframes(80000)

    with wave.open(SMOKE_WAV, "wb") as wf:
        wf.setparams(params)
        wf.writeframes(frames)

    print(f"Smoke test WAV oluşturuldu (5s): {SMOKE_WAV}")
    return SMOKE_WAV

if __name__ == "__main__":
    create_5s_smoke_wav()
