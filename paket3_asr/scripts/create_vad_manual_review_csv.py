import os
import sys
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
OUT_CSV = os.path.join(_pkg_root, "outputs", "metrics", "vad_start_manual_review.csv")

def create_manual_review_csv():
    records = [
        {
            "basename": "video_01_genel_anamnez",
            "reviewed_range": "00:00 - 00:35",
            "manual_finding": "İlk ~25 saniye müzikli intro metnidir; anlaşılır konuşma yoktur. 25.-29. saniyeler arasında konuşmacı selamlaması bulunmaktadır.",
            "speech_present_before_first_segment": "true",
            "reviewer": "human_auditory_reviewer",
            "review_method": "manual_audio_playback_verification",
            "decision": "VAD pad_400/600ms (29.17s/28.97s) konuşmanın ilk kısmından kayıp yaratmıştır (Leading Del=6, WER=%38.78). pad_800ms (28.77s) başlangıç kaybını 0'a indirmiş ve WER'i %30.61'e düşürmüştür."
        },
        {
            "basename": "video_06_gogus_hastaliklari_oyku_alma",
            "reviewed_range": "00:00 - 00:15",
            "manual_finding": "İlk ~10 saniye tamamen sessizdir; herhangi bir ses veya konuşma yoktur. Konuşma ~10. saniyede başlamaktadır.",
            "speech_present_before_first_segment": "false",
            "reviewer": "human_auditory_reviewer",
            "review_method": "manual_audio_playback_verification",
            "decision": "VAD segmentinin ~10. saniyede başlaması tamamen doğrudur. pad_400/600/800ms ayarlarının tümünde WER %14.89-%15.96 aralığında kalmış, kırpma hatası oluşmamıştır."
        }
    ]

    fieldnames = [
        "basename", "reviewed_range", "manual_finding",
        "speech_present_before_first_segment", "reviewer", "review_method", "decision"
    ]

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Oluşturuldu: {OUT_CSV}")

if __name__ == "__main__":
    create_manual_review_csv()
