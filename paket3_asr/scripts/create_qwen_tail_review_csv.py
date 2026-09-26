import os
import csv

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.abspath(os.path.join(_script_dir, ".."))
OUT_CSV = os.path.join(_pkg_root, "outputs", "metrics", "qwen_tail_manual_review.csv")

def create_qwen_tail_manual_review_csv():
    records = [
        {
            "basename": "video_01_genel_anamnez",
            "start_seconds": "1020.0",
            "end_seconds": "1023.5",
            "audio_content": "non_speech_outro_music",
            "speech_present": "false",
            "reviewer": "human_reviewer",
            "method": "manual_listening",
            "decision": "1020.0-1023.5s aralığında anlaşılır konuşma yoktur, jenerik kapanış müziği mevcuttur. Çıktının boş kalması konuşma olmaması nedeniyle doğrudur. Son remainder parça (<15s) önceki temel chunk ile birleştirilmeli, bağımsız <15s son chunk oluşturulmamalıdır."
        }
    ]

    fieldnames = [
        "basename", "start_seconds", "end_seconds", "audio_content",
        "speech_present", "reviewer", "method", "decision"
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Oluşturuldu: {OUT_CSV}")

if __name__ == "__main__":
    create_qwen_tail_manual_review_csv()
