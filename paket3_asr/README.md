# Paket 3 — Türkçe ASR (Konuşma → Metin) Seçimi ve Optimizasyonu

Bu dizin, SimClever İntercom projesinin **Paket 3 (Türkçe ASR Seçimi ve Optimizasyonu)** modülüdür.

---

## 📂 Dizin Yapısı

```
paket3_asr/
├── config/
│   ├── models.yaml             # Aday modeller (Whisper large-v3, turbo, medium, faster-whisper int8 vb.)
│   └── experiment.yaml         # Adil kıyas ve normalizasyon parametreleri (language=tr, temp=0)
├── data/
│   ├── videos/                 # İndirilen videolar (opsiyonel - Git'e eklenmez)
│   ├── audio_raw/              # 📥 İndirdiğiniz ham ses kayıtları (.wav / .mp3)
│   ├── audio_16k/              # ⚙️ Otomatik üretilen 16kHz mono standart sesler
│   ├── audio_denoised/         # ⚙️ Otomatik üretilen Paket 2 denoise sesler (A/B kıyası için)
│   ├── silence_noise/          # ⚙️ Halüsinasyon testi için ayrılan sessizlik kesitleri
│   ├── references/             # 📝 Google AI Studio + manuel kontrol ettiğiniz .txt metinleri
│   └── metadata.csv            # 📄 20 Tıbbi Video Etiketli Veri Seti İndeksi
├── models/                     # Kuantize / Fine-tuned Model Ağırlıkları
├── outputs/
│   ├── transcripts/            # Model transkripsiyon çıktıları
│   ├── metrics/                # WER / CER / RTF / Terim İsabet CSV sonuçları
│   └── reports/                # Karar ve karşılaştırma raporları
├── scripts/
│   ├── inventory.py            # P3.0 Donanım tespiti (CPU / GPU / VRAM / RAM)
│   ├── prepare_audio.py        # audio_raw -> 16kHz WAV ve Paket 2 Denoise dönüştürücü
│   ├── segment_audio.py        # VAD ile sessizlik/gürültü kesit ayrıştırma
│   ├── transcribe.py           # Deterministik ASR transkripsiyon modülü
│   ├── normalize_text.py       # Türkçe ASR metin normalizasyonu (yüz yirmi, noktalama temizliği)
│   ├── calculate_metrics.py    # WER (Kelime) + CER (Karakter) hesaplama engine
│   ├── medical_term_score.py   # Tıbbi terim isabeti & halüsinasyon puanlayıcı
│   └── benchmark.py            # A/B (Ham vs Denoise) Orkestratör ve Raporlayıcı
├── requirements.txt            # Paket 3 bağımlılıkları
└── README.md                   # Kullanım & Veri Yükleme Kılavuzu
```

---

## 🚀 Kullanım Adımları (Veri Yükleme ve Test Rehberi)

### Adım 1: Ses Kayıtlarını ve Referans Metinleri Ekleyin

1. **Ses Kayıtları**: İndirdiğiniz 20 adet tıbbi video sesini `paket3_asr/data/audio_raw/` dizinine kopyalayın:
   - `video_01.wav` (veya `.mp3`)
   - `video_02.wav`
   - ...
   - `video_20.wav`

2. **Referans Metinler (Ground Truth)**: Google AI Studio kullanarak çıkardığınız ve manuel kontrol ettiğiniz düz metin dosyalarını `paket3_asr/data/references/` dizinine kopyalayın:
   - `video_01.txt`
   - `video_02.txt`
   - ...
   - `video_20.txt`

---

### Adım 2: Ses Hazırlama ve Denoise Pipeline Çalıştırma

Aşağıdaki komutu çalıştırarak ses dosyalarınızı 16kHz mono WAV formatına dönüştürün ve Paket 2 gürültü bastırma filtresinden geçirin:

```bash
python paket3_asr/scripts/prepare_audio.py
```

---

### Adım 3: Donanım Envanteri Tespiti

Sisteminizin CPU, GPU ve VRAM imkanlarını tespit edip raporlamak için:

```bash
python paket3_asr/scripts/inventory.py
```

---

### Adım 4: Benchmark & Ham vs Denoise (A/B) Testini Çalıştırma

Tüm modelleri adil kıyas protokolü ile (`language=tr`, `temperature=0.0`, WER+CER) test edip raporlamak için:

```bash
python paket3_asr/scripts/benchmark.py
```

---

## 📊 Metrikler ve Kabul Kriterleri

- **WER (Kelime Hata Oranı)**: Ana metrik (literatür kıyası).
- **CER (Karakter Hata Oranı)**: Türkçe gibi sondan eklemeli diller için tamamlayıcı metrik.
- **Tıbbi Terim İsabeti**: Anamnez, tansiyon, nabız, entübasyon vb. terimlerin doğru yazılma oranı.
- **Halüsinasyon Sayısı**: Sessizlik/gürültü kısımlarında modelin uydurduğu kelime sayısı (ideal: 0).
- **RTF (Real-Time Factor)**: 1 dakikalık sesin işlenme süresi (laptop GPU ve CPU üzerinde).
