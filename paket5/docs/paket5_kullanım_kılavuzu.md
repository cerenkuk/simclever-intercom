# Paket 5 Kullanım Kılavuzu

Bu belge SimClever Intercom projesinin **Paket 5** için kurulum ve kullanım talimatlarını içerir.

## Gereksinimler
- Windows 10/11
- Python 3.13+ (venv içinde çalıştırılması önerilir)
- `PyQt5` (pip ile kurulabilir)
- `sounddevice`, `soundfile`, `webrtcvad` gibi ses işleme bağımlılıkları (requirements.txt içinde listelenmiştir)

## Kurulum
```bash
# Proje kök dizininde
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma
```bash
python run_intercom_p5.py
```

## Konfigürasyon
`config/teacher.json` ve `config/student.json` dosyaları yapılandırma ayarlarını barındırır. **Gürültü filtresi** varsayılan olarak `webrtc_mild` (Düşük) seçilmiştir. UI üzerinden `Ayarlar → Ses Filtre Ayarları` menüsüyle değiştirilebilir.

## Kayıt (Recording)
Kayıt aktif olduğunda `recordings/` klasörüne üç ayrı WAV dosyası ve `sidecar.json` meta‑verisi oluşturulur.

## Test Çalıştırma
```bash
pytest tests/paket5 -v
```

---
*Bu doküman sadece örnek amaçlıdır; projenin tam işlevselliği `README.md` içinde de bulunabilir.*
