import os
import sys
import wave
import numpy as np

def calculate_erle(raw_wav_path, clean_wav_path, ref_wav_path, threshold=0.005):
    """
    Ham mikrofon (raw), temizlenmiş mikrofon (clean) ve hoparlör referans (ref)
    WAV dosyalarını okuyarak, hoparlörün aktif olduğu anlardaki
    Akustik Yankı Bastırma Oranını (ERLE - Echo Return Loss Enhancement) desibel (dB) cinsinden hesaplar.
    """
    print("\n=== GERÇEK ZAMANLI AKUSTİK YANKI SÖNÜMLEME (ERLE) ÖLÇÜMÜ ===")
    
    if not (os.path.exists(raw_wav_path) and os.path.exists(clean_wav_path) and os.path.exists(ref_wav_path)):
        print(f"[Hata] Dosyalar bulunamadı! Lütfen yolları kontrol edin.", flush=True)
        return
        
    # WAV dosyalarını aç
    w_raw = wave.open(raw_wav_path, 'rb')
    w_clean = wave.open(clean_wav_path, 'rb')
    w_ref = wave.open(ref_wav_path, 'rb')
    
    # Parametreleri kontrol et
    params = w_raw.getparams()
    n_frames = min(w_raw.getnframes(), w_clean.getnframes(), w_ref.getnframes())
    
    # Okuma yap
    raw_bytes = w_raw.readframes(n_frames)
    clean_bytes = w_clean.readframes(n_frames)
    ref_bytes = w_ref.readframes(n_frames)
    
    w_raw.close()
    w_clean.close()
    w_ref.close()
    
    # Tamsayıya dönüştür
    raw_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    clean_data = np.frombuffer(clean_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    ref_data = np.frombuffer(ref_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Bloklar halinde incele (10ms blocks = 160 samples at 16kHz)
    block_size = 160
    n_blocks = len(raw_data) // block_size
    
    erle_list = []
    
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        
        raw_block = raw_data[start:end]
        clean_block = clean_data[start:end]
        ref_block = ref_data[start:end]
        
        # Referans sinyalinin (hoparlör sesinin) gücünü ölç (RMS)
        ref_rms = np.sqrt(np.mean(ref_block**2))
        
        # Eğer hoparlör aktifse (yani odaya yankı yayıldıysa) ERLE hesapla
        if ref_rms > threshold:
            raw_energy = np.sum(raw_block**2)
            clean_energy = np.sum(clean_block**2)
            
            if clean_energy > 1e-8:
                # ERLE = 10 * log10( Ham Mikrofon Gücü / Temizlenmiş Mikrofon Gücü )
                erle_db = 10 * np.log10(raw_energy / clean_energy)
                # Anlamlı sınırlar içerisinde filtrele (aşırı uç değerleri önlemek için)
                if erle_db > 0:
                    erle_list.append(erle_db)
                    
    if len(erle_list) == 0:
        print("[Hata] Ölçüm için yeterli düzeyde aktif hoparlör sesi tespit edilemedi!", flush=True)
        print("Lütfen hoca veya öğrenci tarafında ses seviyesinin açık olduğundan emin olun.", flush=True)
        return
        
    avg_erle = np.mean(erle_list)
    max_erle = np.max(erle_list)
    
    print(f"İncelenen Aktif Yankı Bloğu Sayısı: {len(erle_list)}")
    print(f"Maksimum Ölçülen AEC Sönümlemesi: {max_erle:.2f} dB")
    print(f"ORTALAMA AKUSTİK YANKI ENGELLEME (ERLE): {avg_erle:.2f} dB", flush=True)
    
    if avg_erle >= 25.0:
        print("[BAŞARILI] ERLE >= 25 dB geçme kriteri sağlanmıştır.", flush=True)
    else:
        print("[UYARI] Ortalama ERLE 25 dB'in altındadır. Hoparlör sesini arttırarak testi tekrarlayabilirsiniz.", flush=True)
        
    return avg_erle

if __name__ == "__main__":
    # Örnek kullanım için varsayılan dosyaları kontrol et
    raw_path = 'student_raw_mic.wav'
    clean_path = 'student_cleaned_mic.wav'
    ref_path = 'student_processed_output.wav'
    
    if len(sys.argv) > 3:
        raw_path = sys.argv[1]
        clean_path = sys.argv[2]
        ref_path = sys.argv[3]
        
    calculate_erle(raw_path, clean_path, ref_path)
