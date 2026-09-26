import os
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.signal import correlate

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_chirp(sample_rate=16000, duration=0.1, f0=500, f1=3000):
    """
    Korelasyon analizi için 100ms'lik bir linear chirp (frekans süpürme) sinyali üretir.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    # Linear chirp formülü: sin(2*pi * (f0*t + 0.5 * (f1-f0)/T * t^2))
    chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2))
    # Pencereleme yaparak baş ve son geçişleri yumuşat (tıklama sesini engellemek için)
    window = np.hanning(len(t))
    return chirp * window

def measure_local_loopback(sample_rate=16000, duration_sec=2.0):
    """
    Hoparlörden chirp çalar ve aynı anda mikrofondan kayıt alarak
    yerel ses kartı + sürücü (Mouth-to-Ear / Round-Trip) gecikmesini ölçer.
    """
    print("\n=== YEREL DONANIM DÖNGÜSÜ GECİKME ÖLÇÜMÜ (Hardware Loopback) ===", flush=True)
    print("Hoparlör ve mikrofonunuzun birbirine yakın olduğundan emin olun.", flush=True)
    print("Ölçüm başlıyor...", flush=True)
    
    # 100ms chirp üret
    chirp = generate_chirp(sample_rate)
    
    # Sessizlik ve chirp birleşimi olan çalma tamponu oluştur
    # Toplam çalma süresi içinde chirp 0.5. saniyede başlasın
    total_samples = int(duration_sec * sample_rate)
    play_buffer = np.zeros(total_samples, dtype=np.float32)
    start_idx = int(0.5 * sample_rate)
    play_buffer[start_idx:start_idx + len(chirp)] = chirp
    
    # Çalma ve kaydetmeyi aynı anda başlat
    try:
        recorded = sd.playrec(play_buffer, samplerate=sample_rate, channels=1, blocking=True)
    except Exception as e:
        print(f"[Hata] Ses aygıtı başlatılamadı: {e}", flush=True)
        print("Lütfen ses aygıtınızın bağlı ve aktif olduğunu kontrol edin.", flush=True)
        return None
        
    recorded = recorded.flatten()
    
    # Çapraz Korelasyon (Cross-Correlation) hesaplama
    # Kaydedilen sinyal ile orijinal chirp arasındaki korelasyonu bul
    correlation = correlate(recorded, chirp, mode='valid')
    
    # En yüksek korelasyonun indeksini bul (chirp'in yakalandığı an)
    max_idx = np.argmax(np.abs(correlation))
    
    # Gecikme süresi = (indeks - start_idx) / sample_rate
    # start_idx: chirp'ün gönderildiği an (0.5. saniye)
    delay_samples = max_idx - start_idx
    delay_ms = (delay_samples / sample_rate) * 1000.0
    
    print(f"En yüksek korelasyon indeksi: {max_idx} (Örnek)")
    print(f"Gönderim indeksi: {start_idx} (Örnek)")
    print(f"Fark (Örnek sayısı): {delay_samples}")
    
    if delay_ms < 0:
        print("[Hata] Negatif gecikme hesaplandı! Korelasyon yanlış bir gürültüye kilitlenmiş olabilir.", flush=True)
        return None
        
    print(f"Hesaplanan Yerel Gecikme (Round-Trip): {delay_ms:.2f} ms", flush=True)
    return delay_ms

if __name__ == "__main__":
    # Ses cihazlarını listele
    print("--- Mevcut Ses Cihazları ---")
    print(sd.query_devices())
    
    # Varsayılan aygıtları göster
    print(f"\nVarsayılan Giriş/Çıkış Aygıtları: {sd.default.device}")
    
    measure_local_loopback()
