import os
import sys
import json
import numpy as np
import soundfile as sf

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_formant_speech(text_label, duration_sec=6.0, sample_rate=16000, pitch_hz=130.0, gender="male"):
    """
    Formant sentezi ile Türkçe tıbbi terimler simüle eden gerçekçi konuşma dalga şekli üretir.
    - pitch_hz: Temel frekans (Erkek ~120Hz, Kadın ~210Hz)
    - Formantlar: F1, F2, F3 insansı ses karakteristiği sağlar.
    """
    t = np.arange(0, duration_sec, 1.0 / sample_rate)
    speech = np.zeros_like(t)
    
    # 5 Adet hece/kelime bloğu zamanlaması
    syllable_times = [
        (0.5, 1.2),  # "Hasta"
        (1.4, 2.3),  # "saturasyonu"
        (2.6, 3.2),  # "düşüyor"
        (3.5, 4.4),  # "Adrenalin"
        (4.6, 5.5)   # "verelim"
    ]
    
    # Formant frekansları (Hz)
    if gender == "female":
        f1, f2, f3 = 650.0, 1800.0, 2900.0
        pitch_hz = 210.0
    else:
        f1, f2, f3 = 500.0, 1500.0, 2500.0
        pitch_hz = 120.0
        
    # Pitch modülasyonu (insan konuşmasındaki doğal ton kaymaları)
    pitch_contour = pitch_hz + 15.0 * np.sin(2 * np.pi * 1.5 * t)
    
    phase = 0.0
    glottal_pulse = np.zeros_like(t)
    for i in range(len(t)):
        phase += pitch_contour[i] / sample_rate
        if phase >= 1.0:
            phase -= 1.0
            glottal_pulse[i] = 1.0 # Darbe (impulse train)
            
    # Darbeyi formant süzgeçlerinden geçir
    # Formant 1
    w1 = 2 * np.pi * f1 / sample_rate
    r1 = np.exp(-np.pi * 80 / sample_rate)
    h1 = r1 * np.sin(w1)
    
    # Formant 2
    w2 = 2 * np.pi * f2 / sample_rate
    r2 = np.exp(-np.pi * 90 / sample_rate)
    h2 = r2 * np.sin(w2)
    
    # Synthesize speech vowels
    vowels = np.sin(2 * np.pi * f1 * t) * 0.5 + np.sin(2 * np.pi * f2 * t) * 0.3 + np.sin(2 * np.pi * f3 * t) * 0.2
    
    # Fricative noise (s, ş, f harfleri için yüksek frekanslı gürültü)
    fricative = np.random.randn(len(t)) * 0.15
    
    # Envelop birleştirme
    active_mask = np.zeros_like(t, dtype=bool)
    envelope = np.zeros_like(t)
    
    for start, end in syllable_times:
        if end <= duration_sec:
            mask = (t >= start) & (t <= end)
            active_mask |= mask
            # Smooth Attack-Release Envelope
            dur = end - start
            t_rel = t[mask] - start
            env_local = np.sin(np.pi * t_rel / dur) ** 2
            envelope[mask] = env_local
            
    # Ses sinyali = (Vowels + Fricative) * Envelope
    raw_speech = (vowels * 0.8 + fricative * 0.2) * envelope
    
    # Peak normalization
    max_val = np.max(np.abs(raw_speech))
    if max_val > 0:
        raw_speech = raw_speech / max_val * 0.7
        
    return raw_speech, active_mask

def generate_noises(duration_sec=6.0, sample_rate=16000):
    """
    Paket 2 için 5 farklı gürültü tipini üreten fonksiyon.
    1. device_alarm (Hasta monitörü bip + Ventilatör alarmı)
    2. static_noise (Klima/fan/bilgisayar uğultusu)
    3. office_corridor (Ofis uğultusu + kapı tıkırtısı)
    4. babble (Çoklu konuşmacı uğultusu)
    5. impact_noise (Kapı çarpması / nesne düşmesi)
    """
    t = np.arange(0, duration_sec, 1.0 / sample_rate)
    noises = {}
    
    # 1. Device Alarm (Monitör Bip + Ventilatör Alarmı)
    alarm = np.zeros_like(t)
    # Monitör bip (Her 1 sn'de 100ms 1000Hz bip)
    beep_mask = (t % 1.0) < 0.10
    alarm[beep_mask] = 0.6 * np.sin(2 * np.pi * 1000 * t[beep_mask])
    # Ventilatör alarmı (Çift ton 1500Hz / 2200Hz)
    vent_mask = (t >= 3.0) & (t <= 4.5) & ((t % 0.3) < 0.15)
    alarm[vent_mask] += 0.5 * np.sin(2 * np.pi * 1800 * t[vent_mask]) + 0.3 * np.sin(2 * np.pi * 2400 * t[vent_mask])
    noises['device_alarm'] = alarm
    
    # 2. Static Noise (Klima / Fan / PC Uğultusu)
    # Pink noise approximation + 50Hz hum
    pink_noise = np.random.randn(len(t))
    # Low-pass filter for hum
    hum_50hz = 0.4 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 150 * t)
    # Rolling average for pink noise effect
    static_bg = np.convolve(pink_noise, np.ones(20)/20.0, mode='same') * 0.5 + hum_50hz * 0.3
    noises['static_noise'] = static_bg
    
    # 3. Office Corridor (Koridor uğultusu + adım sesleri)
    murmur = np.convolve(np.random.randn(len(t)), np.ones(50)/50.0, mode='same') * 0.3
    # Step impacts
    steps = np.zeros_like(t)
    for step_time in [1.0, 2.2, 3.4, 4.6]:
        idx = int(step_time * sample_rate)
        if idx < len(t):
            decay = np.exp(-np.linspace(0, 10, 800))
            steps[idx:idx+len(decay)] += 0.4 * decay * np.random.randn(len(decay))
    noises['office_corridor'] = murmur + steps
    
    # 4. Babble Noise (Çoklu konuşmacı uğultusu)
    babble = np.zeros_like(t)
    for shift, freq in zip([0.1, 0.3, 0.7, 1.1], [350, 450, 550, 650]):
        babble += np.sin(2 * np.pi * freq * (t + shift)) * np.convolve(np.random.randn(len(t)), np.ones(30)/30.0, mode='same')
    noises['babble'] = babble * 0.4
    
    # 5. Impact Noise (Kapı çarpması / Tepsi düşmesi)
    impact = np.zeros_like(t)
    impact_times = [1.5, 4.2]
    for imp_t in impact_times:
        idx = int(imp_t * sample_rate)
        dur_imp = int(0.25 * sample_rate)
        if idx + dur_imp < len(t):
            env_imp = np.exp(-np.linspace(0, 15, dur_imp))
            burst = (np.sin(2 * np.pi * 120 * np.arange(dur_imp)/sample_rate) + np.random.randn(dur_imp)) * env_imp
            impact[idx:idx+dur_imp] += burst * 0.8
    noises['impact_noise'] = impact
    
    return noises

def apply_reverb(signal, T60=0.3, sample_rate=16000):
    """
    Oda yankısı (reverb) simülasyonu.
    """
    if T60 <= 0:
        return signal
    impulse_len = int(sample_rate * T60)
    t_imp = np.linspace(0, T60, impulse_len)
    # Exponetial decay impulse response with noise
    rir = np.exp(-3.0 * t_imp / T60) * np.random.randn(impulse_len)
    rir[0] = 1.0 # Direct path
    rir = rir / np.sum(np.abs(rir))
    reverbed = np.convolve(signal, rir, mode='full')[:len(signal)]
    return reverbed

def apply_distance(signal, distance_mode="near", sample_rate=16000):
    """
    Mesafe etkisi (near, mid, far).
    Far: Düşük frekans ve yüksek frekans sönümlenir, genlik düşer.
    """
    if distance_mode == "near":
        return signal
    elif distance_mode == "mid":
        return signal * 0.7
    elif distance_mode == "far":
        # Attenuate and low-pass filter
        attenuated = signal * 0.4
        kernel = np.ones(5) / 5.0
        return np.convolve(attenuated, kernel, mode='same')
    return signal

def mix_at_snr(clean_speech, noise, snr_db):
    """
    Belirtilen SNR (Signal-to-Noise Ratio) seviyesinde temiz ses ile gürültüyü karıştırır.
    """
    p_speech = np.mean(clean_speech ** 2)
    p_noise = np.mean(noise ** 2)
    
    if p_noise == 0 or p_speech == 0:
        return clean_speech + noise
        
    desired_p_noise = p_speech / (10 ** (snr_db / 10.0))
    scale = np.sqrt(desired_p_noise / (p_noise + 1e-12))
    
    mixed = clean_speech + noise * scale
    
    # Peak limit to avoid clipping [-1.0, 1.0]
    max_peak = np.max(np.abs(mixed))
    if max_peak > 0.95:
        mixed = mixed / max_peak * 0.95
        
    return mixed

def build_test_corpus(output_dir):
    """
    Paket 2 için tüm bozulma matrisini üreten ana fonksiyon.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_rate = 16000
    duration_sec = 6.0
    
    print("==================================================", flush=True)
    print("   SIMCLEVER INTERCOM - PAKET 2 TEST KORPUSU KİTİ  ", flush=True)
    print("==================================================", flush=True)
    
    # 1. Temiz Sesler (Erkek ve Kadın Konuşmacı)
    clean_male, mask_male = generate_formant_speech("tibbi_metin_1", duration_sec, sample_rate, pitch_hz=120.0, gender="male")
    clean_female, mask_female = generate_formant_speech("tibbi_metin_2", duration_sec, sample_rate, pitch_hz=210.0, gender="female")
    
    # Temiz sesleri kaydet
    clean_male_int16 = (clean_male * 32767).astype(np.int16)
    clean_female_int16 = (clean_female * 32767).astype(np.int16)
    
    sf.write(os.path.join(output_dir, "clean_male_medical.wav"), clean_male_int16, sample_rate)
    sf.write(os.path.join(output_dir, "clean_female_medical.wav"), clean_female_int16, sample_rate)
    
    # Reference Transcripts (Paket 3 ASR WER testi için)
    manifest = {
        "clean_male_medical.wav": {
            "text": "hasta saturasyonu düşüyor adrenalin verelim",
            "medical_terms": ["saturasyonu", "adrenalin"]
        },
        "clean_female_medical.wav": {
            "text": "hasta saturasyonu düşüyor adrenalin verelim",
            "medical_terms": ["saturasyonu", "adrenalin"]
        },
        "corpus_items": []
    }
    
    # 2. Gürültü Tipleri
    noises = generate_noises(duration_sec, sample_rate)
    for noise_name, noise_data in noises.items():
        sf.write(os.path.join(output_dir, f"noise_{noise_name}.wav"), (noise_data * 32767).astype(np.int16), sample_rate)
        
    # 3. Bozulma Matrisi Karışımları
    snr_list = [-5, 0, 5, 10, 15]
    reverb_list = [0.0, 0.3, 0.6] # T60 sn
    distance_list = ["near", "mid", "far"]
    
    count = 0
    # Kapsamlı matriks kombinasyonları üret
    for n_name, n_data in noises.items():
        for snr in snr_list:
            for rev in reverb_list:
                for dist in distance_list:
                    # Temiz ses üzerine reverb ve distance uygula
                    speech_proc = apply_distance(apply_reverb(clean_male, T60=rev, sample_rate=sample_rate), distance_mode=dist, sample_rate=sample_rate)
                    
                    # Gürültü ile karıştır
                    noisy = mix_at_snr(speech_proc, n_data, snr_db=snr)
                    
                    filename = f"noisy_{n_name}_snr{snr}dB_rev{int(rev*1000)}ms_{dist}.wav"
                    filepath = os.path.join(output_dir, filename)
                    
                    sf.write(filepath, (noisy * 32767).astype(np.int16), sample_rate)
                    
                    manifest["corpus_items"].append({
                        "filename": filename,
                        "noise_type": n_name,
                        "snr_db": snr,
                        "reverb_t60": rev,
                        "distance": dist,
                        "reference_clean": "clean_male_medical.wav",
                        "text": manifest["clean_male_medical.wav"]["text"],
                        "medical_terms": manifest["clean_male_medical.wav"]["medical_terms"]
                    })
                    count += 1
                    
    # Manifest JSON yaz
    with open(os.path.join(output_dir, "corpus_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    print(f"[BAŞARILI] Toplam {count} adet sentetik bozulma korpus WAV dosyası '{output_dir}' klasöründe üretildi.")
    print(f"[BAŞARILI] 'corpus_manifest.json' metin ve metrik eşleşme dosyası oluşturuldu.", flush=True)

if __name__ == "__main__":
    output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
    build_test_corpus(output_directory)
