import os
import urllib.request
import numpy as np
# webrtcvad is imported lazily inside VADProcessor.__init__ when method='webrtc'
# so that the module loads cleanly even if webrtcvad is not installed.

class VADProcessor:
    """
    Ses Aktivitesi Algılama (VAD) sınıfı.
    WebRTC VAD, RMS, ZCR+RMS, Spectral Flux ve Silero VAD (ONNX) metotlarını destekler.
    """
    def __init__(self, method='webrtc', mode=3, sample_rate=16000, frame_size=160):
        self.method = method.lower()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.mode = mode
        
        # WebRTC VAD – yalnızca method='webrtc' seçildiğinde başlat
        self.webrtc_vad = None
        
        # RMS & ZCR Parametreleri
        self.rms_threshold_db = -40.0  # Konuşma eşiği (dB)
        self.zcr_min = 0.05            # Minimum Zero Crossing Rate (sessizliği/durağan gürültüyü elemek için)
        self.zcr_max = 0.60            # Maksimum Zero Crossing Rate (tiz ıslıkları/howling'i elemek için)
        
        # Spectral Flux değişkenleri
        self.prev_spectrum = None
        self.flux_threshold = 0.08     # Spektral akış değişim eşiği
        
        # Silero VAD değişkenleri
        self.silero_session = None
        self.silero_h = None
        self.silero_c = None
        self.silero_history = np.zeros(512, dtype=np.float32) # 32ms geçmiş tamponu (v4 ONNX için)
        
        if self.method == 'webrtc':
            try:
                import webrtcvad as _wvad
                self.webrtc_vad = _wvad.Vad()
                self.webrtc_vad.set_mode(self.mode)
                print(f"[VAD] WebRTC VAD başlatıldı (Mod: {self.mode}).", flush=True)
            except ModuleNotFoundError:
                print("[VAD] webrtcvad bulunamadı – VAD 'none' moduna düşüyor.", flush=True)
                self.method = 'none'

        elif self.method == 'rms':
            print(f"[VAD] RMS Enerji tabanlı VAD başlatıldı (Eşik: {self.rms_threshold_db} dB).", flush=True)
            
        elif self.method == 'zcr_rms':
            print(f"[VAD] ZCR + RMS Karma VAD başlatıldı (RMS Eşik: {self.rms_threshold_db} dB, ZCR: {self.zcr_min}-{self.zcr_max}).", flush=True)
            
        elif self.method == 'spectral_flux':
            print(f"[VAD] Spectral Flux Frekans tabanlı VAD başlatıldı.", flush=True)
            
        elif self.method == 'silero':
            import onnxruntime as ort
            # Silero model dosyasını kontrol et, yoksa indir
            model_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(model_dir, "silero_vad.onnx")
            
            if not os.path.exists(self.model_path):
                print("[VAD] Silero VAD ONNX modeli indiriliyor...", flush=True)
                url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
                try:
                    urllib.request.urlretrieve(url, self.model_path)
                    print("[VAD] Silero VAD modeli başarıyla indirildi.", flush=True)
                except Exception as e:
                    print(f"[VAD] Model indirme hatası: {e}. WebRTC VAD'a fallback yapılıyor.", flush=True)
                    self.method = 'webrtc'
                    return
            
            # ONNX oturumunu başlat (CPU üzerinden çalıştırmak üzere)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.silero_session = ort.InferenceSession(self.model_path, sess_options=opts, providers=['CPUExecutionProvider'])
            
            # LSTM durumlarını sıfırla
            self.reset_silero_state()
            print("[VAD] Silero ONNX Deep Learning VAD başlatıldı.", flush=True)
            
        elif self.method == 'none':
            print("[VAD] VAD devredışı bırakıldı (Sürekli iletim mod).", flush=True)
        else:
            raise ValueError(f"Bilinmeyen VAD metodu: {self.method}. 'webrtc', 'rms', 'zcr_rms', 'spectral_flux', 'silero' veya 'none' olmalıdır.")

    def reset_silero_state(self):
        # Silero VAD v4 LSTM durumunu sıfırlar
        self.silero_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.silero_history.fill(0)

    def calculate_rms_db(self, pcm_frame):
        """
        Sinyalin RMS (Root Mean Square) enerjisini dB olarak hesaplar.
        """
        if len(pcm_frame) == 0:
            return -100.0
        # float dönüşümü yapıp hesaplayalım
        rms = np.sqrt(np.mean(pcm_frame.astype(np.float64)**2))
        if rms < 1e-5:
            return -100.0
        return 20 * np.log10(rms)

    def is_speech(self, pcm_frame):
        """
        Giriş ses bloğunda insan sesi olup olmadığını algılar.
        pcm_frame: numpy array (int16, frame_size uzunluğunda)
        Döndürür: True (konuşma var) veya False (sessizlik/gürültü)
        """
        if self.method == 'none':
            return True
            
        if isinstance(pcm_frame, bytes):
            pcm_frame = np.frombuffer(pcm_frame, dtype=np.int16)

        if len(pcm_frame) != self.frame_size:
            # Boyut eşleşmezse webrtcvad hata verir, güvenli modda True dönelim
            return True

        # 1. WebRTC VAD
        if self.method == 'webrtc':
            pcm_bytes = pcm_frame.tobytes()
            try:
                return self.webrtc_vad.is_speech(pcm_bytes, self.sample_rate)
            except Exception as e:
                print(f"[VAD] WebRTC VAD hatası: {e}", flush=True)
                return True

        # 2. RMS Enerji VAD
        elif self.method == 'rms':
            rms_db = self.calculate_rms_db(pcm_frame)
            return rms_db > self.rms_threshold_db

        # 3. ZCR + RMS VAD
        elif self.method == 'zcr_rms':
            rms_db = self.calculate_rms_db(pcm_frame)
            if rms_db <= self.rms_threshold_db:
                return False
                
            # Zero Crossing Rate hesapla (işaret değişim oranı)
            zero_crossings = np.sum(np.diff(np.sign(pcm_frame)) != 0)
            zcr = zero_crossings / len(pcm_frame)
            
            # ZCR makul sınırlar içindeyse konuşmadır (hiss ve howling elenir)
            return self.zcr_min <= zcr <= self.zcr_max

        # 4. Spectral Flux Frekans VAD
        elif self.method == 'spectral_flux':
            # FFT alıp genlik spektrumunu normalize edelim
            fft_vals = np.abs(np.fft.rfft(pcm_frame.astype(np.float32) / 32768.0))
            fft_sum = np.sum(fft_vals)
            
            if fft_sum < 1e-4:
                return False
                
            fft_norm = fft_vals / fft_sum
            
            if self.prev_spectrum is None:
                self.prev_spectrum = fft_norm
                return False
                
            # Spektral akış = sum(max(0, current - prev)^2)
            diff = fft_norm - self.prev_spectrum
            flux = np.sum(np.maximum(0.0, diff) ** 2)
            
            # Bir sonraki sefere sakla
            self.prev_spectrum = fft_norm
            
            # RMS kontrolü ile birleştir (çok sessiz ortamda spektral akış sapıtabilir)
            rms_db = self.calculate_rms_db(pcm_frame)
            return flux > self.flux_threshold and rms_db > -50.0

        # 5. Silero VAD (ONNX Deep Learning)
        elif self.method == 'silero':
            # Geçmiş tamponu kaydır ve yeni ses verisini ekle
            self.silero_history[:-self.frame_size] = self.silero_history[self.frame_size:]
            self.silero_history[-self.frame_size:] = pcm_frame.astype(np.float32) / 32768.0
            
            # Giriş sinyalini [1, 512] boyutunda normalize edilmiş float32 yap
            input_f32 = np.expand_dims(self.silero_history, axis=0) # [1, 512]
            
            # Girdileri hazırla
            ort_inputs = {
                'input': input_f32,
                'state': self.silero_state,
                'sr': np.array(self.sample_rate, dtype=np.int64)
            }
            
            try:
                ort_outs = self.silero_session.run(None, ort_inputs)
                out = ort_outs[0] # speech probability [1, 1]
                self.silero_state = ort_outs[1] # new state
                
                # Olasılık 0.50'den büyükse konuşmadır
                prob = out[0][0]
                return prob > 0.50
            except Exception as e:
                print(f"[VAD] Silero VAD ONNX çalıştırılamadı: {e}", flush=True)
                return True

        return True
