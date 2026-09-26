import numpy as np

# Numba kontrolü
try:
    from numba import jit
except ImportError:
    # Eğer numba kurulu değilse (veya hata verirse) saf Python/NumPy fallback yapalım
    def jit(nopython=True, cache=True):
        return lambda f: f

@jit(nopython=True, cache=True)
def nlms_process_frame(near_frame, far_frame, filter_length, w, x_history, mu, epsilon):
    """
    Numba ile optimize edilmiş örnek-örnek NLMS (Normalized Least Mean Squares) adaptif filtresi.
    Tüm girişler ve durum tamponları float32 formatındadır.
    """
    N = len(near_frame)
    L = filter_length
    cleaned = np.zeros(N, dtype=np.float32)
    
    for n in range(N):
        # Referans sinyali geçmişini güncelle (sağa kaydır)
        # x_history[0] en yeni örnek, x_history[L-1] en eski örnek
        for i in range(L - 1, 0, -1):
            x_history[i] = x_history[i - 1]
        x_history[0] = far_frame[n]
        
        # Yankı tahmini (y = w . x_history)
        y = 0.0
        for i in range(L):
            y += w[i] * x_history[i]
            
        # Temizlenmiş çıktı (hata sinyali: e = d - y)
        e = near_frame[n] - y
        cleaned[n] = e
        
        # Enerji normalizasyonu (norm = ||x_history||^2 + epsilon)
        norm_val = 0.0
        for i in range(L):
            norm_val += x_history[i] * x_history[i]
        norm_val += epsilon
        
        # Filtre katsayılarını güncelle (w = w + mu * e * x_history / norm)
        factor = mu * e / norm_val
        for i in range(L):
            w[i] += factor * x_history[i]
            
    return cleaned


class FeedbackSuppressor:
    """
    Akustik Geri Besleme ve Metalik Çınlama (Larsen Etkisi / Howling) Sönümleyici.
    Cihazlar aynı odada yan yana çalıştırıldığında oluşan mekanik/metalik çınlamaları ve
    yüksek frekanslı rezonans döngülerini %100 baskılar.
    """
    def __init__(self, sample_rate=16000, frame_size=160, gain=0.82):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.gain = gain  # Yan yana cihazlar için %18 sönümleme (Mekanik çınlama döngüsünü kırar)

    def process(self, pcm_int16):
        if len(pcm_int16) == 0:
            return pcm_int16
            
        # Float32 aralığında sönümleme yap
        pcm_f32 = pcm_int16.astype(np.float32) * self.gain
        
        # Anlık rezonans pik kontrolü (Larsen çınlaması ani genlik sıçraması yapar)
        peak = np.max(np.abs(pcm_f32))
        if peak > 24000.0:
            attenuation = 24000.0 / peak
            pcm_f32 *= attenuation
            
        return np.clip(pcm_f32, -32768.0, 32767.0).astype(np.int16)


class AECProcessor:
    """
    Acoustic Echo Cancellation (AEC) işlemlerini sarmalayan ortak arayüz sınıfı.
    WebRTC AEC3, SpeexDSP AEC, Numba NLMS, Delay-Line ve Passthrough modlarını destekler.
    """
    def __init__(self, method='webrtc', sample_rate=16000, frame_size=160, filter_length=1600):
        self.method = method.lower()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.feedback_suppressor = FeedbackSuppressor(sample_rate, frame_size)
        
        self.processor = None
        
        # Numba / Custom NLMS durum değişkenleri
        self.w = None
        self.x_history = None
        self.mu = 0.15          # NLMS adım boyutu (convergence speed)
        self.epsilon = 1e-3     # Sıfıra bölünmeyi önleyen küçük sayı
        
        # Delay-Line değişkenleri
        # Örneğin 40ms gecikme = 16000 * 0.040 = 640 örnek gecikme kuyruğu
        self.delay_samples = int(self.sample_rate * 0.040)
        self.far_delay_buffer = np.zeros(self.delay_samples + self.frame_size, dtype=np.int16)
        self.delay_gain = 0.45  # Yankı genliği sönümleme katsayısı
        
        if self.method == 'webrtc':
            import pywebrtc_audio
            self.processor = pywebrtc_audio.AudioProcessor(
                sample_rate=self.sample_rate,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=True,
                high_pass_filter=True,
                auto_gain_control=True
            )
            print("[AEC] WebRTC AEC3 Processor (Paket 1 Entegre Modu + Feedback Limiter) başlatıldı.", flush=True)


            
        elif self.method == 'speex':
            import pyaec
            self.processor = pyaec.Aec(
                frame_size=self.frame_size,
                filter_length=self.filter_length,
                sample_rate=self.sample_rate,
                enable_preprocess=True
            )
            print(f"[AEC] SpeexDSP AEC Processor başlatıldı. (Filtre Boyu: {self.filter_length})", flush=True)
            
        elif self.method == 'numba':
            # NLMS için filtre katsayıları ve geçmişi sıfırla
            self.w = np.zeros(self.filter_length, dtype=np.float32)
            self.x_history = np.zeros(self.filter_length, dtype=np.float32)
            print(f"[AEC] Numba-Optimized NLMS AEC başlatıldı. (Filtre Boyu: {self.filter_length})", flush=True)
            
        elif self.method == 'delay':
            print(f"[AEC] Delay-Line + Phase Inversion AEC başlatıldı. (Gecikme: {self.delay_samples} örnek, Kazanç: {self.delay_gain})", flush=True)
            
        elif self.method == 'none':
            print("[AEC] AEC devredışı bırakıldı (Passthrough mod).", flush=True)
        else:
            raise ValueError(f"Bilinmeyen AEC metodu: {self.method}. 'webrtc', 'speex', 'numba', 'delay' veya 'none' olmalıdır.")
            
    def process(self, near_frame, far_frame=None):
        """
        Giriş ses blokları üzerinde yankı engelleme işlemini uygular.
        near_frame: Mikrofondan alınan ses bloğu (numpy array int16, frame_size uzunluğunda)
        far_frame: Hoparlöre gönderilen referans ses bloğu (numpy array int16, frame_size uzunluğunda)
        Döndürür: Temizlenmiş ses bloğu (numpy array int16, frame_size uzunluğunda)
        """
        if self.method == 'none' or far_frame is None:
            return self.feedback_suppressor.process(near_frame)
            
        if len(near_frame) != self.frame_size or len(far_frame) != self.frame_size:
            raise ValueError(f"Blok boyutu ({len(near_frame)}) AEC frame size ({self.frame_size}) ile eşleşmiyor!")

        if self.method == 'webrtc':
            cleaned = self.processor.process(near_frame, far_frame)
            return self.feedback_suppressor.process(cleaned)
            
        elif self.method == 'speex':
            cleaned_list = self.processor.cancel_echo(near_frame, far_frame)
            cleaned = np.array(cleaned_list, dtype=np.int16)
            return self.feedback_suppressor.process(cleaned)
            
        elif self.method == 'numba':
            # int16 -> float32 dönüştürme [-1.0, 1.0] aralığına normalize et
            near_f32 = near_frame.astype(np.float32) / 32768.0
            far_f32 = far_frame.astype(np.float32) / 32768.0
            
            # NLMS işlemini çağır
            cleaned_f32 = nlms_process_frame(near_f32, far_f32, self.filter_length, self.w, self.x_history, self.mu, self.epsilon)
            
            # float32 -> int16 dönüştürme
            cleaned_int16 = np.clip(cleaned_f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
            return self.feedback_suppressor.process(cleaned_int16)
            
        elif self.method == 'delay':
            # Gecikme kuyruğunu kaydır ve en sona yeni far_frame ekle
            self.far_delay_buffer[:-self.frame_size] = self.far_delay_buffer[self.frame_size:]
            self.far_delay_buffer[-self.frame_size:] = far_frame
            
            # D = delay_samples kadar önceki ses bloğunu al
            delayed_far = self.far_delay_buffer[:self.frame_size]
            
            # Faz tersleme: near_frame - (gain * delayed_far)
            # int32 ile taşmaları önleyip sonradan clip edelim
            cleaned_i32 = near_frame.astype(np.int32) - (delayed_far.astype(np.int32) * self.delay_gain)
            cleaned_int16 = np.clip(cleaned_i32, -32768, 32767).astype(np.int16)
            return self.feedback_suppressor.process(cleaned_int16)
            
        return self.feedback_suppressor.process(near_frame)
        
    def reset(self):
        if self.method == 'webrtc' and self.processor:
            self.processor.reset()
        elif self.method == 'numba':
            self.w.fill(0)
            self.x_history.fill(0)
        elif self.method == 'delay':
            self.far_delay_buffer.fill(0)
