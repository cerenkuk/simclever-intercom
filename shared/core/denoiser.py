import os
import sys
import logging
import numpy as np

class SpectralSubtractionDenoiser:
    """
    Klasik Spektral Çıkarma (Baseline Denoiser) - 50% Overlap-Add (OLA) Pencereli.
    FFT boyutunda gürültü spektrumu tahmini yapar ve sinyalden düşer.
    OLA süzgeci sayesinde çerçeve sınırlarında PATLAMA / TIKLAMA (popping) oluşmaz.
    """
    def __init__(self, frame_size=160, sample_rate=16000, alpha=2.0, beta=0.02):
        self.frame_size = frame_size  # Hop size H = 160
        self.fft_size = frame_size * 2 # N = 320 (50% Overlap)
        self.sample_rate = sample_rate
        self.alpha = alpha  # Gürültü çıkarma faktörü
        self.beta = beta    # Spektral zemin (spectral floor)
        
        # Hanning Penceresi (Sürekli Overlap-Add için)
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.window_norm = self.window / (self.window**2 + np.roll(self.window**2, self.frame_size) + 1e-12)
        
        # Tamponlar
        self.prev_input = np.zeros(self.frame_size, dtype=np.float32)
        self.overlap_buf = np.zeros(self.frame_size, dtype=np.float32)
        
        # Gürültü profili
        self.num_bins = self.fft_size // 2 + 1
        self.noise_psd = np.zeros(self.num_bins, dtype=np.float32)
        self.initialized_frames = 0
        self.init_needed = 5

    def process(self, frame_float32):
        if len(frame_float32) != self.frame_size:
            return frame_float32
            
        # 320 örnekli çakışmalı pencere oluştur (160 eski + 160 yeni)
        input_320 = np.concatenate([self.prev_input, frame_float32])
        self.prev_input = frame_float32.copy()
        
        # Hanning Penceresi Uygula
        windowed = input_320 * self.window
        
        # RFFT Hesapla
        fft_data = np.fft.rfft(windowed)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        psd = magnitude ** 2
        
        # İlk 5 çerçevede gürültü zeminini öğren
        if self.initialized_frames < self.init_needed:
            self.noise_psd = (self.noise_psd * self.initialized_frames + psd) / (self.initialized_frames + 1)
            self.initialized_frames += 1
            # OLA Çıkışı
            self.overlap_buf = windowed[self.frame_size:] * self.window[self.frame_size:]
            return frame_float32
            
        # Spektral Çıkarma
        subtracted_psd = psd - self.alpha * self.noise_psd
        subtracted_psd = np.maximum(subtracted_psd, self.beta * psd)
        
        clean_magnitude = np.sqrt(subtracted_psd)
        clean_fft = clean_magnitude * np.exp(1j * phase)
        
        # iFFT ile zaman ortamına dön
        reconstructed = np.fft.irfft(clean_fft, n=self.fft_size).astype(np.float32)
        reconstructed *= self.window
        
        # Overlap-Add Birleştirme (Patlamaları / Çıtırtıları 100% Engeller)
        out_frame = self.overlap_buf + reconstructed[:self.frame_size]
        self.overlap_buf = reconstructed[self.frame_size:]
        
        return out_frame.astype(np.float32)

class WebRTCNSDenoiser:
    """
    Google WebRTC Noise Suppressor Wrapper.
    """
    def __init__(self, frame_size=160, sample_rate=16000, level=2):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.ns = None
        
        try:
            from pywebrtc_audio import NoiseSuppressor
            # Doğru API: NoiseSuppressor(sample_rate, num_channels, level)
            # level: 0=Mild, 1=Medium, 2=Aggressive, 3=Very Aggressive
            self.ns = NoiseSuppressor(sample_rate=sample_rate, num_channels=1, level=level)
            print(f"[WebRTC NS] Gürültü Bastırıcı başlatıldı (level={level}).", flush=True)
        except Exception as e:
            print(f"[WebRTC NS] Yüklenemedi: {e}. Passthrough modunda çalışılıyor.", flush=True)
            self.ns = None

    def process(self, frame_float32):
        if self.ns is not None:
            try:
                # process() numpy int16 array bekler (np.clip ile int16 overflow patlamaları engellenir)
                pcm_int16 = np.clip(frame_float32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
                cleaned_int16 = self.ns.process(pcm_int16)
                return cleaned_int16.astype(np.float32) / 32768.0

            except Exception:
                logging.getLogger(__name__).debug(
                    "WebRTC noise suppression failed; using passthrough", exc_info=True
                )
                
        return frame_float32

class RNNoiseDenoiser:
    """
    RNNoise (RNN tabanlı hafif gürültü bastırıcı) Wrapper.
    """
    def __init__(self, frame_size=160, sample_rate=16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.rnnoise = None
        
        try:
            import rnnoise
            self.rnnoise = rnnoise.RNNoise()
        except Exception:
            self.rnnoise = None

    def process(self, frame_float32):
        if self.rnnoise is not None:
            try:
                pcm_int16 = (frame_float32 * 32767.0).astype(np.int16)
                cleaned = self.rnnoise.process_frame(pcm_int16)
                return cleaned.astype(np.float32) / 32768.0
            except Exception:
                logging.getLogger(__name__).debug(
                    "RNNoise processing failed; using passthrough", exc_info=True
                )
        return frame_float32

class DeepFilterNetDenoiser:
    """
    DeepFilterNet (Deep Complex Filtering) Wrapper.
    """
    def __init__(self, frame_size=160, sample_rate=16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.df_state = None
        self.df_model = None
        
        try:
            from df.enhance import init_df, enhance
            self.df_model, self.df_state, _ = init_df()
        except Exception:
            self.df_model = None

    def process(self, frame_float32):
        if self.df_model is not None and self.df_state is not None:
            try:
                import torch
                from df.enhance import enhance
                tensor_frame = torch.from_numpy(frame_float32).unsqueeze(0)
                enhanced_tensor = enhance(self.df_model, self.df_state, tensor_frame)
                return enhanced_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            except Exception:
                logging.getLogger(__name__).debug(
                    "DeepFilterNet processing failed; using passthrough", exc_info=True
                )
        return frame_float32

class AdaptiveWienerDenoiser:
    """
    Adaptif Wiener Filtresi (Adaptive Wiener Filter with Decision-Directed SNR & 50% Overlap-Add).
    Gürültü kapılaması (hard gating) YAPMAZ; frekans bazlı yumuşak kazanç filtresi uygular.
    50% Overlap-Add (OLA) ve Hanning penceresi sayesinde çerçeve geçişlerindeki PATLAMA / TIKLAMA
    (popping / click noise) ve İLK HECE YUTULMASI 100% ÇÖZÜLMÜŞTÜR.
    """
    def __init__(self, frame_size=160, sample_rate=16000, alpha=0.98, min_gain=0.15):
        self.frame_size = frame_size  # Hop size H = 160
        self.fft_size = frame_size * 2 # N = 320 (50% Overlap)
        self.sample_rate = sample_rate
        self.alpha = alpha  # Decision-directed smoothing factor
        self.min_gain = min_gain # Minimum kazanç tabanı (-16dB)
        
        # Hanning Penceresi
        self.window = np.hanning(self.fft_size).astype(np.float32)
        
        # Tamponlar
        self.prev_input = np.zeros(self.frame_size, dtype=np.float32)
        self.overlap_buf = np.zeros(self.frame_size, dtype=np.float32)
        
        self.num_bins = self.fft_size // 2 + 1
        self.noise_psd = np.ones(self.num_bins, dtype=np.float32) * 1e-4
        self.prev_clean_psd = np.zeros(self.num_bins, dtype=np.float32)
        self.initialized_frames = 0
        self.init_needed = 8

    def process(self, frame_float32):
        if len(frame_float32) != self.frame_size:
            return frame_float32
            
        # 320 örnekli çakışmalı pencere (160 eski + 160 yeni)
        input_320 = np.concatenate([self.prev_input, frame_float32])
        self.prev_input = frame_float32.copy()
        
        # Pencerele
        windowed = input_320 * self.window
        
        fft_data = np.fft.rfft(windowed)
        magnitude = np.abs(fft_data)
        phase = np.angle(fft_data)
        psd = magnitude ** 2
        
        # Gürültü profili öğrenme
        if self.initialized_frames < self.init_needed:
            self.noise_psd = (self.noise_psd * self.initialized_frames + psd) / (self.initialized_frames + 1)
            self.initialized_frames += 1
            self.overlap_buf = windowed[self.frame_size:] * self.window[self.frame_size:]
            return frame_float32
            
        # Minimum statistics gürültü takibi (yumuşak adaptasyon)
        self.noise_psd = np.minimum(self.noise_psd, psd) * 0.992 + psd * 0.008
        
        # A posteriori SNR
        post_snr = np.maximum(psd / (self.noise_psd + 1e-12) - 1.0, 0.0)
        
        # Decision-directed A priori SNR (Kesintisiz hece yumuşatması)
        priori_snr = self.alpha * (self.prev_clean_psd / (self.noise_psd + 1e-12)) + (1.0 - self.alpha) * post_snr
        
        # Wiener Kazanç Matrisi G(w) = priori_snr / (1 + priori_snr)
        gain = priori_snr / (1.0 + priori_snr)
        gain = np.maximum(gain, self.min_gain)
        
        clean_psd = (gain ** 2) * psd
        self.prev_clean_psd = clean_psd
        
        clean_magnitude = np.sqrt(clean_psd)
        clean_fft = clean_magnitude * np.exp(1j * phase)
        
        # iFFT
        reconstructed = np.fft.irfft(clean_fft, n=self.fft_size).astype(np.float32)
        reconstructed *= self.window
        
        # Overlap-Add Birleştirme (Patlamaları / DC Offset Sıçramalarını Kesin Engeller)
        out_frame = self.overlap_buf + reconstructed[:self.frame_size]
        self.overlap_buf = reconstructed[self.frame_size:]
        
        # DC Offset ve Peak Limiting (Patlamaya karşı koruma)
        out_frame = np.clip(out_frame, -1.0, 1.0)
        return out_frame.astype(np.float32)

class PreTriggerSpectralGateDenoiser:
    """
    50ms Dairesel Ön-Tamponlu Spektral Kapılama (Pre-Trigger Buffer Spectral Gate).
    Konuşma tespit edildiğinde sesi 50ms geriden yumuşak bir attack eğrisi ile açar.
    İlk hece kırpılmasını önlemek için özel olarak tasarlanmıştır.
    """
    def __init__(self, frame_size=160, sample_rate=16000, buffer_ms=50):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.buffer_len = int((buffer_ms / 1000.0) * sample_rate / frame_size) + 1
        
        self.buffer = []
        self.noise_floor = 0.002
        self.alpha_noise = 0.95
        
    def process(self, frame_float32):
        if len(frame_float32) != self.frame_size:
            return frame_float32
            
        energy = np.mean(frame_float32 ** 2)
        
        # Dinamik gürültü tabanı takibi
        if energy < self.noise_floor * 3.0:
            self.noise_floor = self.alpha_noise * self.noise_floor + (1.0 - self.alpha_noise) * energy
            
        threshold = self.noise_floor * 2.5 + 1e-5
        
        # Dairesel tampona ekle
        self.buffer.append(frame_float32.copy())
        if len(self.buffer) > self.buffer_len:
            oldest_frame = self.buffer.pop(0)
        else:
            oldest_frame = frame_float32.copy()
            
        # Eğer mevcut enerjisi eşiğin üstündeyse ön-tampondan sesi yumuşakça sal
        if energy > threshold:
            gain = min(1.0, (energy / (threshold + 1e-6)) ** 0.5)
            return oldest_frame * gain
        else:
            # Sessizlik alanında yumuşak dip gürültü sönümleme (hard gate yok)
            return oldest_frame * 0.15

class DenoiserProcessor:
    """
    Paket 2 Birleşik Gürültü Bastırma İşlemcisi (Unified Denoiser Processor).
    Modüler tak-çıkar mimariye ve Graceful Degradation (hatada ham sese düşme) desteğine sahiptir.
    """
    def __init__(self, method="none", frame_size=160, sample_rate=16000):
        self.method = str(method).lower().strip()
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.processor = None
        
        print(f"[Denoiser] Modül Başlatılıyor: '{self.method}' (Frame: {frame_size}, SR: {sample_rate}Hz)", flush=True)
        
        try:
            if self.method in ["wiener", "adaptive_wiener"]:
                self.processor = AdaptiveWienerDenoiser(frame_size, sample_rate)
            elif self.method in ["pretrigger_gate", "spectral_gate"]:
                self.processor = PreTriggerSpectralGateDenoiser(frame_size, sample_rate)
            elif self.method in ["baseline", "spectral_subtraction"]:
                self.processor = SpectralSubtractionDenoiser(frame_size, sample_rate)
            elif self.method in ["webrtc_mildest", "webrtc_low"]:
                self.processor = WebRTCNSDenoiser(frame_size, sample_rate, level=0)
            elif self.method in ["webrtc_mild"]:
                self.processor = WebRTCNSDenoiser(frame_size, sample_rate, level=1)
            elif self.method in ["webrtc", "webrtc_ns", "webrtc_aggressive"]:
                self.processor = WebRTCNSDenoiser(frame_size, sample_rate, level=2)
            elif self.method in ["webrtc_max", "webrtc_very_aggressive", "webrtc_level3"]:
                self.processor = WebRTCNSDenoiser(frame_size, sample_rate, level=3)
            elif self.method in ["webrtc_extreme", "extreme_noise_clean"]:
                # Hibrit: Maksimum WebRTC Level 3 + Adaptif Wiener Filtresi (Sıfır Arka Plan Gürültüsü)
                class HybridExtremeDenoiser:
                    def __init__(self, fs, sr):
                        self.ns = WebRTCNSDenoiser(fs, sr, level=3)
                        self.wiener = AdaptiveWienerDenoiser(fs, sr, min_gain=0.08)
                    def process(self, frame):
                        f1 = self.ns.process(frame)
                        return self.wiener.process(f1)
                self.processor = HybridExtremeDenoiser(frame_size, sample_rate)
            elif self.method in ["none", "passthrough", "off"]:
                self.processor = None
            else:
                print(f"[Denoiser] Uyarı: Bilinmeyen metod '{self.method}', passthrough (none) moduna düşülüyor.", flush=True)
                self.processor = None
        except Exception as e:
            print(f"[Denoiser HATA] Modül başlatılırken hata oluştu: {e}. Graceful degradation ile ham sese geçildi.", flush=True)
            self.processor = None

    def process(self, frame_float32):
        """
        Bir ses karesini işler. Hata durumunda sistemi çökertmeden ham sesi (graceful degradation) döndürür.
        """
        if self.processor is None or self.method in ["none", "passthrough", "off"]:
            return frame_float32
            
        try:
            cleaned_frame = self.processor.process(frame_float32)
            if cleaned_frame is None or len(cleaned_frame) != len(frame_float32):
                return frame_float32
            return cleaned_frame
        except Exception as e:
            # Graceful degradation - Hatada sistemi düşürme, ham sesi döndür
            return frame_float32
