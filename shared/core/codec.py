import os
import sys
import glob
import shutil
import io
import logging
import numpy as np
import soundfile as sf

# Numba kontrolü
try:
    from numba import jit
except ImportError:
    def jit(nopython=True, cache=True):
        return lambda f: f

# ----------------- IMA ADPCM GEREKSİNİMLERİ (JIT UYUMLU) -----------------
adpcm_step_table = np.array([
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 62, 69, 76, 84, 94, 104, 114, 126,
    139, 153, 169, 186, 205, 226, 248, 273, 301, 331,
    364, 400, 440, 484, 532, 585, 644, 708, 779, 857,
    943, 1037, 1141, 1255, 1381, 1519, 1671, 1838, 2022, 2224,
    2446, 2691, 2960, 3256, 3582, 3940, 4334, 4767, 5244, 5768,
    6345, 6980, 7678, 8446, 9291, 10220, 11242, 12366, 13603, 14963,
    16459, 18105, 19915, 21907, 24098, 26508, 29159, 32075, 32767
], dtype=np.int32)

adpcm_index_table = np.array([
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8
], dtype=np.int32)

@jit(nopython=True, cache=True)
def _adpcm_encode_numba(pcm, step_table, index_table):
    N = len(pcm)
    out = np.zeros(N // 2, dtype=np.uint8)
    pred_sample = 0
    step_idx = 0
    
    for i in range(0, N, 2):
        # Örnek 1
        diff = int(pcm[i]) - pred_sample
        sign = 0
        if diff < 0:
            sign = 8
            diff = -diff
            
        step = step_table[step_idx]
        code = 0
        temp_step = step
        if diff >= temp_step:
            code |= 4
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 2
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 1
            
        diffq = step >> 3
        if code & 4:
            diffq += step
        if code & 2:
            diffq += step >> 1
        if code & 1:
            diffq += step >> 2
            
        if sign:
            pred_sample -= diffq
        else:
            pred_sample += diffq
            
        if pred_sample > 32767:
            pred_sample = 32767
        elif pred_sample < -32768:
            pred_sample = -32768
            
        step_idx += index_table[code]
        if step_idx < 0:
            step_idx = 0
        elif step_idx > 88:
            step_idx = 88
            
        code1 = code | sign
        
        # Örnek 2
        diff = int(pcm[i+1]) - pred_sample
        sign = 0
        if diff < 0:
            sign = 8
            diff = -diff
            
        step = step_table[step_idx]
        code = 0
        temp_step = step
        if diff >= temp_step:
            code |= 4
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 2
            diff -= temp_step
        temp_step >>= 1
        if diff >= temp_step:
            code |= 1
            
        diffq = step >> 3
        if code & 4:
            diffq += step
        if code & 2:
            diffq += step >> 1
        if code & 1:
            diffq += step >> 2
            
        if sign:
            pred_sample -= diffq
        else:
            pred_sample += diffq
            
        if pred_sample > 32767:
            pred_sample = 32767
        elif pred_sample < -32768:
            pred_sample = -32768
            
        step_idx += index_table[code]
        if step_idx < 0:
            step_idx = 0
        elif step_idx > 88:
            step_idx = 88
            
        code2 = code | sign
        
        out[i // 2] = (code2 << 4) | code1
        
    return out

@jit(nopython=True, cache=True)
def _adpcm_decode_numba(adpcm_bytes, step_table, index_table):
    N = len(adpcm_bytes)
    out = np.zeros(N * 2, dtype=np.int16)
    pred_sample = 0
    step_idx = 0
    
    for i in range(N):
        byte_val = adpcm_bytes[i]
        code1 = byte_val & 0x0F
        code2 = (byte_val >> 4) & 0x0F
        
        # Örnek 1 çöz
        step = step_table[step_idx]
        diffq = step >> 3
        if code1 & 4:
            diffq += step
        if code1 & 2:
            diffq += step >> 1
        if code1 & 1:
            diffq += step >> 2
            
        if code1 & 8:
            pred_sample -= diffq
        else:
            pred_sample += diffq
            
        if pred_sample > 32767:
            pred_sample = 32767
        elif pred_sample < -32768:
            pred_sample = -32768
            
        step_idx += index_table[code1]
        if step_idx < 0:
            step_idx = 0
        elif step_idx > 88:
            step_idx = 88
            
        out[2 * i] = pred_sample
        
        # Örnek 2 çöz
        step = step_table[step_idx]
        diffq = step >> 3
        if code2 & 4:
            diffq += step
        if code2 & 2:
            diffq += step >> 1
        if code2 & 1:
            diffq += step >> 2
            
        if code2 & 8:
            pred_sample -= diffq
        else:
            pred_sample += diffq
            
        if pred_sample > 32767:
            pred_sample = 32767
        elif pred_sample < -32768:
            pred_sample = -32768
            
        step_idx += index_table[code2]
        if step_idx < 0:
            step_idx = 0
        elif step_idx > 88:
            step_idx = 88
            
        out[2 * i + 1] = pred_sample
        
    return out

def load_opus_dll():
    """opus.dll'i bul ve Windows DLL arama yoluna ekle.

    Arama sırası:
      1. Proje kök dizininde opus.dll zaten varsa direkt kullan.
      2. PyInstaller _MEIPASS (EXE bundle) içindeki opus.dll.
      3. Tüm bilinen site-packages av.libs konumları
         (venv, .venv-release, Anaconda global, sistem site-packages).
      Bulduğu ilk geçerli DLL'i proje köküne kopyalar ve PATH/add_dll_directory'e ekler.
    """
    import site
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dll_dest = os.path.join(project_root, "opus.dll")

    # 1. Zaten proje kökünde varsa doğrudan kayıt et
    if os.path.exists(dll_dest):
        _register_dll_dir(project_root)
        return

    # 2. PyInstaller bundle (_MEIPASS)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and os.path.exists(os.path.join(meipass, "opus.dll")):
        _register_dll_dir(meipass)
        return

    # 3. site-packages içindeki av.libs klasörlerini tara
    candidate_roots = [
        # venv ve .venv-release (proje altında)
        os.path.join(project_root, "venv"),
        os.path.join(project_root, ".venv-release"),
    ]
    # Sistem / Anaconda / global site-packages
    try:
        candidate_roots += site.getsitepackages()
    except AttributeError:
        pass
    if site.getusersitepackages():
        candidate_roots.append(site.getusersitepackages())
    # sys.path içindeki site-packages
    for p in sys.path:
        if "site-packages" in p:
            candidate_roots.append(p)

    for root in candidate_roots:
        # av.libs doğrudan
        for search_dir in [
            os.path.join(root, "av.libs"),
            os.path.join(root, "Lib", "site-packages", "av.libs"),
        ]:
            if not os.path.isdir(search_dir):
                continue
            dlls = glob.glob(os.path.join(search_dir, "*opus*.dll"))
            if dlls:
                try:
                    shutil.copy2(dlls[0], dll_dest)
                    _register_dll_dir(project_root)
                    return
                except OSError:
                    logging.getLogger(__name__).debug(
                        "Bundled Opus DLL copy failed from %s", dlls[0], exc_info=True
                    )


def _register_dll_dir(directory):
    """Verilen klasörü Windows DLL arama yoluna ekler."""
    if directory not in os.environ.get("PATH", ""):
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(directory)
    except (AttributeError, OSError):
        logging.getLogger(__name__).debug(
            "Opus DLL search path registration failed for %s", directory, exc_info=True
        )


load_opus_dll()


# Ctypes util kütüphanesini yamalayarak opuslib_next'in Windows'ta DLL'i bulmasını sağla
import ctypes.util
original_find_library = ctypes.util.find_library

def custom_find_library(name):
    if name == 'opus':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        dll_path = os.path.join(project_root, "opus.dll")
        if os.path.exists(dll_path):
            return dll_path
    return original_find_library(name)

ctypes.util.find_library = custom_find_library

try:
    import opuslib_next as opuslib
except Exception as e:
    opuslib = None
    print(f"[Codec] Uyarı: opuslib_next yüklenemedi. Opus devredışı kalabilir. Hata: {e}", flush=True)


class IntercomCodec:
    """
    Genişletilmiş Ses Kodek Sınıfı.
    Opus (64k), Opus (16k), Ham PCM, G.711 (u-law), ADPCM ve FLAC modlarını destekler.
    """
    def __init__(self, method='opus_64', sample_rate=16000, channels=1):
        self.method = method.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        
        self.opus_encoder = None
        self.opus_decoder = None
        
        # Opus Başlatma
        if 'opus' in self.method:
            if opuslib is None:
                raise ImportError("Opus kodek seçildi ancak opuslib_next kütüphanesi yüklenemedi veya 'opus.dll' bulunamadı. "
                                  "Lütfen proje kök dizininde 'opus.dll' dosyasının bulunduğunu ve bağımlılıkların yüklendiğini kontrol edin.")
            bitrate = 64000 if '64' in self.method else 16000
            self.opus_encoder = opuslib.Encoder(self.sample_rate, self.channels, opuslib.APPLICATION_VOIP)
            self.opus_encoder.bitrate = bitrate
            self.opus_decoder = opuslib.Decoder(self.sample_rate, self.channels)
            print(f"[Codec] Opus Kodek başlatıldı ({bitrate / 1000.0:.0f} kbps mod).", flush=True)
            
        elif self.method == 'pcm':
            print("[Codec] Ham PCM mod aktif (Sıkıştırmasız).", flush=True)
            
        elif self.method == 'g711':
            print("[Codec] G.711 u-law mod aktif (Python 3.13 uyumlu logaritmik sıkıştırma).", flush=True)
            
        elif self.method == 'adpcm':
            print("[Codec] Numba IMA ADPCM mod aktif (4-bit sıkıştırma).", flush=True)
            
        elif self.method == 'flac':
            print("[Codec] FLAC mod aktif (soundfile tabanlı kayıpsız sıkıştırma).", flush=True)
            
        else:
            print(f"[Codec] Bilinmeyen kodek: '{self.method}'. Passthrough mod kullanılacak.", flush=True)
            self.method = 'pcm'

    def encode(self, pcm_data):
        """
        PCM int16 array verisini seçilen kodek yöntemiyle sıkıştırır.
        pcm_data: numpy array (int16)
        Döndürür: bytes
        """
        if not isinstance(pcm_data, np.ndarray):
            pcm_data = np.frombuffer(pcm_data, dtype=np.int16)
            
        # 1. Opus Kodlama
        if 'opus' in self.method and self.opus_encoder:
            pcm_bytes = pcm_data.tobytes()
            frame_size = len(pcm_bytes) // (2 * self.channels)
            return self.opus_encoder.encode(pcm_bytes, frame_size)
            
        # 2. Ham PCM (Düz bayt verisi)
        elif self.method == 'pcm':
            return pcm_data.tobytes()
            
        # 3. G.711 u-law (Logaritmik 16-bit to 8-bit)
        elif self.method == 'g711':
            # [-1.0, 1.0] aralığına normalize et
            x = pcm_data.astype(np.float32) / 32768.0
            sign = np.sign(x)
            abs_x = np.abs(x)
            mu = 255.0
            # u-law sıkıştırma formülü
            u = sign * np.log(1.0 + mu * abs_x) / np.log(1.0 + mu)
            # [-1.0, 1.0] -> [0, 255] byte aralığına doğrusal haritala
            u_byte = np.clip((u + 1.0) * 127.5, 0, 255).astype(np.uint8)
            return u_byte.tobytes()
            
        # 4. IMA ADPCM (4-bit sıkıştırma, JIT-hızlandırılmış)
        elif self.method == 'adpcm':
            # Çift sayıda örnek olmasını garanti edelim
            if len(pcm_data) % 2 != 0:
                pcm_data = np.append(pcm_data, 0)
            return _adpcm_encode_numba(pcm_data, adpcm_step_table, adpcm_index_table).tobytes()
            
        # 5. FLAC (Kayıpsız sıkıştırma)
        elif self.method == 'flac':
            buf = io.BytesIO()
            sf.write(buf, pcm_data, self.sample_rate, format='FLAC', subtype='PCM_16')
            return buf.getvalue()
            
        return pcm_data.tobytes()

    def decode(self, compressed_bytes, frame_size):
        """
        Sıkıştırılmış veriyi çözüp PCM int16 array formatına getirir.
        compressed_bytes: bytes
        frame_size: int (örnek sayısı, örn: 160)
        Döndürür: numpy array (int16)
        """
        # 1. Opus Çözme
        if 'opus' in self.method and self.opus_decoder:
            try:
                decoded_bytes = self.opus_decoder.decode(compressed_bytes, frame_size)
                return np.frombuffer(decoded_bytes, dtype=np.int16)
            except Exception:
                # Gelen paket ham PCM olabilir (320 byte)
                if len(compressed_bytes) == frame_size * 2 * self.channels:
                    return np.frombuffer(compressed_bytes, dtype=np.int16)
                raise
            
        # 2. Ham PCM Çözme
        elif self.method == 'pcm':
            # Gelen paket Opus paketi ise ve opuslib varsa otomatik çöz
            if len(compressed_bytes) < frame_size * 2 * self.channels and opuslib is not None:
                if not hasattr(self, "_fallback_opus_dec") or self._fallback_opus_dec is None:
                    self._fallback_opus_dec = opuslib.Decoder(self.sample_rate, self.channels)
                try:
                    dec = self._fallback_opus_dec.decode(compressed_bytes, frame_size)
                    return np.frombuffer(dec, dtype=np.int16)
                except Exception:
                    pass
            return np.frombuffer(compressed_bytes, dtype=np.int16)
            
        # 3. G.711 u-law Çözme
        elif self.method == 'g711':
            u_byte = np.frombuffer(compressed_bytes, dtype=np.uint8).astype(np.float32)
            # [0, 255] -> [-1.0, 1.0]
            u = (u_byte / 127.5) - 1.0
            mu = 255.0
            # Ters u-law formülü
            x = np.sign(u) * (((1.0 + mu) ** np.abs(u) - 1.0) / mu)
            pcm = np.clip(x * 32768.0, -32768, 32767).astype(np.int16)
            return pcm
            
        # 4. IMA ADPCM Çözme (4-bit to 16-bit, JIT-hızlandırılmış)
        elif self.method == 'adpcm':
            adpcm_data = np.frombuffer(compressed_bytes, dtype=np.uint8)
            return _adpcm_decode_numba(adpcm_data, adpcm_step_table, adpcm_index_table)
            
        # 5. FLAC Çözme
        elif self.method == 'flac':
            buf = io.BytesIO(compressed_bytes)
            data, _ = sf.read(buf, dtype='int16')
            return data
            
        return np.frombuffer(compressed_bytes, dtype=np.int16)
