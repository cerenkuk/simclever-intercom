import sounddevice as sd
import numpy as np

def list_audio_devices():
    """
    Sistemdeki kullanılabilir ses giriş ve çıkış cihazlarını listeler.
    """
    devices = sd.query_devices()
    input_devices = []
    output_devices = []
    
    for idx, dev in enumerate(devices):
        device_info = {
            'index': idx,
            'name': dev['name'],
            'max_input_channels': dev['max_input_channels'],
            'max_output_channels': dev['max_output_channels'],
            'default_samplerate': dev['default_samplerate']
        }
        if dev['max_input_channels'] > 0:
            input_devices.append(device_info)
        if dev['max_output_channels'] > 0:
            output_devices.append(device_info)
            
    return input_devices, output_devices

class AudioIO:
    def __init__(self, sample_rate=16000, frame_duration_ms=10, input_device_idx=None, output_device_idx=None):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.chunk_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        
        self.input_device_idx = input_device_idx
        self.output_device_idx = output_device_idx
        
        self.input_stream = None
        self.output_stream = None
        
    def start_input_stream(self, callback):
        """
        Mikrofon giriş akışını başlatır.
        callback fonksiyonu numpy array (chunk_size, 1) alır.
        """
        def sd_callback(indata, frames, time_info, status):
            if status:
                print(f"Giriş Akışı Durumu: {status}", flush=True)
            # numpy float32 tipinden int16'ya dönüştür
            # sounddevice varsayılan olarak float32 okur, dtype='int16' yapıyoruz
            callback(indata.copy())

        self.input_stream = sd.InputStream(
            device=self.input_device_idx,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='int16',
            callback=sd_callback
        )
        self.input_stream.start()
        
    def start_output_stream(self):
        """
        Hoparlör çıkış akışını başlatır.
        """
        # Çıkışı ham veri yazarak beslemek için OutputStream kullanıyoruz
        self.output_stream = sd.OutputStream(
            device=self.output_device_idx,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='int16'
        )
        self.output_stream.start()
        
    def play_audio(self, data):
        """
        Hoparlörden ses çalar (blocking olmayan hızlı yazma).
        data: numpy array (chunk_size, 1) veya (chunk_size,) int16
        """
        if self.output_stream:
            # Ses seviyesini kontrol altında tutmak veya doğrudan yazmak için
            # Eğer data 1D ise 2D yapıyoruz (channels=1 için)
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=1)
            self.output_stream.write(data)
            
    def stop_all(self):
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
