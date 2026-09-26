import os
import sys
import time
import wave
import socket
import struct
import argparse
import threading
import numpy as np
import sounddevice as sd

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor
from core.denoiser import DenoiserProcessor


# RTP Yardımcı Fonksiyonları
def make_rtp_packet(seq_num, timestamp, ssrc, payload):
    header = struct.pack("!BBHII", 0x80, 96, seq_num & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc)
    return header + payload

def parse_rtp_packet(packet):
    if len(packet) < 12:
        return None, packet
    header = struct.unpack("!BBHII", packet[:12])
    seq_num = header[2]
    payload = packet[12:]
    return seq_num, payload

# Arka Planda Çalışan Ses Kayıt İş parçacığı
def recorder_thread_func(raw_filename, clean_filename, proc_filename, raw_queue, clean_queue, proc_queue, stop_event):
    raw_wav = wave.open(raw_filename, 'wb')
    raw_wav.setnchannels(1)
    raw_wav.setsampwidth(2) # 16-bit
    raw_wav.setframerate(16000)
    
    clean_wav = wave.open(clean_filename, 'wb')
    clean_wav.setnchannels(1)
    clean_wav.setsampwidth(2) # 16-bit
    clean_wav.setframerate(16000)
    
    proc_wav = wave.open(proc_filename, 'wb')
    proc_wav.setnchannels(1)
    proc_wav.setsampwidth(2) # 16-bit
    proc_wav.setframerate(16000)
    
    try:
        while not stop_event.is_set() or len(raw_queue) > 0 or len(clean_queue) > 0 or len(proc_queue) > 0:
            if len(raw_queue) > 0:
                raw_frame = raw_queue.pop(0)
                raw_int16 = (raw_frame * 32767.0).astype(np.int16)
                raw_wav.writeframes(raw_int16.tobytes())
            if len(clean_queue) > 0:
                clean_frame = clean_queue.pop(0)
                clean_int16 = (clean_frame * 32767.0).astype(np.int16)
                clean_wav.writeframes(clean_int16.tobytes())
            if len(proc_queue) > 0:
                proc_frame = proc_queue.pop(0)
                proc_int16 = (proc_frame * 32767.0).astype(np.int16)
                proc_wav.writeframes(proc_int16.tobytes())
            time.sleep(0.005)
    except Exception as e:
        print(f"[Recorder] Hata oluştu: {e}", flush=True)
    finally:
        raw_wav.close()
        clean_wav.close()
        proc_wav.close()
        print("[Recorder] Ses dosyaları başarıyla kapatıldı.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="SimClever Intercom - Öğrenci Düğümü (Student Node)")
    parser.add_argument('--ip', type=str, default=None, help="Hoca bilgisayarının IP adresi")
    parser.add_argument('--port', type=int, default=5000, help="Haberleşme port numarası")
    parser.add_argument('--codec', type=str, default='pcm', help="Kodek (pcm, opus_64, opus_16, g711, adpcm, flac)")
    parser.add_argument('--aec', type=str, default='webrtc', help="Yankı engelleme (webrtc, speex, numba, delay, none)")
    parser.add_argument('--vad', type=str, default='none', help="Konuşma algılama (webrtc, rms, zcr_rms, spectral_flux, silero, none). Varsayılan: none (Sürekli iletim)")

    parser.add_argument('--frame-size', type=int, default=160, help="Frame boyutu (örnek sayısı)")
    # Paket 2 — Gürültü Bastırma Parametreleri
    parser.add_argument('--denoiser', type=str, default='none',
                        help="Gürültü bastırma (none, webrtc, rnnoise, deepfilternet, baseline). Varsayılan: none")
    parser.add_argument('--arch', type=str, default='m2',
                        help="Mimari (m1: Öğrenci denoise, m2: Hoca denoise). Varsayılan: m2 (İnce İstemci)")
    
    args = parser.parse_args()
    
    if not args.ip:
        print("\n================ SİMCLEVER ÖĞRENCİ DÜĞÜMÜ ================", flush=True)
        print("Lütfen Hoca bilgisayarının ekranında yazan IP adresini girin.", flush=True)
        user_ip = input("👉 HOCA IP ADRESİ: ").strip()
        args.ip = user_ip if user_ip else "192.168.137.110"
        print(f"Hedef IP Ayarlandı: {args.ip}\n", flush=True)


    
    print("\n================ ÖĞRENCİ DÜĞÜMÜ BAŞLATILIYOR ================", flush=True)
    print(f"Hedef Hoca IP: {args.ip}:{args.port}")
    print(f"Kodek: {args.codec} | AEC: {args.aec} | VAD: {args.vad}")
    arch_label = "M1 — Yerel İşleme (Öğrenci Denoise)" if args.arch.lower() == 'm1' else "M2 — İnce İstemci (Hoca Denoise)"
    print(f"Denoiser: {args.denoiser} | Mimari: {arch_label}")
    
    # İşlemcileri ve kodekleri yükle
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=args.frame_size)
    vad = VADProcessor(method=args.vad, frame_size=args.frame_size)
    # Paket 2: M1 mimarisinde öğrenci tarafında gürültü bastırma aktif edilir
    denoiser = DenoiserProcessor(
        method=args.denoiser if args.arch.lower() == 'm1' else 'none',
        frame_size=args.frame_size,
        sample_rate=16000
    )
    
    # Soket ve Ağ Yapısı
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port)) # Öğrenci de Hoca ile aynı port (args.port) üzerinden dinler (Simetrik Port)
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
            
    dest_addr = (args.ip, args.port)
    
    running = True
    seq_num = 0
    ssrc = 54321
    
    # Tampon kuyrukları (Lock-free ve Thread-safe erişim için listeler)
    play_queue = []
    send_queue = []
    raw_record_queue = []
    clean_record_queue = []
    proc_record_queue = []
    queue_lock = threading.Lock()
    
    stop_event = threading.Event()
    
    # 1. Kayıtcı İş Parçacığını Başlat
    record_thread = threading.Thread(
        target=recorder_thread_func, 
        args=('student_raw_mic.wav', 'student_cleaned_mic.wav', 'student_processed_output.wav', raw_record_queue, clean_record_queue, proc_record_queue, stop_event)
    )
    record_thread.start()
    
    sent_count = [0]
    recv_count = [0]
    
    # 2. Alıcı Ağ İş Parçacığı (UDP dinler, çözer, play_queue'ya atar)
    def network_receiver():
        nonlocal running
        while running:
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(4096)
                if data == b"ping":
                    continue
                _, payload = parse_rtp_packet(data)
                
                # Kod çöz
                try:
                    pcm_frame = codec.decode(payload, args.frame_size)
                    if pcm_frame is None or len(pcm_frame) != args.frame_size:
                        continue
                except Exception:
                    continue
                
                recv_count[0] += 1
                if recv_count[0] == 1:
                    print(f"\n[Ağ] 📡 Hoca'dan İLK SES PAKETİ ALINDI! ({addr[0]})", flush=True)
                
                # Float32'ye normalize et (çıtırtı/bozulmayı engellemek için)
                pcm_float = pcm_frame.astype(np.float32) / 32768.0
                pcm_float = np.clip(pcm_float, -1.0, 1.0)
                
                with queue_lock:
                    play_queue.append(pcm_float)
                    # Gecikme tamponu: Wi-Fi takılmalarında paket düşürmek yerine 25 frame tamponla
                    if len(play_queue) > 25:
                        play_queue.pop(0)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            except Exception as e:
                if running:
                    print(f"[NetRecv] Hata: {e}", flush=True)
                    
    recv_thread = threading.Thread(target=network_receiver, daemon=True)
    recv_thread.start()
    
    # 3. Gönderici Ağ İş Parçacığı (send_queue'daki paketleri ağa basar)
    def network_sender():
        nonlocal running
        while running:
            if len(send_queue) > 0:
                with queue_lock:
                    packet = send_queue.pop(0)
                try:
                    sock.sendto(packet, dest_addr)
                    sent_count[0] += 1
                    if sent_count[0] == 1:
                        print(f"[Ağ] 📤 Hoca'ya İLK SES PAKETİ GÖNDERİLDİ! ({dest_addr[0]}:{dest_addr[1]})", flush=True)
                except Exception as e:
                    print(f"[NetSend Hatası]: Hoca'ya ulaşılamıyor ({dest_addr}): {e}", flush=True)
                    time.sleep(0.5)
            else:
                time.sleep(0.001)
                
    send_thread = threading.Thread(target=network_sender, daemon=True)
    send_thread.start()

    # 3b. Kalp Atışı (Heartbeat) İş Parçacığı - Hoca'ya sürekli ping göndererek güvenlik duvarını açık tutar
    def heartbeat_sender():
        nonlocal running
        ping_err_logged = False
        while running:
            try:
                sock.sendto(b"ping", dest_addr)
            except Exception as e:
                if not ping_err_logged:
                    print(f"[Ağ Uyarısı] Hoca IP'sine ({dest_addr[0]}) ping atılamıyor: {e}", flush=True)
                    ping_err_logged = True
            time.sleep(1.0) # Her 1 saniyede bir ping at
            
    heartbeat_thread = threading.Thread(target=heartbeat_sender, daemon=True)
    heartbeat_thread.start()
    
    # 4. Ayrı Giriş ve Çıkış Ses Callback Fonksiyonları
    aec_reference_queue = []
    
    def playback_callback(outdata, frames, time_info, status):
        # A. Alıcı Tarafı: Hoparlöre çalma işlemi
        with queue_lock:
            if len(play_queue) > 0:
                far_frame = play_queue.pop(0)
            else:
                far_frame = np.zeros(frames, dtype=np.float32)
                
        outdata[:, 0] = far_frame
        proc_record_queue.append(far_frame.copy())
        
        # AEC referansı için kuyruğa ekle
        with queue_lock:
            aec_reference_queue.append(far_frame.copy())
            if len(aec_reference_queue) > 10:
                aec_reference_queue.pop(0)
                
    mic_active_printed = [False]
    mic_silent_printed = [False]
    
    def capture_callback(indata, frames, time_info, status):
        nonlocal seq_num
        # B. Gönderici Tarafı: Mikrofondan yakalama
        near_frame = indata[:, 0].copy()
        raw_record_queue.append(near_frame.copy())
        
        peak_level = np.max(np.abs(near_frame))
        if peak_level > 0.01 and not mic_active_printed[0]:
            print("\n[Mikrofon] 🎙️ Mikrofondan ses sinyali yakalanıyor! (Konuşma algılandı)", flush=True)
            mic_active_printed[0] = True
        elif peak_level == 0.0 and not mic_silent_printed[0]:
            print("\n[Uyarı] ⚠️ Mikrofondan sıfır ses geliyor! (Windows Mikrofon İznini ve Ses Düzeyini Kontrol Edin)", flush=True)
            mic_silent_printed[0] = True

        
        # AEC referansını çek
        with queue_lock:
            if len(aec_reference_queue) > 0:
                far_frame = aec_reference_queue.pop(0)
            else:
                far_frame = np.zeros(frames, dtype=np.float32)
                
        # int16 formatına dönüştür
        near_int16 = np.clip(near_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        far_int16 = np.clip(far_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        
        # 1. Akustik Yankı Engelleme (AEC)
        cleaned_int16 = aec.process(near_int16, far_int16)
        
        # Denoiser 'none' ise doğrudan AEC çıktısını kullan
        if denoiser.method in ["none", "off", "passthrough"]:
            tx_int16 = cleaned_int16
        else:
            clean_frame = cleaned_int16.astype(np.float32) / 32768.0
            denoised_frame = denoiser.process(clean_frame)
            tx_int16 = np.clip(denoised_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        
        # 2. Konuşma Algılama (VAD)
        if vad.is_speech(tx_int16):
            # 3. Kodlama ve Gönderim Kuyruğuna Ekle
            compressed = codec.encode(tx_int16)
            
            # RTP Başlığı Ekle
            timestamp = int(time.time() * 16000)
            rtp_packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
            seq_num += 1
            
            with queue_lock:
                send_queue.append(rtp_packet)
                if len(send_queue) > 15:
                    send_queue.pop(0)

    # Sounddevice akışlarını başlat (Giriş ve Çıkış Ayrı)
    try:
        input_stream = sd.InputStream(

            samplerate=16000, 
            channels=1, 
            callback=capture_callback, 
            blocksize=args.frame_size,
            dtype='float32'
        )
        output_stream = sd.OutputStream(
            samplerate=16000, 
            channels=1, 
            callback=playback_callback, 
            blocksize=args.frame_size,
            dtype='float32'
        )
        input_stream.start()
        output_stream.start()
        print("\n[Öğrenci] Tam dupleks ses akışı aktif (Giriş & Çıkış bağımsız).", flush=True)
        print("Konuşabilirsiniz... Durdurmak için Ctrl+C tuşlarına basın.", flush=True)
        
        # Programı canlı tut
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nDurduruluyor...", flush=True)
    finally:
        running = False
        if 'input_stream' in locals():
            try:
                input_stream.stop()
                input_stream.close()
            except Exception:
                pass
        if 'output_stream' in locals():
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass
        sock.close()
        
        # Thread'lerin bitmesini bekle
        recv_thread.join(1.0)
        send_thread.join(1.0)
        
        stop_event.set()
        record_thread.join(2.0)
        print("[Öğrenci] Başarıyla sonlandırıldı.", flush=True)

if __name__ == "__main__":
    main()
