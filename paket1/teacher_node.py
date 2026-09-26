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

# PTT (Bas-Konuş) Tuş Dinleme İş Parçacığı
def ptt_listener_thread_func(state_holder, stop_event):
    print("\n-------------------------------------------------------------", flush=True)
    print(">>> [PTT MODU AKTİF] Sesinizi açıp kapatmak için ENTER tuşuna basın.", flush=True)
    print(">>> Varsayılan durum: SESSİZ", flush=True)
    print("-------------------------------------------------------------\n", flush=True)
    
    while not stop_event.is_set():
        try:
            # Enter tuşuna basılmasını bekle (Bloklayıcı okuma)
            sys.stdin.readline()
            if stop_event.is_set():
                break
            
            # Durumu tersine çevir (Toggle)
            state_holder['transmitting'] = not state_holder['transmitting']
            
            if state_holder['transmitting']:
                print(">>> [PTT AKTİF] Konuşuyorsunuz... (Susturmak için ENTER'a basın)", flush=True)
            else:
                print(">>> [PTT SESSİZ] Mikrofonunuz kapatıldı. (Konuşmak için ENTER'a basın)", flush=True)
        except Exception:
            break

def print_local_ips(port):
    print("\n-------------------------------------------------------------", flush=True)
    print("📌 HOCA BİLGİSAYARI AKTİF İLETİŞİM IP ADRESLERİ:", flush=True)
    try:
        hostname = socket.gethostname()
        ips = socket.gethostbyname_ex(hostname)[2]
        for ip in ips:
            if not ip.startswith("127."):
                print(f"   👉 HOCA IP ADRESİ:  {ip}  (Port: {port})", flush=True)
    except Exception:
        pass
    print("\n   ⚠️  Öğrenci bilgisayarında bu IP adresini girmelisiniz!", flush=True)
    print("-------------------------------------------------------------\n", flush=True)

def main():
    parser = argparse.ArgumentParser(description="SimClever Intercom - Hoca Paneli (Teacher Node)")
    parser.add_argument('--ip', type=str, default=None, help="Öğrenci bilgisayarının IP adresi (güvenlik duvarını aşmak için opsiyonel)")
    parser.add_argument('--port', type=int, default=5000, help="Haberleşme port numarası")
    parser.add_argument('--codec', type=str, default='pcm', help="Kodek (pcm, opus_64, opus_16, g711, adpcm, flac)")
    parser.add_argument('--aec', type=str, default='webrtc', help="Yankı engelleme (webrtc, speex, numba, delay, none)")
    parser.add_argument('--vad', type=str, default='none', help="Konuşma algılama (webrtc, rms, zcr_rms, spectral_flux, silero, none). Varsayılan: none (Sürekli iletim)")

    parser.add_argument('--ptt', action='store_true', help="Bas-Konuş (PTT) modunu aktifleştir")
    parser.add_argument('--frame-size', type=int, default=160, help="Frame boyutu (örnek sayısı)")
    # Paket 2 — Gürültü Bastırma Parametreleri
    parser.add_argument('--denoiser', type=str, default='webrtc_mild',
                        help="Gürültü bastırma (webrtc_mild, webrtc_aggressive, baseline, none). Varsayılan: webrtc_mild")
    parser.add_argument('--arch', type=str, default='m2',
                        help="Mimari (m1: Öğrenci denoise, m2: Hoca denoise). Varsayılan: m2 (İnce İstemci)")

    
    args = parser.parse_args()
    
    print("\n================ HOCA PANELİ BAŞLATILIYOR ================", flush=True)
    print_local_ips(args.port)
    
    if args.ip:
        print(f"Hedef Öğrenci IP: {args.ip}:{args.port}")
    print(f"Haberleşme Portu: {args.port} | Mod: {'PTT (Bas-Konuş)' if args.ptt else 'Tam Dupleks'}")
    print(f"Kodek: {args.codec} | AEC: {args.aec} | VAD: {args.vad}")
    arch_label = "M2 — İnce İstemci (Hoca Denoise)" if args.arch.lower() == 'm2' else "M1 — Yerel İşleme (Öğrenci Denoise)"
    print(f"Denoiser: {args.denoiser} | Mimari: {arch_label}")
    
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=args.frame_size)
    vad = VADProcessor(method=args.vad, frame_size=args.frame_size)
    # Paket 2: M2 mimarisinde hoca tarafında gürültü bastırma aktif edilir
    denoiser = DenoiserProcessor(
        method=args.denoiser if args.arch.lower() == 'm2' else 'none',
        frame_size=args.frame_size,
        sample_rate=16000
    )
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port)) # Hoca port adresinde dinler
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
            
    running = True
    seq_num = 0
    ssrc = 12345
    
    # Tampon kuyrukları
    play_queue = []
    send_queue = []
    raw_record_queue = []
    clean_record_queue = []
    proc_record_queue = []
    queue_lock = threading.Lock()
    
    stop_event = threading.Event()
    
    # Dinamik Hoca Durum Bilgisi (PTT için)
    # Eğer PTT seçilmişse başlangıçta sessiz modda başlar, seçilmemişse doğrudan iletir
    # Eğer IP belirtilmişse doğrudan o adrese göndermeye hazır başlar, belirtilmemişse ilk gelen paketten öğrenir
    state_holder = {
        'transmitting': not args.ptt,
        'client_addr': (args.ip, args.port) if args.ip else None
    }
    
    # 1. Kayıtcı İş Parçacığını Başlat
    record_thread = threading.Thread(
        target=recorder_thread_func, 
        args=('teacher_raw_mic.wav', 'teacher_cleaned_mic.wav', 'teacher_processed_output.wav', raw_record_queue, clean_record_queue, proc_record_queue, stop_event)
    )
    record_thread.start()
    
    # 2. PTT Dinleyici İş Parçacığı (Eğer PTT aktifse)
    ptt_thread = None
    if args.ptt:
        ptt_thread = threading.Thread(target=ptt_listener_thread_func, args=(state_holder, stop_event))
        ptt_thread.daemon = True
        ptt_thread.start()
        
    # 3. Alıcı Ağ İş Parçacığı
    sent_count = [0]
    recv_count = [0]
    ping_count = [0]
    
    def network_receiver():
        nonlocal running
        while running:
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(4096)
                state_holder['client_addr'] = addr
                
                if data == b"ping":
                    ping_count[0] += 1
                    if ping_count[0] == 1:
                        print(f"\n[Ağ] 🔔 Öğrenci Kayıt Olundu! (Öğrenci IP: {addr[0]}:{addr[1]})", flush=True)
                    continue
                
                # Paket çözümleme
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
                    print(f"\n[Ağ] 📡 Öğrenci'den İLK SES PAKETİ ALINDI! ({addr[0]}:{addr[1]})", flush=True)
                
                pcm_float = pcm_frame.astype(np.float32) / 32768.0
                
                # Denoiser 'none' ise passthrough (Pürüzsüz Paket 1 sesi)
                if denoiser.method in ["none", "off", "passthrough"]:
                    out_float = pcm_float
                else:
                    out_float = denoiser.process(pcm_float)
                
                out_float = np.clip(out_float, -1.0, 1.0)
                
                with queue_lock:
                    play_queue.append(out_float)
                    if len(play_queue) > 25:
                        play_queue.pop(0)

            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            except Exception as e:
                if running:
                    print(f"[NetRecv] Hata: {e}", flush=True)
                    
    recv_thread = threading.Thread(target=network_receiver)
    recv_thread.start()
    
    # 4. Gönderici Ağ İş Parçacığı
    def network_sender():
        nonlocal running
        while running:
            if len(send_queue) > 0 and state_holder['client_addr'] is not None:
                with queue_lock:
                    packet = send_queue.pop(0)
                try:
                    sock.sendto(packet, state_holder['client_addr'])
                    sent_count[0] += 1
                    if sent_count[0] == 1:
                        print(f"[Ağ] 📤 Öğrenci'ye İLK SES PAKETİ GÖNDERİLDİ! ({state_holder['client_addr'][0]}:{state_holder['client_addr'][1]})", flush=True)
                except Exception as e:
                    print(f"[NetSend Hatası]: Öğrenciye ({state_holder['client_addr']}) ses gönderilemiyor: {e}", flush=True)
                    time.sleep(0.5)
            else:
                time.sleep(0.001)
                
    send_thread = threading.Thread(target=network_sender)
    send_thread.start()

    
    # 5. Ayrı Giriş ve Çıkış Ses Callback Fonksiyonları
    import sounddevice as sd
    
    aec_reference_queue = []
    
    def playback_callback(outdata, frames, time_info, status):
        # A. Alıcı Tarafı: Hoparlöre çalma
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
            print("\n[Mikrofon] 🎙️ Mikrofondan ses sinyali yakalanıyor!", flush=True)
            mic_active_printed[0] = True
        elif peak_level == 0.0 and not mic_silent_printed[0]:
            print("\n[Uyarı] ⚠️ Mikrofondan sıfır ses geliyor! (Windows Mikrofon İznini Kontrol Edin)", flush=True)
            mic_silent_printed[0] = True

        
        # AEC referansını çek
        with queue_lock:
            if len(aec_reference_queue) > 0:
                far_frame = aec_reference_queue.pop(0)
            else:
                far_frame = np.zeros(frames, dtype=np.float32)
                
        # int16 dönüşümü (np.clip ile int16 overflow patlamaları engellenir)
        near_int16 = np.clip(near_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        far_int16 = np.clip(far_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)

        
        # 1. Akustik Yankı Engelleme (AEC)
        cleaned_int16 = aec.process(near_int16, far_int16)
        
        # Temizlenmiş sesi kayıt kuyruğuna at (float32 olarak)
        clean_frame = cleaned_int16.astype(np.float32) / 32768.0
        clean_record_queue.append(clean_frame)
        
        # 2. VAD ve PTT İletim Koşulu Kontrolü
        if state_holder['transmitting'] and vad.is_speech(cleaned_int16):
            compressed = codec.encode(cleaned_int16)
            
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
        print("\n[Hoca] Tam dupleks ses akışı aktif (Giriş & Çıkış bağımsız).", flush=True)
        if not args.ptt:
            print("Konuşabilirsiniz... Durdurmak için Ctrl+C tuşlarına basın.", flush=True)
        
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
        
        recv_thread.join(1.0)
        send_thread.join(1.0)
        
        stop_event.set()
        record_thread.join(2.0)
        print("[Hoca] Başarıyla sonlandırıldı.", flush=True)

if __name__ == "__main__":
    main()
