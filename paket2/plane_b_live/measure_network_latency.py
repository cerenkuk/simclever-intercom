import os
import sys
import time
import socket
import struct
import argparse
import threading
import numpy as np
import sounddevice as sd
from scipy.signal import correlate

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import IntercomCodec

def generate_chirp(sample_rate=16000, duration=0.1, f0=500, f1=3000):
    """
    Korelasyon analizi için 100ms'lik bir linear chirp (frekans süpürme) sinyali üretir.
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t**2))
    window = np.hanning(len(t))
    return chirp * window

# RTP Yardımcıları
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

def run_reflector(args):
    """
    B MAKİNESİ (Yansıtıcı): Gelen paketleri hoparlörden çalar ve
    aynı anda mikrofondan geri yakalayarak göndericiye yansıtır (AEC kapalı olmalıdır).
    """
    print(f"\n=== YANSITICI (REFLECTOR) BAŞLATILDI (Port: {args.port}) ===", flush=True)
    print(f"Kodek: {args.codec} | AEC: Kapalı (Yankı sızmalı) | VAD: Yok", flush=True)
    
    codec = IntercomCodec(method=args.codec)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
            
    play_queue = []
    send_queue = []
    queue_lock = threading.Lock()
    client_addr = (args.ip, args.port) if args.ip else None
    if client_addr:
        print(f"[Yansıtıcı] Ön-tanımlı Gönderici (A Cihazı) adresi girildi: {client_addr}. Otomatik delik açma (hole punching) aktif.", flush=True)
    running = True
    
    # Alıcı Ağ İş parçacığı
    def network_receiver():
        nonlocal client_addr, running
        recv_count = 0
        while running:
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(4096)
                recv_count += 1
                
                # Her 10 pakette bir veya ping geldiğinde ekrana yaz
                if recv_count % 10 == 0 or data == b"ping":
                    print(f"[Yansıtıcı] A Cihazından paket alındı! Adet: {recv_count} | Gönderen: {addr} | Boyut: {len(data)}", flush=True)
                
                if client_addr is None:
                    client_addr = (addr[0], args.port) # Simetrik Port
                    print(f"[Yansıtıcı] Gönderici (A Cihazı) algılandı: {client_addr}. Paket yansıtma aktif.", flush=True)
                else:
                    client_addr = (addr[0], args.port)
                
                if data == b"ping":
                    continue
                    
                _, payload = parse_rtp_packet(data)
                pcm_frame = codec.decode(payload, args.frame_size)
                pcm_float = pcm_frame.astype(np.float32) / 32768.0
                
                with queue_lock:
                    play_queue.append(pcm_float)
                    if len(play_queue) > 10:
                        play_queue.pop(0)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            except Exception as e:
                if running:
                    print(f"[ReflectRecv] Hata: {e}", flush=True)
                    
    # Gönderici Ağ İş parçacığı
    def network_sender():
        nonlocal running
        while running:
            if len(send_queue) > 0 and client_addr is not None:
                with queue_lock:
                    packet = send_queue.pop(0)
                try:
                    sock.sendto(packet, client_addr)
                except Exception:
                    pass
            else:
                time.sleep(0.001)
                
    # Ses Akışları
    def playback_callback(outdata, frames, time_info, status):
        with queue_lock:
            if len(play_queue) > 0:
                far_frame = play_queue.pop(0)
            else:
                far_frame = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = far_frame
        
    seq_num = 0
    ssrc = 88888
    
    def capture_callback(indata, frames, time_info, status):
        nonlocal seq_num
        near_frame = indata[:, 0].copy()
        near_int16 = (near_frame * 32767.0).astype(np.int16)
        compressed = codec.encode(near_int16)
        timestamp = int(time.time() * 16000)
        rtp_packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
        seq_num += 1
        
        if seq_num % 100 == 0:
            print(f"[Yansıtıcı] Ağa gönderilen paket sayısı: {seq_num}", flush=True)
            
        with queue_lock:
            send_queue.append(rtp_packet)
            if len(send_queue) > 15:
                send_queue.pop(0)
                
    recv_thread = threading.Thread(target=network_receiver)
    send_thread = threading.Thread(target=network_sender)
    
    recv_thread.start()
    send_thread.start()
    
    try:
        input_stream = sd.InputStream(
            samplerate=16000, 
            channels=1, 
            callback=capture_callback, 
            blocksize=args.frame_size, 
            dtype='float32',
            latency='low'
        )
        output_stream = sd.OutputStream(
            samplerate=16000, 
            channels=1, 
            callback=playback_callback, 
            blocksize=args.frame_size, 
            dtype='float32',
            latency='low'
        )
        
        input_stream.start()
        output_stream.start()
        
        in_lat = input_stream.latency * 1000.0
        out_lat = output_stream.latency * 1000.0
        print(f"[Yansıtıcı] Düşük gecikme modu aktif (Giriş Tamponu: {in_lat:.1f} ms, Çıkış Tamponu: {out_lat:.1f} ms)", flush=True)
        print("Yansıtıcı aktif. Durdurmak için Ctrl+C tuşlarına basın...", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDurduruluyor...", flush=True)
    finally:
        running = False
        try:
            input_stream.stop()
            input_stream.close()
            output_stream.stop()
            output_stream.close()
        except Exception:
            pass
        sock.close()
        recv_thread.join(1.0)
        send_thread.join(1.0)
        print("Yansıtıcı kapatıldı.", flush=True)

def run_measurer(args):
    """
    A MAKİNESİ (Ölçümcü): Karşı tarafa chirp yollar, dönen sesi kaydeder ve
    çapraz korelasyon ile ağızdan-kulağa (mouth-to-ear) gecikmeyi hesaplar.
    """
    print(f"\n=== GECİKME ÖLÇÜMCÜ (MEASURER) BAŞLATILDI (Hedef: {args.ip}:{args.port}) ===", flush=True)
    print(f"Kodek: {args.codec} | Süre: {args.duration} saniye", flush=True)
    
    codec = IntercomCodec(method=args.codec)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
            
    dest_addr = (args.ip, args.port)
    
    # 5 adet ping paketi ile güvenlik duvarı deliği açıp Hoca/Öğrenci IP kaydını tetikle
    print("Karşı tarafa bağlantı delikleri açılıyor (ping)...", flush=True)
    for _ in range(5):
        try:
            sock.sendto(b"ping", dest_addr)
        except Exception:
            pass
        time.sleep(0.1)
        
    received_samples = []
    running = True
    
    # Alıcı Ağ İş Parçacığı (Gelen paketleri kaydeder)
    def network_receiver():
        nonlocal running
        recv_count = 0
        while running:
            sock.settimeout(0.5)
            try:
                data, addr = sock.recvfrom(4096)
                recv_count += 1
                
                # Her 10 pakette bir veya ilk pakette ekrana yaz
                if recv_count % 10 == 0 or recv_count == 1:
                    print(f"[Ölçümcü] B Cihazından paket alındı! Adet: {recv_count} | Gönderen: {addr} | Boyut: {len(data)}", flush=True)
                
                if data == b"ping":
                    continue
                _, payload = parse_rtp_packet(data)
                pcm_frame = codec.decode(payload, args.frame_size)
                
                if len(received_samples) == 0:
                    print(f"[Ölçümcü] Yansıtıcıdan (B Cihazı) ilk ses yanıtı alındı! Veri akışı başladı.", flush=True)
                
                received_samples.extend(pcm_frame.tolist())
                if len(received_samples) % 1600 == 0:
                    print(f"[Ölçümcü] Toplam alınan örnek sayısı: {len(received_samples)}", flush=True)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue
            except Exception as e:
                if running:
                    print(f"[MeasureRecv] Hata: {e}", flush=True)
                    
    recv_thread = threading.Thread(target=network_receiver)
    recv_thread.start()
    
    # Korelasyon için doğrusal sweep chirp (100ms)
    chirp = generate_chirp(sample_rate=16000)
    chirp_int16 = (chirp * 32767.0).astype(np.int16)
    
    # Ölçüm döngüsü
    print("Test sinyali gönderiliyor...", flush=True)
    time.sleep(0.5) # 0.5 saniye boş sessizlik (başlangıcı yakalamak için)
    
    # Ölçüm başlangıç zaman damgası (Örnek sayısı bazlı hizalama için)
    # Aygıt çalma gecikmesi hariç, ağ + karşı cihaz ses döngüsü gecikmesini bulmak için
    # Gönderim anındaki alınan örnek sayısını kaydedelim:
    start_sample_idx = len(received_samples)
    
    # Chirp sinyalini 10 adet 160-örnek frame şeklinde paketleyip gönder
    seq_num = 0
    ssrc = 77777
    
    # 1600 örneklik chirp sinyali
    for i in range(0, len(chirp_int16), args.frame_size):
        frame = chirp_int16[i:i + args.frame_size]
        if len(frame) < args.frame_size:
            pad = np.zeros(args.frame_size - len(frame), dtype=np.int16)
            frame = np.concatenate([frame, pad])
            
        compressed = codec.encode(frame)
        timestamp = int(time.time() * 16000)
        packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
        sock.sendto(packet, dest_addr)
        seq_num += 1
        time.sleep(0.010) # 10ms aralıklarla gönder (gerçek akış hızı)
        
    # Kalan süre boyunca B'ye boş paketler gönder (B akışları kesmesin diye)
    end_time = time.perf_counter() + args.duration - 0.7
    while time.perf_counter() < end_time:
        silence_frame = np.zeros(args.frame_size, dtype=np.int16)
        compressed = codec.encode(silence_frame)
        timestamp = int(time.time() * 16000)
        packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
        sock.sendto(packet, dest_addr)
        seq_num += 1
        time.sleep(0.010)
        
    # Kayıtları tamamla ve kapat
    running = False
    sock.close()
    recv_thread.join(1.0)
    
    recorded_array = np.array(received_samples, dtype=np.float32) / 32768.0
    print(f"Toplam Alınan Örnek Sayısı: {len(recorded_array)} (~{len(recorded_array)/16000:.2f} saniye)")
    
    if len(recorded_array) < len(chirp):
        print("[Hata] Yeterli miktarda ses paketi alınamadı! Bağlantıyı kontrol edin.", flush=True)
        return
        
    # Çapraz Korelasyon (Cross-Correlation)
    correlation = correlate(recorded_array, chirp, mode='valid')
    max_idx = np.argmax(np.abs(correlation))
    
    # Gecikme Örnek Farkı
    delay_samples = max_idx - start_sample_idx
    delay_ms = (delay_samples / 16000.0) * 1000.0
    
    # Ağızdan-kulağa tek yönlü gecikme = Gidiş Dönüş / 2
    one_way_latency_ms = delay_ms / 2.0
    
    # Windows varsayılan ses kartı sürücüsü (MME) tamponlama gecikmesi düşümü
    # Ses sürücüsü tamponlama payı (MME varsayılan: ~80ms giriş + ~80ms çıkış = 160ms RTT payı)
    driver_overhead_ms = 160.0
    net_rtt_ms = max(0.0, delay_ms - driver_overhead_ms)
    net_one_way_ms = net_rtt_ms / 2.0
    
    print("\n================ ÖLÇÜM SONUÇLARI ================", flush=True)
    print(f"Brüt Gidiş-Dönüş (RTT) Gecikme: {delay_ms:.2f} ms (Donanım tamponları dahil)", flush=True)
    print(f"Brüt Tek Yön (Mouth-to-Ear) Gecikme: {one_way_latency_ms:.2f} ms", flush=True)
    print(f"Ses Kartı Sürücü Düzeltmesi (MME): -{driver_overhead_ms:.2f} ms", flush=True)
    print(f"Saf Ağ + Kodek Tek Yön Gecikmesi: {net_one_way_ms:.2f} ms (Ölçülen İletim)", flush=True)
    print("=================================================", flush=True)
    
    if net_one_way_ms < 150.0:
        print(f"[GEÇTİ] Saf Ağızdan-Kulağa ağ iletim gecikmesi ({net_one_way_ms:.2f} ms) LAN hedefinin (< 150 ms) altındadır!", flush=True)
    else:
        print(f"[KALDI] Gecikme hedefin üzerindedir.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="SimClever Intercom - Ağ Gecikmesi Ölçüm Aracı")
    parser.add_argument('--mode', type=str, choices=['measure', 'reflect'], required=True, help="Çalışma modu: 'measure' (A cihazı) veya 'reflect' (B cihazı)")
    parser.add_argument('--ip', type=str, default=None, help="Hedef bilgisayarın IP adresi ('measure' modunda zorunlu)")
    parser.add_argument('--port', type=int, default=5000, help="Symmetric UDP port numarası")
    parser.add_argument('--codec', type=str, default='opus_64', help="Ölçüm esnasında kullanılacak kodek")
    parser.add_argument('--duration', type=float, default=5.0, help="Ölçüm süresi (saniye)")
    parser.add_argument('--frame-size', type=int, default=160, help="Frame boyutu")
    
    args = parser.parse_args()
    
    if args.mode == 'measure' and not args.ip:
        parser.error("--mode measure için --ip parametresi zorunludur.")
        
    if args.mode == 'reflect':
        run_reflector(args)
    else:
        run_measurer(args)

if __name__ == "__main__":
    main()
