r"""
SimClever Intercom — Paket 2 Canlı A/B Denoiser Test Scripti (Tam Dupleks)
==========================================================================
Bu script, farklı gürültü bastırma algoritmalarını canlı LAN ortamında
gerçek zamanlı olarak klavye ile değiştirmek için kullanılır.

Kullanım:
    Hoca Paneli (Bu bilgisayar):
    ./venv/Scripts/python.exe plane_b_live/live_ab_test.py --role teacher

    Öğrenci Düğümü (Monster Laptop):
    ./venv/Scripts/python.exe plane_b_live/live_ab_test.py --role student --ip 192.168.1.XX

Canlı testte (Hoca Panelinde):
    - SPACE / ENTER: Bir sonraki denoiser'a geç
    - 0: Ham Ses (none)
    - 1: Spektral Çıkarma (baseline)
    - 2: WebRTC NS Mild (level=1)
    - 3: WebRTC NS Aggressive (level=2)
    - R: Mevcut denoiser ile 10sn kayıt yap (öncesi/sonrası WAV)
    - Q: Çıkış
"""

import os
import sys
import time
import wave
import socket
import struct
import argparse
import threading
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor
from core.denoiser import DenoiserProcessor

# ----- Denoiser Listesi -----
DENOISER_SEQUENCE = [
    ("none",              "Ham Ses — Referans (Gürültü Bastırma Yok)"),
    ("webrtc_mild",       "WebRTC NS Mild (level=1) — Google Üretim Kalitesi ⭐"),
    ("baseline",          "Spektral Çıkarma — Klasik DSP Gürültü Bastırma"),
    ("webrtc_aggressive", "WebRTC NS Aggressive (level=2) — Agresif Alarm/Fan Bastırma"),
    ("wiener",            "Adaptif Wiener Filtresi — Özel Frekans Filtresi"),
    ("pretrigger_gate",   "Pre-Trigger 50ms Ön-Tamponlu Kapılama"),
]

def make_rtp_packet(seq_num, timestamp, ssrc, payload):
    header = struct.pack("!BBHII", 0x80, 96, seq_num & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc)
    return header + payload

def parse_rtp_packet(data):
    if len(data) < 12:
        return None, data
    header = struct.unpack("!BBHII", data[:12])
    return header, data[12:]

def print_banner(current_idx, role_name):
    print("\n" + "="*70, flush=True)
    print(f"   SIMCLEVER INTERCOM — CANLI A/B TEST PANENELİ [{role_name.upper()}]", flush=True)
    print("="*70, flush=True)
    print(f" Aktif Denoiser: [{current_idx}] -> {DENOISER_SEQUENCE[current_idx][1]}", flush=True)
    print("-"*70, flush=True)
    print(" [SPACE/ENTER] : Sıradaki Algoritmaya Geç", flush=True)
    print(" [0-3]         : Doğrudan Algoritma Seç (0=None, 1=WebRTC Mild, 2=Spektral, 3=Aggressive)", flush=True)
    print(" [R]           : 10 Saniyelik Ses Kaydı Al (Öncesi/Sonrası WAV)", flush=True)
    print(" [Q]           : Çıkış", flush=True)
    print("="*70 + "\n", flush=True)

def save_wav(filename, buffer_list, sample_rate=16000):
    if not buffer_list:
        return
    audio_data = np.concatenate(buffer_list)
    pcm_int16 = np.clip(audio_data * 32767.0, -32768.0, 32767.0).astype(np.int16)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_evidence")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())

def run_teacher(args):
    sample_rate = 16000
    frame_size = args.frame_size
    
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=frame_size)
    vad = VADProcessor(method='none', frame_size=frame_size)
    
    current_idx = [0]
    denoiser_holder = [DenoiserProcessor(method=DENOISER_SEQUENCE[0][0], frame_size=frame_size, sample_rate=sample_rate)]
    recording = [False]
    record_buffer_raw = []
    record_buffer_clean = []
    
    print_banner(0, "HOCA")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
            
    queue_lock = threading.Lock()
    play_queue = []
    send_queue = []
    aec_ref_queue = []
    client_addr = [None]
    running = True
    seq_num = 0
    ssrc = 12345
    
    # 1. Alıcı Ağ İş Parçacığı
    def network_receiver():
        while running:
            sock.settimeout(1.0)
            try:
                data, addr = sock.recvfrom(4096)
                client_addr[0] = addr  # Dinamik Öğrenci IP ve Port adresi
                
                if data == b"ping":
                    continue
                    
                _, payload = parse_rtp_packet(data)
                pcm_frame = codec.decode(payload, frame_size)
                if pcm_frame is None or len(pcm_frame) != frame_size:
                    continue
                
                raw_float = pcm_frame.astype(np.float32) / 32768.0
                denoised = denoiser_holder[0].process(raw_float)
                denoised = np.clip(denoised, -1.0, 1.0)
                
                if recording[0]:
                    record_buffer_raw.append(raw_float.copy())
                    record_buffer_clean.append(denoised.copy())
                
                with queue_lock:
                    play_queue.append(denoised)
                    if len(play_queue) > 25:
                        play_queue.pop(0)
            except socket.timeout:
                continue
            except Exception:
                continue
    
    recv_thread = threading.Thread(target=network_receiver, daemon=True)
    recv_thread.start()

    # 2. Gönderici Ağ İş Parçacığı
    def network_sender():
        while running:
            if len(send_queue) > 0 and client_addr[0] is not None:
                with queue_lock:
                    packet = send_queue.pop(0)
                try:
                    sock.sendto(packet, client_addr[0])
                except Exception:
                    pass
            else:
                time.sleep(0.001)

    send_thread = threading.Thread(target=network_sender, daemon=True)
    send_thread.start()

    # 3. Klavye Giriş İş Parçacığı
    def input_handler():
        nonlocal running
        while running:
            try:
                key = input().strip().lower()
                if key == 'q':
                    print("\n[Hoca] Çıkış yapılıyor...", flush=True)
                    running = False
                    break
                elif key == 'r':
                    if not recording[0]:
                        record_buffer_raw.clear()
                        record_buffer_clean.clear()
                        recording[0] = True
                        print("[Kayıt] ▶ 10 saniye kayıt başladı...", flush=True)
                        time.sleep(10)
                        recording[0] = False
                        ts = int(time.time())
                        method = DENOISER_SEQUENCE[current_idx[0]][0]
                        save_wav(f"ab_raw_{method}_{ts}.wav", record_buffer_raw)
                        save_wav(f"ab_denoised_{method}_{ts}.wav", record_buffer_clean)
                        print(f"[Kayıt] ✅ Öncesi/Sonrası WAV kaydedildi: ab_raw_{method}_{ts}.wav", flush=True)
                elif key == ' ' or key == '':
                    current_idx[0] = (current_idx[0] + 1) % len(DENOISER_SEQUENCE)
                    method = DENOISER_SEQUENCE[current_idx[0]][0]
                    denoiser_holder[0] = DenoiserProcessor(method=method, frame_size=frame_size, sample_rate=sample_rate)
                    print_banner(current_idx[0], "HOCA")
                elif key.isdigit() and 0 <= int(key) < len(DENOISER_SEQUENCE):
                    current_idx[0] = int(key)
                    method = DENOISER_SEQUENCE[current_idx[0]][0]
                    denoiser_holder[0] = DenoiserProcessor(method=method, frame_size=frame_size, sample_rate=sample_rate)
                    print_banner(current_idx[0], "HOCA")
            except EOFError:
                break
    
    input_thread = threading.Thread(target=input_handler, daemon=True)
    input_thread.start()

    # 4. Ses Oynatma ve Mikrofon Yakalama
    import sounddevice as sd
    
    def playback_callback(outdata, frames, time_info, status):
        with queue_lock:
            if len(play_queue) > 0:
                frame = play_queue.pop(0)
            else:
                frame = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = frame
        with queue_lock:
            aec_ref_queue.append(frame.copy())
            if len(aec_ref_queue) > 10:
                aec_ref_queue.pop(0)

    def capture_callback(indata, frames, time_info, status):
        nonlocal seq_num
        near_frame = indata[:, 0].copy()
        with queue_lock:
            far_frame = aec_ref_queue.pop(0) if aec_ref_queue else np.zeros(frames, dtype=np.float32)
        
        near_int16 = np.clip(near_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        far_int16 = np.clip(far_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)

        cleaned_int16 = aec.process(near_int16, far_int16)
        compressed = codec.encode(cleaned_int16)
        
        timestamp = int(time.time() * sample_rate)
        rtp_packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
        seq_num += 1
        
        with queue_lock:
            send_queue.append(rtp_packet)
            if len(send_queue) > 15:
                send_queue.pop(0)

    input_stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=frame_size, callback=capture_callback)
    output_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=frame_size, callback=playback_callback)
    
    print("[Hoca] Tam Dupleks A/B Ses Paneli Başlatıldı. Öğrenciyi bekliyorum...", flush=True)
    
    with input_stream, output_stream:
        try:
            while running:
                time.sleep(0.1)
        except (KeyboardInterrupt, SystemExit):
            running = False

def run_student(args):
    sample_rate = 16000
    frame_size = args.frame_size
    
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=frame_size)
    vad = VADProcessor(method='none', frame_size=frame_size)
    
    print("[Öğrenci] M2 mimarisinde çalışıyor — Tam Dupleks Aktif.", flush=True)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0)) # İşletim sistemi dinamik port atasın
    if sys.platform == 'win32':
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass
    
    dest_addr = (args.ip, args.port)
    queue_lock = threading.Lock()
    send_queue = []
    play_queue = []
    aec_ref_queue = []
    running = True
    seq_num = 0
    ssrc = 54321
    
    def network_receiver():
        while running:
            sock.settimeout(1.0)
            try:
                data, _ = sock.recvfrom(4096)
                if data == b"ping":
                    continue
                _, payload = parse_rtp_packet(data)
                pcm_frame = codec.decode(payload, frame_size)
                if pcm_frame is None or len(pcm_frame) != frame_size:
                    continue
                pcm_float = pcm_frame.astype(np.float32) / 32768.0
                pcm_float = np.clip(pcm_float, -1.0, 1.0)
                with queue_lock:
                    play_queue.append(pcm_float)
                    if len(play_queue) > 25:
                        play_queue.pop(0)
            except socket.timeout:
                continue
            except Exception:
                continue
                
    def network_sender():
        while running:
            if len(send_queue) > 0:
                with queue_lock:
                    packet = send_queue.pop(0)
                try:
                    sock.sendto(packet, dest_addr)
                except Exception:
                    pass
            else:
                time.sleep(0.001)

    def heartbeat_sender():
        while running:
            try:
                sock.sendto(b"ping", dest_addr)
            except Exception:
                pass
            time.sleep(1.0) # Her 1 saniyede bir ping yolla (Güvenlik duvarını açık tutar)
    
    recv_thread = threading.Thread(target=network_receiver, daemon=True)
    send_thread = threading.Thread(target=network_sender, daemon=True)
    heart_thread = threading.Thread(target=heartbeat_sender, daemon=True)
    
    recv_thread.start()
    send_thread.start()
    heart_thread.start()
    
    import sounddevice as sd
    
    def playback_callback(outdata, frames, time_info, status):
        with queue_lock:
            if len(play_queue) > 0:
                frame = play_queue.pop(0)
            else:
                frame = np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = frame
        with queue_lock:
            aec_ref_queue.append(frame.copy())
            if len(aec_ref_queue) > 10:
                aec_ref_queue.pop(0)
    
    def capture_callback(indata, frames, time_info, status):
        nonlocal seq_num
        near_frame = indata[:, 0].copy()
        with queue_lock:
            far_frame = aec_ref_queue.pop(0) if aec_ref_queue else np.zeros(frames, dtype=np.float32)
        
        near_int16 = np.clip(near_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
        far_int16 = np.clip(far_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)

        cleaned_int16 = aec.process(near_int16, far_int16)
        compressed = codec.encode(cleaned_int16)
        
        timestamp = int(time.time() * sample_rate)
        rtp_packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
        seq_num += 1
        
        with queue_lock:
            send_queue.append(rtp_packet)
            if len(send_queue) > 15:
                send_queue.pop(0)
    
    print(f"[Öğrenci] Hoca {args.ip}:{args.port} adresine bağlanıyor...", flush=True)
    
    input_stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=frame_size, callback=capture_callback)
    output_stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=frame_size, callback=playback_callback)
    
    with input_stream, output_stream:
        print("[Öğrenci] Tam dupleks çalışıyor. Kapatmak için Ctrl+C", flush=True)
        try:
            while running:
                time.sleep(0.5)
        except (KeyboardInterrupt, SystemExit):
            running = False

def main():
    parser = argparse.ArgumentParser(
        description="SimClever Intercom — Paket 2 Canlı A/B Denoiser Test Scripti",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ip', type=str, default='127.0.0.1', help="Hoca IP adresi (öğrenci modunda zorunlu)")
    parser.add_argument('--port', type=int, default=5000, help="Haberleşme UDP portu (varsayılan: 5000)")
    parser.add_argument('--role', type=str, choices=['teacher', 'student'], default='teacher', help="Rol: teacher veya student")
    parser.add_argument('--codec', type=str, default='pcm', help="Kodek (pcm, opus, g711a, g711u)")
    parser.add_argument('--aec', type=str, default='webrtc', help="AEC (webrtc, speex, numba, delay, none)")
    parser.add_argument('--frame-size', type=int, default=160, help="Frame boyutu (örnek sayısı)")
    
    args = parser.parse_args()
    
    if args.role == 'teacher':
        run_teacher(args)
    else:
        run_student(args)

if __name__ == "__main__":
    main()
