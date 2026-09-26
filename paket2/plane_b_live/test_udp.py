import os
import sys
import time
import socket
import struct
import argparse
import threading
import numpy as np

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor

# RTP Yardımcı Fonksiyonları
def make_rtp_packet(seq_num, timestamp, ssrc, payload):
    header = struct.pack("!BBHII", 0x80, 96, seq_num & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc)
    return header + payload

def parse_rtp_packet(packet):
    if len(packet) < 12:
        return None, None, None, packet
    header = struct.unpack("!BBHII", packet[:12])
    seq_num = header[2]
    timestamp = header[3]
    ssrc = header[4]
    payload = packet[12:]
    return seq_num, timestamp, ssrc, payload

def run_udp_node(args):
    # Kodek, AEC, VAD başlat
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=args.frame_size)
    vad = VADProcessor(method=args.vad, frame_size=args.frame_size)
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
    
    running = True
    seq_num = 0
    ssrc = 12345
    
    if args.mode == 'recv':
        sock.bind(('0.0.0.0', args.port))
        print(f"[UDP/RTP Node] Dinleniyor: 0.0.0.0:{args.port} | Dummy: {args.dummy}", flush=True)
        
        stream = None
        output_queue = []
        queue_lock = threading.Lock()
        
        if not args.dummy:
            import sounddevice as sd
            def audio_callback(outdata, frames, time_info, status):
                with queue_lock:
                    if len(output_queue) > 0:
                        frame = output_queue.pop(0)
                        outdata[:, 0] = frame
                    else:
                        outdata.fill(0)
            stream = sd.OutputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=args.frame_size)
            stream.start()
            
        last_seq = -1
        lost_packets = 0
        total_packets = 0
        
        start_time = time.time()
        try:
            while running:
                if args.duration > 0 and time.time() - start_time > args.duration:
                    break
                sock.settimeout(1.0)
                try:
                    data, addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue

                
                total_packets += 1
                
                if args.use_rtp:
                    res = parse_rtp_packet(data)
                    if res[0] is None:
                        continue
                    seq, ts, _, payload = res
                    if last_seq != -1 and seq > last_seq + 1:
                        lost_packets += (seq - last_seq - 1)
                    last_seq = seq
                else:
                    payload = data
                
                pcm_frame = codec.decode(payload, args.frame_size)
                pcm_float = pcm_frame.astype(np.float32) / 32768.0
                
                if not args.dummy:
                    with queue_lock:
                        output_queue.append(pcm_float)
                        if len(output_queue) > 10:
                            output_queue.pop(0)
                            
        except KeyboardInterrupt:
            pass
        finally:
            if stream:
                stream.stop()
                stream.close()
            sock.close()
            lost_ratio = (lost_packets / (total_packets + lost_packets)) * 100 if (total_packets + lost_packets) > 0 else 0
            print(f"[UDP/RTP Node] Alıcı durduruldu. Paket: {total_packets}, Kayıp: {lost_packets} (%{lost_ratio:.2f})", flush=True)
            
    else:
        dest_addr = (args.ip, args.port)
        print(f"[UDP/RTP Node] Gönderiliyor: {args.ip}:{args.port} | Dummy: {args.dummy}", flush=True)
        
        stream = None
        input_queue = []
        queue_lock = threading.Lock()
        
        if not args.dummy:
            import sounddevice as sd
            def audio_callback(indata, frames, time_info, status):
                with queue_lock:
                    input_queue.append(indata[:, 0].copy())
            stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=args.frame_size)
            stream.start()
            
        start_time = time.time()
        try:
            while running:
                if args.duration > 0 and time.time() - start_time > args.duration:
                    break
                    
                frame = None
                if not args.dummy:
                    with queue_lock:
                        if len(input_queue) > 0:
                            frame = input_queue.pop(0)
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    pcm_frame = (frame * 32767.0).astype(np.int16)
                else:
                    # Dummy modda 400Hz ton üret
                    t = time.time() - start_time
                    s_t = np.arange(seq_num * args.frame_size, (seq_num + 1) * args.frame_size) / 16000.0
                    frame = 0.5 * np.sin(2 * np.pi * 400.0 * s_t)
                    pcm_frame = (frame * 32767.0).astype(np.int16)
                    # 10ms hızı simüle et
                    time.sleep(0.0098)
                
                cleaned_frame = aec.process(pcm_frame)
                
                if not vad.is_speech(cleaned_frame):
                    # VAD sessizlik algılarsa paket atla (fakat seq_num yine de ilerlesin)
                    seq_num += 1
                    continue
                    
                compressed = codec.encode(cleaned_frame)
                
                if args.use_rtp:
                    timestamp = int((time.time() - start_time) * 16000)
                    packet = make_rtp_packet(seq_num, timestamp, ssrc, compressed)
                else:
                    packet = compressed
                    
                seq_num += 1
                try:
                    sock.sendto(packet, dest_addr)
                except Exception:
                    pass
                
        except KeyboardInterrupt:
            pass
        finally:
            if stream:
                stream.stop()
                stream.close()
            sock.close()
            print("[UDP/RTP Node] Gönderici durduruldu.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimClever Intercom UDP/RTP Test Script")
    parser.add_argument('--mode', type=str, choices=['send', 'recv'], required=True)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--codec', type=str, default='opus_64')
    parser.add_argument('--aec', type=str, default='none')
    parser.add_argument('--vad', type=str, default='none')
    parser.add_argument('--use-rtp', action='store_true')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--frame-size', type=int, default=160)
    parser.add_argument('--dummy', action='store_true', help="Donanımsız simülasyon modu")
    
    args = parser.parse_args()
    run_udp_node(args)
