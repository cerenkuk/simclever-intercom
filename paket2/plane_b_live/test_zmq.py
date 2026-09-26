import os
import sys
import time
import argparse
import threading
import numpy as np
import zmq

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor

def run_zmq_node(args):
    # Kodek, AEC, VAD başlat
    codec = IntercomCodec(method=args.codec)
    aec = AECProcessor(method=args.aec, frame_size=args.frame_size)
    vad = VADProcessor(method=args.vad, frame_size=args.frame_size)
    
    # ZeroMQ Context
    context = zmq.Context()
    running = True
    seq_num = 0
    
    if args.mode == 'recv':
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 1000)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        
        addr = f"tcp://{args.ip}:{args.port}"
        sock.connect(addr)
        print(f"[ZMQ Node] Bağlanılıyor (SUB): {addr} | Dummy: {args.dummy}", flush=True)
        
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
            
        total_packets = 0
        
        start_time = time.time()
        try:
            while running:
                if args.duration > 0 and time.time() - start_time > args.duration:
                    break
                if sock.poll(100):
                    payload = sock.recv()
                    total_packets += 1
                    
                    pcm_frame = codec.decode(payload, args.frame_size)
                    pcm_float = pcm_frame.astype(np.float32) / 32768.0
                    
                    if not args.dummy:
                        with queue_lock:
                            output_queue.append(pcm_float)
                            if len(output_queue) > 10:
                                output_queue.pop(0)
                else:
                    time.sleep(0.001)

                    
        except KeyboardInterrupt:
            pass
        finally:
            if stream:
                stream.stop()
                stream.close()
            sock.close()
            context.term()
            print(f"[ZMQ Node] Alıcı durduruldu. Paket: {total_packets}", flush=True)
            
    else:
        sock = context.socket(zmq.PUB)
        sock.setsockopt(zmq.SNDHWM, 1000)
        
        addr = f"tcp://*:{args.port}"
        sock.bind(addr)
        print(f"[ZMQ Node] Dinleniyor (PUB): {addr} | Dummy: {args.dummy}", flush=True)
        
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
                    s_t = np.arange(seq_num * args.frame_size, (seq_num + 1) * args.frame_size) / 16000.0
                    frame = 0.5 * np.sin(2 * np.pi * 400.0 * s_t)
                    pcm_frame = (frame * 32767.0).astype(np.int16)
                    time.sleep(0.0098)
                
                cleaned_frame = aec.process(pcm_frame)
                
                if not vad.is_speech(cleaned_frame):
                    seq_num += 1
                    continue
                    
                compressed = codec.encode(cleaned_frame)
                
                seq_num += 1
                try:
                    sock.send(compressed)
                except Exception:
                    pass
                
        except KeyboardInterrupt:
            pass
        finally:
            if stream:
                stream.stop()
                stream.close()
            sock.close()
            context.term()
            print("[ZMQ Node] Gönderici durduruldu.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimClever Intercom ZeroMQ PUB/SUB Test Script")
    parser.add_argument('--mode', type=str, choices=['send', 'recv'], required=True)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--codec', type=str, default='opus_64')
    parser.add_argument('--aec', type=str, default='none')
    parser.add_argument('--vad', type=str, default='none')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--frame-size', type=int, default=160)
    parser.add_argument('--dummy', action='store_true', help="Donanımsız simülasyon modu")
    
    args = parser.parse_args()
    run_zmq_node(args)
