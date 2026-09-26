import os
import sys
import time
import asyncio
import socket
import argparse
import threading
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration
from av import AudioFrame

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor

class IntercomAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self, codec_name, aec_name, vad_name, frame_size=160, dummy=False):
        super().__init__()
        self.codec = IntercomCodec(method=codec_name)
        self.aec = AECProcessor(method=aec_name, frame_size=frame_size)
        self.vad = VADProcessor(method=vad_name, frame_size=frame_size)
        self.frame_size = frame_size
        self.dummy = dummy
        
        self.stream = None
        self.input_queue = []
        self.queue_lock = threading.Lock()
        
        if not self.dummy:
            import sounddevice as sd
            def audio_callback(indata, frames, time_info, status):
                with self.queue_lock:
                    self.input_queue.append(indata[:, 0].copy())
            self.stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=self.frame_size)
            self.stream.start()
        
        self.timestamp = 0

    async def recv(self):
        if not self.dummy:
            while True:
                frame = None
                with self.queue_lock:
                    if len(self.input_queue) > 0:
                        frame = self.input_queue.pop(0)
                if frame is not None:
                    break
                await asyncio.sleep(0.001)
            pcm_frame = (frame * 32767.0).astype(np.int16)
        else:
            await asyncio.sleep(0.0098)
            s_t = np.arange(self.timestamp, self.timestamp + self.frame_size) / 16000.0
            frame = 0.5 * np.sin(2 * np.pi * 400.0 * s_t)
            pcm_frame = (frame * 32767.0).astype(np.int16)
            
        cleaned_frame = self.aec.process(pcm_frame)
        if not self.vad.is_speech(cleaned_frame):
            cleaned_frame.fill(0)
            
        av_frame = AudioFrame.from_ndarray(cleaned_frame.reshape(1, -1), format='s16', layout='mono')
        av_frame.sample_rate = 16000
        av_frame.pts = self.timestamp
        self.timestamp += self.frame_size
        return av_frame
        
    def stop(self):
        super().stop()
        if self.stream:
            self.stream.stop()
            self.stream.close()

async def play_audio_track(track, frame_size=160, dummy=False):
    stream = None
    output_queue = []
    queue_lock = threading.Lock()
    
    if not dummy:
        import sounddevice as sd
        def audio_callback(outdata, frames, time_info, status):
            with queue_lock:
                if len(output_queue) > 0:
                    frame = output_queue.pop(0)
                    outdata[:, 0] = frame
                else:
                    outdata.fill(0)
        stream = sd.OutputStream(samplerate=16000, channels=1, callback=audio_callback, blocksize=frame_size)
        stream.start()
    
    try:
        while True:
            frame = await track.recv()
            if not dummy:
                data = frame.to_ndarray()
                pcm_float = data[0].astype(np.float32) / 32768.0
                with queue_lock:
                    output_queue.append(pcm_float)
                    if len(output_queue) > 10:
                        output_queue.pop(0)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[WebRTC] Oynatma hatası: {e}", flush=True)
    finally:
        if stream:
            stream.stop()
            stream.close()

def configure_socket(sock):
    sock.setblocking(False)
    if sys.platform == 'win32':
        try:
            # Windows'ta UDP bağlantı sıfırlama (WinError 10054) hatalarını kapatır
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except Exception:
            pass

async def main():
    parser = argparse.ArgumentParser(description="SimClever Intercom WebRTC Test Script")
    parser.add_argument('--mode', type=str, choices=['send', 'recv'], required=True)
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--codec', type=str, default='opus_64')
    parser.add_argument('--aec', type=str, default='none')
    parser.add_argument('--vad', type=str, default='none')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--frame-size', type=int, default=160)
    parser.add_argument('--dummy', action='store_true', help="Donanımsız simülasyon modu")
    
    args = parser.parse_args()
    
    # Yerel ağda/localhost'ta STUN sunucu arayışını kapatarak bağlantıyı anında kurmasını sağlarız
    config = RTCConfiguration(iceServers=[])
    pc = RTCPeerConnection(configuration=config)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WebRTC pc] Bağlantı durumu değişti: {pc.connectionState}", flush=True)
        
    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.create_task(play_audio_track(track, args.frame_size, args.dummy))

    loop = asyncio.get_running_loop()

    if args.mode == 'send':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', args.port))
        configure_socket(sock)
        
        track = IntercomAudioTrack(args.codec, args.aec, args.vad, args.frame_size, args.dummy)
        pc.addTrack(track)
        
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        
        offer_str = f"{pc.localDescription.sdp}###{pc.localDescription.type}"
        offer_bytes = offer_str.encode('utf-8')
        
        print("[WebRTC Signaling] Alıcı bekleniyor (ping)...", flush=True)
        while True:
            data, client_addr = await loop.sock_recvfrom(sock, 1024)
            if not data or data != b"ping":
                continue
            break
            
        print(f"[WebRTC Signaling] Alıcı pingi alındı: {client_addr}. Teklif gönderiliyor...", flush=True)
        
        dest_addr = (client_addr[0], args.port + 1)
        for _ in range(5):
            await loop.sock_sendto(sock, offer_bytes, dest_addr)
            await asyncio.sleep(0.1)
            
        print("[WebRTC Signaling] Yanıt bekleniyor...", flush=True)
        try:
            while True:
                ans_bytes, _ = await asyncio.wait_for(loop.sock_recvfrom(sock, 65536), timeout=5.0)
                if not ans_bytes:
                    continue
                break
            sdp_ans, type_ans = ans_bytes.decode('utf-8').split("###")
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_ans, type=type_ans))
            print("[WebRTC] Bağlantı başarıyla kuruldu!", flush=True)
        except asyncio.TimeoutError:
            print("[WebRTC] Sinyalleşme zaman aşımı! Alıcıdan yanıt gelmedi.", flush=True)
        finally:
            sock.close()
        
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', args.port + 1))
        configure_socket(sock)
        
        async def ping_loop():
            for _ in range(120):
                if pc.remoteDescription is not None:
                    break
                try:
                    await loop.sock_sendto(sock, b"ping", (args.ip, args.port))
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            
        asyncio.create_task(ping_loop())
        
        print(f"[WebRTC Signaling] Teklif bekleniyor (0.0.0.0:{args.port + 1})...", flush=True)
        while True:
            offer_bytes, sender_addr = await loop.sock_recvfrom(sock, 65536)
            if not offer_bytes:
                continue
            break
            
        sdp_off, type_off = offer_bytes.decode('utf-8').split("###")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_off, type=type_off))
        
        track = IntercomAudioTrack(args.codec, args.aec, args.vad, args.frame_size, args.dummy)
        pc.addTrack(track)
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        answer_str = f"{pc.localDescription.sdp}###{pc.localDescription.type}"
        answer_bytes = answer_str.encode('utf-8')
        
        dest_addr = (sender_addr[0], args.port)
        for _ in range(5):
            await loop.sock_sendto(sock, answer_bytes, dest_addr)
            await asyncio.sleep(0.1)
            
        print("[WebRTC] Yanıt iletildi, bağlantı bekleniyor...", flush=True)
        
        async def close_later():
            await asyncio.sleep(2.0)
            sock.close()
        asyncio.create_task(close_later())

    if args.duration > 0:
        await asyncio.sleep(args.duration)
    else:
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
            
    track.stop()
    await pc.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
