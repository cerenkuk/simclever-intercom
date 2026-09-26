import os
import sys
import time
import socket
import struct
import threading
import queue
import re
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import sounddevice as sd

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.codec import IntercomCodec
from core.aec import AECProcessor
from core.vad import VADProcessor
from core.denoiser import DenoiserProcessor

# Opsiyonel WAN Sinyalleşme ve Röle modülleri
try:
    from core.signaling import WANSignalingClient, get_public_ip_via_stun
    from core.relay import AudioRelayServer, AudioRelayClient
    HAS_WAN_MODULES = True
except ImportError:
    HAS_WAN_MODULES = False


def make_rtp_packet(seq_num, timestamp, ssrc, payload):
    header = struct.pack("!BBHII", 0x80, 96, seq_num & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc)
    return header + payload


def parse_rtp_packet(packet):
    if len(packet) < 12:
        return None, None, packet
    header = struct.unpack("!BBHII", packet[:12])
    seq_num = header[2]
    ssrc_val = header[4]
    payload = packet[12:]
    return seq_num, ssrc_val, payload


if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


def get_local_ips():
    """Bilgisayardaki tüm aktif yerel IP adreslerini konsol penceresi açmadan ve kasılma olmadan listeler."""
    ips = set()
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        main_ip = s.getsockname()[0]
        s.close()
        if not main_ip.startswith("127."):
            ips.add(main_ip)
    except Exception:
        pass

    res = list(ips)
    return res if res else ["127.0.0.1"]


def get_all_discovery_targets(local_ips):
    """Mobil Hotspot (iPhone/Android) ve tüm Wi-Fi ağlarında AP isolation engelini aşan hedef IP ve multicast listesi."""
    targets = set()
    targets.add("255.255.255.255")
    targets.add("224.0.0.1")    # Multicast All Hosts
    targets.add("224.0.0.251")  # Multicast mDNS
    for lip in local_ips:
        parts = lip.split(".")
        if len(parts) == 4:
            subnet_prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
            targets.add(f"{subnet_prefix}255")
            # Mobil Hotspot için ağ geçitleri ve hızlı IP aralıkları
            for host_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]:
                targets.add(f"{subnet_prefix}{host_id}")
    return list(targets)



class SimCleverIntercomGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SimClever Intercom — Kontrol Dashboard (LAN UDP)")
        self.root.geometry("960x820")
        self.root.minsize(880, 720)

        # Renk Paleti (Koyu Modern Dark Theme)
        self.bg_color = "#181825"
        self.card_bg = "#1e1e2e"
        self.accent_blue = "#89b4fa"
        self.accent_green = "#a6e3a1"
        self.accent_red = "#f38ba8"
        self.accent_yellow = "#f9e2af"
        self.text_color = "#cdd6f4"
        self.subtext_color = "#a6adc8"

        self.root.configure(bg=self.bg_color)

        # Çalışma Durum Değişkenleri
        self.is_running = False
        self.app_closing = False
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.queue_lock = threading.Lock()

        # Kesintisiz Canlı Rol Geçişü SSRC Referansı
        self.current_ssrc_ref = [12345]

        # Takılmasız Canlı Algoritma Hot-Swap Referansları
        self.active_codec = [IntercomCodec(method="pcm")]
        self.active_denoiser = [DenoiserProcessor(method="none", frame_size=160, sample_rate=16000)]
        self.active_aec = [AECProcessor(method="webrtc", frame_size=160)]

        # PTT (Bas-Konuş) State
        self.ptt_active = False

        # Metrikler ve Günlük
        self.current_peak = 0.0
        self.current_latency = 0
        self._last_vu_val = 0.0
        self._last_lat_val = 0
        self.log_queue = queue.Queue()
        self.log_lock = threading.Lock()

        # Ağ Bilgileri
        self.local_ips = get_local_ips()
        self.public_ip = "Algılanıyor..."

        # Stiller ve Arayüz Oluşturma
        self.setup_styles()
        self.build_ui()

        # Arka Plan İşlemleri
        threading.Thread(target=self._resolve_public_ip, daemon=True).start()
        self._start_global_beacon_listener()
        self._start_continuous_background_tasks()

        # GUI Periyodik Güncelleme Döngüsü
        self.root.after(100, self._periodic_gui_update)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.log("Sistem hazır. Modunuzu seçip 'YAYINI BAŞLAT' butonuna basın.")

    def _resolve_public_ip(self):
        if HAS_WAN_MODULES:
            try:
                self.public_ip = get_public_ip_via_stun()
            except Exception:
                self.public_ip = "Yerel Ağ (WAN Servis Yok)"
        else:
            self.public_ip = "LAN Modu Aktif"

    def _start_continuous_background_tasks(self):
        """Ağ değişikliklerini otomatik algılayan ve Mobil Hotspot dahil önceden cihaz bulan arka plan görevi."""
        def bg_loop():
            ping_msg = b"SIMCLEVER_DISCOVERY_PING"
            bsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            bsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            while not self.app_closing:
                try:
                    # 1. Otomatik Ağ IP Değişimi Kontrolü (Wi-Fi değiştiğinde anında güncelle)
                    fresh_ips = get_local_ips()
                    if set(fresh_ips) != set(self.local_ips):
                        self.local_ips = fresh_ips
                        def update_ui_ips():
                            self.ip_info_label.config(text=f"Aktif Canlı IP: {', '.join(self.local_ips)}")
                            clean_vals = [v for v in self.ip_cb["values"] if v not in self.local_ips and v not in ["127.0.0.1", "0.0.0.0"]]
                            self.ip_cb["values"] = clean_vals
                            self.log(f"🌐 Yeni Ağ Algılandı! Güncel Canlı IP'ler: {', '.join(self.local_ips)}")
                        self.root.after(0, update_ui_ips)

                    # 2. Mobil Hotspot & Wi-Fi Ağlarında Kesintisiz Arka Plan Ön-Araması
                    targets = get_all_discovery_targets(self.local_ips)
                    for target in targets:
                        try:
                            bsock.sendto(ping_msg, (target, 5001))
                            bsock.sendto(ping_msg, (target, 5000))
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(2.5)

            try:
                bsock.close()
            except Exception:
                pass

        threading.Thread(target=bg_loop, daemon=True).start()

    def _start_global_beacon_listener(self):
        """Ağdaki tüm arama sorgularına (UDP Port 5001) yanıt veren dinleyici."""
        def listener_loop():
            bsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            bsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            bsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            try:
                bsock.bind(("0.0.0.0", 5001))
            except Exception:
                return

            bsock.settimeout(1.0)
            while not self.app_closing:
                try:
                    data, addr = bsock.recvfrom(512)
                    if data in [b"SIMCLEVER_DISCOVERY_PING", b"SIMCLEVER_DISCOVERY_STUDENT_PING", b"ping"]:
                        if addr[0] in self.local_ips:
                            continue

                        reply_msg = f"SIMCLEVER_NODE_HERE:{self.role_var.get()}:{self.local_ips[0]}".encode()
                        try:
                            bsock.sendto(reply_msg, addr)
                        except Exception:
                            pass

                        def auto_add_found():
                            if addr[0] not in self.local_ips and addr[0] not in ["127.0.0.1", "0.0.0.0"]:
                                curr = [v for v in self.ip_cb["values"] if v not in self.local_ips and v not in ["127.0.0.1", "0.0.0.0"]]
                                if addr[0] not in curr:
                                    curr.append(addr[0])
                                    self.ip_cb["values"] = curr
                                    if not self.ip_cb.get().strip():
                                        self.ip_cb.set(addr[0])
                                    self.log(f"🔗 Ağda Karşı Cihaz Algılandı → Combobox'a Eklendi: {addr[0]}")
                        self.root.after(0, auto_add_found)

                except socket.timeout:
                    continue
                except Exception:
                    break
            bsock.close()

        threading.Thread(target=listener_loop, daemon=True).start()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.card_bg, relief="flat", borderwidth=1)
        style.configure("TLabel", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=self.bg_color, foreground=self.accent_blue, font=("Segoe UI", 15, "bold"))
        style.configure("SubHeader.TLabel", background=self.card_bg, foreground=self.accent_yellow, font=("Segoe UI", 11, "bold"))
        style.configure("TRadiobutton", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10, "bold"))
        style.configure("TCheckbutton", background=self.card_bg, foreground=self.text_color, font=("Segoe UI", 10, "bold"))
        style.configure("TCombobox", fieldbackground=self.bg_color, background=self.card_bg, foreground=self.text_color)
        style.map("TCombobox", fieldbackground=[("readonly", self.bg_color)], foreground=[("readonly", self.text_color)])

    def build_ui(self):
        # 1. ÜST HEADER
        header_frame = tk.Frame(self.root, bg=self.bg_color, pady=10)
        header_frame.pack(fill="x", padx=20)

        title_label = ttk.Label(header_frame, text="SimClever Intercom — Kontrol Dashboard (LAN UDP)", style="Header.TLabel")
        title_label.pack(side="left")

        ip_str = ", ".join(self.local_ips)
        self.ip_info_label = tk.Label(
            header_frame,
            text=f"Aktif Canlı IP: {ip_str}",
            bg=self.bg_color,
            fg=self.accent_green,
            font=("Segoe UI", 9, "bold")
        )
        self.ip_info_label.pack(side="right", pady=5)

        # 2. ANA İÇERİK PANELERİ
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill="both", expand=True, padx=15, pady=5)

        left_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=5)

        right_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        right_card.pack(side="right", fill="both", expand=True, padx=(10, 0), pady=5)

        # ------------------- SOL KOLON: KONTROLLER ------------------- #
        
        # Section 1: Düğüm Rolü Seçimi
        ttk.Label(left_card, text="1. Düğüm Rolü Seçimi", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))
        self.role_var = tk.StringVar(value="teacher")

        role_frame = tk.Frame(left_card, bg=self.card_bg)
        role_frame.pack(fill="x", pady=5)

        r_teacher = ttk.Radiobutton(
            role_frame,
            text="🔘 Hoca Paneli (Teacher)",
            value="teacher",
            variable=self.role_var,
            command=self.on_role_change
        )
        r_teacher.pack(side="left", padx=(0, 15))

        r_student = ttk.Radiobutton(
            role_frame,
            text="🎓 Öğrenci Düğümü (Student)",
            value="student",
            variable=self.role_var,
            command=self.on_role_change
        )
        r_student.pack(side="left")

        ttk.Separator(left_card, orient="horizontal").pack(fill="x", pady=8)

        # Section 2: Bağlantı Modu (Sadece Doğrudan LAN UDP)
        ttk.Label(left_card, text="2. Bağlantı Modu: ⚡ Doğrudan LAN UDP (Port 5000)", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))
        self.net_mode_var = tk.StringVar(value="lan")

        m_lan_info = tk.Label(
            left_card,
            text="🔒 Yerel Wi-Fi / Ethernet Ağında Doğrudan UDP İletimi (Ultra Düşük Gecikme)",
            bg=self.card_bg,
            fg=self.accent_blue,
            font=("Segoe UI", 9, "italic")
        )
        m_lan_info.pack(anchor="w", pady=2)

        ttk.Separator(left_card, orient="horizontal").pack(fill="x", pady=8)

        # Section 3: Ağ & IP Ayarları
        ttk.Label(left_card, text="3. Ağ & IP Ayarları (Port: 5000)", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))

        ip_frame = tk.Frame(left_card, bg=self.card_bg)
        ip_frame.pack(fill="x", pady=5)
        ttk.Label(ip_frame, text="Hedef IP:").pack(side="left")

        self.ip_cb = ttk.Combobox(ip_frame, values=[], state="normal", width=18, font=("Consolas", 10))
        self.ip_cb.set("")
        self.ip_cb.pack(side="left", padx=(5, 5))

        self.btn_refresh = tk.Button(
            ip_frame,
            text="🔄 YENİLE",
            bg="#313244",
            fg="white",
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            cursor="hand2",
            command=self.refresh_networks
        )
        self.btn_refresh.pack(side="left", padx=2)

        self.btn_auto_ip = tk.Button(
            ip_frame,
            text="🔍 OTOMATİK BUL",
            bg="#45475a",
            fg=self.accent_blue,
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            cursor="hand2",
            command=self.auto_discover_teacher_ip
        )
        self.btn_auto_ip.pack(side="left", padx=2)

        port_frame = tk.Frame(left_card, bg=self.card_bg)
        port_frame.pack(fill="x", pady=3)
        ttk.Label(port_frame, text="Port (UDP):").pack(side="left")
        self.port_entry = tk.Entry(port_frame, bg="#11111b", fg="white", insertbackground="white", font=("Consolas", 10), width=10)
        self.port_entry.insert(0, "5000")
        self.port_entry.pack(side="left", padx=(10, 0))

        ttk.Separator(left_card, orient="horizontal").pack(fill="x", pady=8)

        # Section 4: Algoritma Seçici (Combobox & Şiddet Ayarı)
        ttk.Label(left_card, text="4. Algoritma Seçici (Canlı Akışta Anında Değişir)", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))

        # Gürültü Bastırıcı Kontrol Kartı
        denoise_card = tk.Frame(left_card, bg=self.card_bg)
        denoise_card.pack(fill="x", pady=4)

        # 1. Satır: Label & Combobox
        denoise_row1 = tk.Frame(denoise_card, bg=self.card_bg)
        denoise_row1.pack(fill="x", pady=2)
        ttk.Label(denoise_row1, text="🔇 Gürültü Bastırıcı:").pack(side="left")

        self.denoiser_cb = ttk.Combobox(
            denoise_row1,
            values=[
                "webrtc_aggressive (Level 2 - Üretim Standardı)",
                "webrtc_mild (Level 1 - Hafif Bastırma)",
                "webrtc_max (Level 3 - Maksimum Bastırma)",
                "webrtc_extreme (Level 3 + Wiener)",
                "none (Ham Ses / Passthrough)"
            ],
            state="readonly",
            width=32
        )
        self.denoiser_cb.set("webrtc_aggressive (Level 2 - Üretim Standardı)")
        self.denoiser_cb.pack(side="right")
        self.denoiser_cb.bind("<<ComboboxSelected>>", self._on_denoiser_change)

        # 2. Satır: Seçili Algoritma Şiddet Slider (%0 - %150) & Canlı Gösterge (Meter)
        denoise_row2 = tk.Frame(denoise_card, bg=self.card_bg)
        denoise_row2.pack(fill="x", pady=(4, 2))

        self.lbl_intensity = ttk.Label(denoise_row2, text="Şiddet: %100")
        self.lbl_intensity.pack(side="left")

        self.denoise_intensity_var = tk.DoubleVar(value=100.0)
        self.denoise_slider = ttk.Scale(
            denoise_row2,
            from_=0,
            to=150,
            orient="horizontal",
            variable=self.denoise_intensity_var,
            command=self._on_intensity_slider_move
        )
        self.denoise_slider.pack(side="left", padx=8, fill="x", expand=True)

        # Canlı Gürültü Temizleme Göstergesi (Meter)
        self.lbl_denoise_pct = ttk.Label(denoise_row2, text="%0 Temizleme", font=("Consolas", 9, "bold"), foreground="#a6e3a1")
        self.lbl_denoise_pct.pack(side="right", padx=(5, 0))

        self.denoise_meter = ttk.Progressbar(denoise_row2, maximum=100, length=80, mode="determinate")
        self.denoise_meter.pack(side="right", padx=2)


        # Ses Kodeği
        codec_frame = tk.Frame(left_card, bg=self.card_bg)
        codec_frame.pack(fill="x", pady=3)
        ttk.Label(codec_frame, text="Ses Kodeği:").pack(side="left")
        self.codec_cb = ttk.Combobox(
            codec_frame,
            values=[
                "pcm (Ham PCM - Kayıpsız / 0-Jitter)",
                "opus_64 (Opus 64kbps Hi-Fi)",
                "opus_16 (Opus 16kbps Düşük Bant)"
            ],
            state="readonly",
            width=32
        )
        self.codec_cb.set("pcm (Ham PCM - Kayıpsız / 0-Jitter)")
        self.codec_cb.pack(side="right")
        self.codec_cb.bind("<<ComboboxSelected>>", self._on_codec_change)

        # Yankı Engelleme (AEC)
        aec_frame = tk.Frame(left_card, bg=self.card_bg)
        aec_frame.pack(fill="x", pady=3)
        ttk.Label(aec_frame, text="Yankı Engelleme (AEC):").pack(side="left")
        self.aec_cb = ttk.Combobox(
            aec_frame,
            values=[
                "webrtc (WebRTC AEC3)",
                "speex (SpeexDSP AEC)",
                "none (Devredışı)"
            ],
            state="readonly",
            width=32
        )
        self.aec_cb.set("webrtc (WebRTC AEC3)")
        self.aec_cb.pack(side="right")
        self.aec_cb.bind("<<ComboboxSelected>>", self._on_aec_change)

        ttk.Separator(left_card, orient="horizontal").pack(fill="x", pady=8)

        # Section 5: PTT & YAYINI BAŞLAT Butonları
        ptt_frame = tk.Frame(left_card, bg=self.card_bg)
        ptt_frame.pack(fill="x", pady=4)

        self.ptt_var = tk.BooleanVar(value=False)
        self.ptt_check = ttk.Checkbutton(
            ptt_frame,
            text="🎙️ Bas-Konuş (PTT) Modunu Aktifleştir",
            variable=self.ptt_var,
            command=self.on_ptt_toggle
        )
        self.ptt_check.pack(side="left")

        # Özel PTT Butonu
        self.btn_ptt_action = tk.Button(
            ptt_frame,
            text="🎙️ BAS KONUŞ (BUTONA BASILI TUTUN)",
            bg="#45475a",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            state="disabled",
            cursor="hand2"
        )
        self.btn_ptt_action.pack(side="right", padx=5)

        # Fare ile Basılı Tutma / Bırakma Olayları
        self.btn_ptt_action.bind("<ButtonPress-1>", self._on_ptt_btn_press)
        self.btn_ptt_action.bind("<ButtonRelease-1>", self._on_ptt_btn_release)

        # YAYINI BAŞLAT BUTONU
        self.btn_toggle = tk.Button(
            left_card,
            text="🚀 YAYINI BAŞLAT",
            bg=self.accent_green,
            fg="#11111b",
            font=("Segoe UI", 12, "bold"),
            activebackground="#a6e3a1",
            relief="flat",
            cursor="hand2",
            pady=8,
            command=self.toggle_intercom
        )
        self.btn_toggle.pack(fill="x", pady=(10, 0))

        # Initial Role Setup (Default Teacher)
        self.on_role_change()

        # ------------------- SAĞ KOLON: METRİKLER VE GÜNLÜK ------------------- #
        ttk.Label(right_card, text="5. Canlı Sinyal & Gecikme Metre", style="SubHeader.TLabel").pack(anchor="w", pady=(0, 5))

        meter_frame = tk.Frame(right_card, bg=self.card_bg)
        meter_frame.pack(fill="x", pady=5)

        ttk.Label(meter_frame, text="Mikrofon Sinyali:").pack(side="left")
        self.vu_bar = ttk.Progressbar(meter_frame, length=180, mode="determinate")
        self.vu_bar.pack(side="left", padx=10)

        self.latency_label = ttk.Label(meter_frame, text="Gecikme: 0 ms", font=("Segoe UI", 10, "bold"))
        self.latency_label.pack(side="right")

        ttk.Label(right_card, text="Canlı Sistem ve Paket Günlüğü", style="SubHeader.TLabel").pack(anchor="w", pady=(15, 5))

        self.log_area = scrolledtext.ScrolledText(
            right_card,
            bg="#11111b",
            fg=self.accent_green,
            font=("Consolas", 9),
            insertbackground="white",
            relief="flat"
        )
        self.log_area.pack(fill="both", expand=True, pady=(5, 0))

    # ------------------- EVENT & CALLBACK METODLARI ------------------- #

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        with self.log_lock:
            self.log_queue.put(timestamp + str(message) + "\n")

    def _on_intensity_slider_move(self, val):
        pct = int(round(float(val)))
        if hasattr(self, "lbl_intensity"):
            self.lbl_intensity.config(text=f"Şiddet: %{pct}")

    def _periodic_gui_update(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_area.insert("end", msg)
            self.log_area.see("end")

        if abs(self.current_peak - self._last_vu_val) > 1.0:
            self.vu_bar["value"] = min(100.0, self.current_peak)
            self._last_vu_val = self.current_peak
        elif self.current_peak == 0.0 and self._last_vu_val > 0.0:
            self.vu_bar["value"] = 0
            self._last_vu_val = 0.0

        if self.current_latency != self._last_lat_val:
            self.latency_label.config(text=f"Gecikme: {self.current_latency} ms")
            self._last_lat_val = self.current_latency

        if hasattr(self, "denoise_meter"):
            red_val = getattr(self, "current_denoise_reduction", 0.0)
            self.denoise_meter["value"] = red_val
            self.lbl_denoise_pct.config(text=f"%{int(red_val)} Temizleme")

        if not self.app_closing:
            self.root.after(100, self._periodic_gui_update)

    def _on_denoiser_change(self, event=None):
        new_name = self.denoiser_cb.get().split()[0]
        def _bg():
            try:
                proc = DenoiserProcessor(method=new_name, frame_size=160, sample_rate=16000)
                self.active_denoiser[0] = proc
                self.log(f"⚡ Canlı Akışta Gürültü Bastırıcı Değiştirildi → {new_name}")
            except Exception as e:
                self.log(f"Denoise Değişim Hatası: {e}")
        threading.Thread(target=_bg, daemon=True).start()

    def _on_codec_change(self, event=None):
        new_name = self.codec_cb.get().split()[0]
        def _bg():
            try:
                proc = IntercomCodec(method=new_name)
                self.active_codec[0] = proc
                self.log(f"⚡ Canlı Akışta Ses Kodeği Değiştirildi → {new_name}")
            except Exception as e:
                self.log(f"Kodek Değişim Hatası: {e}")
        threading.Thread(target=_bg, daemon=True).start()

    def _on_aec_change(self, event=None):
        new_name = self.aec_cb.get().split()[0]
        def _bg():
            try:
                proc = AECProcessor(method=new_name, frame_size=160)
                self.active_aec[0] = proc
                self.log(f"⚡ Canlı Akışta AEC Değiştirildi → {new_name}")
            except Exception as e:
                self.log(f"AEC Değişim Hatası: {e}")
        threading.Thread(target=_bg, daemon=True).start()

    def on_role_change(self):
        role = self.role_var.get()
        # Kesintisiz Canlı Rol Geçişi SSRC Güncellemesi
        self.current_ssrc_ref[0] = 12345 if role == "teacher" else 54321

        # Hem Hoca hem Öğrenci düğümünde tüm Gürültü Bastırma algoritmaları aktif!
        all_denoisers = [
            "webrtc_aggressive (Level 2 - Üretim Standardı)",
            "webrtc_mild (Level 1 - Hafif Bastırma)",
            "webrtc_max (Level 3 - Maksimum Bastırma)",
            "webrtc_extreme (Level 3 + Wiener)",
            "none (Ham Ses / Passthrough)"
        ]
        self.denoiser_cb["values"] = all_denoisers

        if role == "teacher":
            self.log("Rol Değiştirildi: HOCA PANELİ (Akış Kesintisiz Sürdürülüyor)")
            self.btn_auto_ip.config(state="disabled")
        else:
            self.log("Rol Değiştirildi: ÖĞRENCİ DÜĞÜMÜ → Anında Otomatik IP Taraması Başlatıldı!")
            self.btn_auto_ip.config(state="normal")
            # Öğrenci seçildiğinde Hoca/Karşı Cihaz IP'sini ANINDA otomatik bul!
            self.auto_discover_teacher_ip()

    def on_mode_change(self):
        self.log("⚡ Bağlantı Modu: LAN DİREKT UDP (Aynı Wi-Fi / Ethernet Ağında Ultra Düşük Gecikme)")

    def on_ptt_toggle(self):
        if self.ptt_var.get():
            self.btn_ptt_action.config(state="normal", bg="#89b4fa", fg="#11111b")
            self.log("🎙️ PTT Modu Aktif: Konuşmak için aşağıdaki PTT Butonuna basılı tutun.")
        else:
            self.btn_ptt_action.config(state="disabled", bg="#45475a", fg="white", text="🎙️ BAS KONUŞ (BUTONA BASILI TUTUN)")
            self.ptt_active = False
            self.log("🎙️ PTT Modu Devre Dışı: Sürekli İletim Modu.")

    def _on_ptt_btn_press(self, event):
        if self.ptt_var.get():
            self.ptt_active = True
            self.btn_ptt_action.config(bg=self.accent_green, text="🎙️ KONUŞULUYOR (SES İLETİLİYOR)")

    def _on_ptt_btn_release(self, event):
        if self.ptt_var.get():
            self.ptt_active = False
            self.btn_ptt_action.config(bg="#89b4fa", text="🎙️ BAS KONUŞ (BUTONA BASILI TUTUN)")

    def refresh_networks(self):
        def _bg_worker():
            self.local_ips = get_local_ips()
            self.root.after(0, self._apply_refreshed_ips)

        threading.Thread(target=_bg_worker, daemon=True).start()

    def _apply_refreshed_ips(self):
        self.ip_info_label.config(text=f"Aktif Canlı IP: {', '.join(self.local_ips)}")
        # Combobox'tan kendi yerel IP'lerimizi temizle ki kendimize bağlanmayalım
        clean_vals = [v for v in self.ip_cb["values"] if v not in self.local_ips and v not in ["127.0.0.1", "0.0.0.0"]]
        self.ip_cb["values"] = clean_vals
        if self.ip_cb.get().strip() in self.local_ips:
            self.ip_cb.set(clean_vals[0] if clean_vals else "")
        self.log(f"🔄 Ağlar Yenilendi! Cihazınızın IP'leri: {', '.join(self.local_ips)}")

    def auto_discover_teacher_ip(self):
        self.log("🔍 Ağda ve Mobil Hotspot'ta otomatik cihaz taraması başlatılıyor...")

        def scan_worker():
            found_ips = []
            try:
                bsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                bsock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                bsock.settimeout(0.3)

                ping_msg = b"SIMCLEVER_DISCOVERY_PING"
                targets = get_all_discovery_targets(self.local_ips)

                for target in targets:
                    try:
                        bsock.sendto(ping_msg, (target, 5001))
                        bsock.sendto(ping_msg, (target, 5000))
                    except Exception:
                        pass

                start = time.time()
                while time.time() - start < 0.8:
                    try:
                        data, addr = bsock.recvfrom(512)
                        if data.startswith(b"SIMCLEVER_") or data == b"ping":
                            if addr[0] not in found_ips and addr[0] not in self.local_ips and addr[0] not in ["127.0.0.1", "0.0.0.0"]:
                                found_ips.append(addr[0])
                    except socket.timeout:
                        break
                    except Exception:
                        break
                bsock.close()
            except Exception as e:
                self.log(f"Arama Hatası: {e}")

            self.root.after(0, lambda: self._apply_discovered_ips(found_ips))

        threading.Thread(target=scan_worker, daemon=True).start()

    def _apply_discovered_ips(self, found_ips):
        clean_found = [ip for ip in found_ips if ip not in self.local_ips and ip not in ["127.0.0.1", "0.0.0.0"]]
        if clean_found:
            curr_vals = [v for v in self.ip_cb["values"] if v not in self.local_ips and v not in ["127.0.0.1", "0.0.0.0"]]
            for ip in clean_found:
                if ip not in curr_vals:
                    curr_vals.append(ip)
            self.ip_cb["values"] = curr_vals
            self.ip_cb.set(clean_found[0])
            self.log(f"✅ Karşı Cihaz Bulundu! Combobox'a eklendi ve seçildi: {clean_found[0]}")
        else:
            self.log("⚠️ Otomatik taramada yeni harici IP bulunamadı. Karşı cihaz yayın başlattığında otomatik eklenecektir.")

    def toggle_intercom(self):
        if not self.is_running:
            self.start_intercom()
        else:
            self.stop_intercom()

    def start_intercom(self):
        if self.is_running:
            self.stop_intercom()
            time.sleep(0.2)

        role = self.role_var.get()
        mode = "lan"
        target_ip = self.ip_cb.get().strip()
        port = int(self.port_entry.get().strip())

        # Kendi IP'sine bağlanmayı kesin olarak engelle
        if target_ip in self.local_ips or target_ip == "127.0.0.1":
            self.log("⚠️ Kendi bilgisayarınızın IP'sine bağlanamazsınız! Karşı cihazın IP'si bekleniyor.")
            target_ip = ""

        self.is_running = True
        self.stop_event.clear()

        self.btn_toggle.config(text="🛑 YAYINI BAŞLATILDI (DURDUR)", bg=self.accent_red, activebackground="#f38ba8")
        role = self.role_var.get()
        mode = "lan"
        target_ip = self.ip_cb.get().strip()
        port = int(self.port_entry.get().strip())

        denoiser_val = self.denoiser_cb.get().split()[0]
        codec_val = self.codec_cb.get().split()[0]
        aec_val = self.aec_cb.get().split()[0]

        self.log(f"Yayın Başlatılıyor... Rol: {role.upper()} | Mod: LAN UDP | IP: {target_ip if target_ip else 'Otomatik Bekleniyor'}:{port}")
        self.log(f"İşlemciler -> Kodek: {codec_val} | AEC: {aec_val} | Denoise: {denoiser_val}")

        def wake_ping():
            try:
                psock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                psock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                targets = get_all_discovery_targets(self.local_ips)
                for t in targets:
                    try:
                        psock.sendto(b"SIMCLEVER_DISCOVERY_PING", (t, 5001))
                        psock.sendto(b"ping", (t, port))
                    except Exception:
                        pass
                psock.close()
            except Exception:
                pass
        threading.Thread(target=wake_ping, daemon=True).start()

        self.worker_thread = threading.Thread(
            target=self.run_audio_node,
            args=(role, mode, target_ip, port, denoiser_val, codec_val, aec_val),
            daemon=True
        )
        self.worker_thread.start()

    def stop_intercom(self):
        self.log("Yayın durduruluyor...")
        self.is_running = False
        self.stop_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            try:
                self.worker_thread.join(timeout=0.5)
            except Exception:
                pass

        self.btn_toggle.config(text="🚀 YAYINI BAŞLAT", bg=self.accent_green, activebackground="#a6e3a1")
        self.current_peak = 0.0
        self.current_latency = 0
        self.vu_bar["value"] = 0
        self.latency_label.config(text="Gecikme: 0 ms")

    def run_audio_node(self, role, mode, target_ip, port, denoiser_method, codec_method, aec_method):
        frame_size = 160
        sample_rate = 16000

        # Başlangıç Algoritma Nesnelerini Hazırla (Arka plan kanalları üzerinden takılmasız canlı swap)
        self.active_codec[0] = IntercomCodec(method=codec_method)
        self.active_aec[0] = AECProcessor(method=aec_method, frame_size=frame_size)
        self.active_denoiser[0] = DenoiserProcessor(method=denoiser_method, frame_size=frame_size, sample_rate=sample_rate)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        try:
            sock.bind(("0.0.0.0", port))
        except Exception as e:
            self.log(f"Soket Bağlantı Hatası (Port {port}): {e}")

        if sys.platform == "win32":
            try:
                sock.ioctl(socket.SIO_UDP_CONNRESET, False)
            except Exception:
                pass

        play_queue = []
        send_queue = []
        send_event = threading.Event()
        queue_lock = threading.Lock()

        dest_addr = [ (target_ip, port) if (target_ip and target_ip not in self.local_ips and target_ip != "127.0.0.1") else None ]

        seq_num = [0]
        self.current_ssrc_ref[0] = 12345 if role == "teacher" else 54321

        aec_ref_queue = []

        # 1. Alıcı Ağ Döngüsü (Pürüzsüz Hece Akışı & 6 Frame ~60ms Tampon)
        def receiver_loop():
            nonlocal dest_addr
            while not self.stop_event.is_set():
                sock.settimeout(0.5)
                try:
                    data, addr = sock.recvfrom(4096)

                    if data in [b"ping", b"SIMCLEVER_DISCOVERY_PING"]:
                        continue

                    parsed_res = parse_rtp_packet(data)
                    if parsed_res[0] is None:
                        continue
                    seq_num_in, pkt_ssrc, payload = parsed_res

                    # Kendi gönderdiğimiz paketi ele
                    if pkt_ssrc == self.current_ssrc_ref[0]:
                        continue

                    # Karşı tarafın IP'sini kaydet
                    if dest_addr[0] is None or dest_addr[0][0] != addr[0]:
                        if addr[0] not in ["127.0.0.1", "0.0.0.0"]:
                            dest_addr[0] = (addr[0], port)
                            self.log(f"🔗 Karşı Cihaz Bağlandı & IP Eşleşti: {addr[0]}:{port}")
                            def update_cb():
                                vals = list(self.ip_cb["values"])
                                if addr[0] not in vals:
                                    vals.append(addr[0])
                                    self.ip_cb["values"] = vals
                                self.ip_cb.set(addr[0])
                            self.root.after(0, update_cb)

                    # Takılmasız Kodek Çözümü
                    pcm_frame = self.active_codec[0].decode(payload, frame_size)
                    if pcm_frame is None or len(pcm_frame) != frame_size:
                        continue

                    pcm_float = pcm_frame.astype(np.float32) / 32768.0

                    # Takılmasız Canlı Gürültü Bastırma & Seçili Algoritmanın Canlı Şiddet Ölçeklemesi
                    cur_den = self.active_denoiser[0]
                    intensity = float(self.denoise_intensity_var.get()) / 100.0 if hasattr(self, "denoise_intensity_var") else 1.0

                    if cur_den.method in ["none", "passthrough"] or intensity <= 0.0:
                        out_float = pcm_float
                        self.current_denoise_reduction = 0.0
                    else:
                        raw_pwr = float(np.mean(pcm_float ** 2))
                        proc_float = cur_den.process(pcm_float)

                        # Seçilen Algoritma Üzerine Şiddet Ölçeklemesi (Lerp Scaling)
                        out_float = pcm_float * (1.0 - intensity) + proc_float * intensity

                        clean_pwr = float(np.mean(out_float ** 2))
                        if raw_pwr > 1e-6:
                            red = max(0.0, min(100.0, (1.0 - np.sqrt(clean_pwr / (raw_pwr + 1e-9))) * 100.0))
                            self.current_denoise_reduction = red
                        else:
                            self.current_denoise_reduction = 0.0

                    out_float = np.clip(out_float, -1.0, 1.0)

                    # DENGELİ TAMPON: Çatırtıları önleyen 6 Frame (~60ms) Cap
                    with queue_lock:
                        play_queue.append(out_float)
                        while len(play_queue) > 6:
                            play_queue.pop(0)

                except socket.timeout:
                    continue
                except Exception:
                    continue

        # 2. Gönderici Ağ Döngüsü (Olay Tetiklemeli - Anında İletim & 0 Hece Gecikmesi)
        def sender_loop():
            while not self.stop_event.is_set():
                send_event.wait(timeout=0.01)
                send_event.clear()
                while len(send_queue) > 0 and not self.stop_event.is_set():
                    with queue_lock:
                        pkt = send_queue.pop(0)
                    try:
                        if dest_addr[0] is not None:
                            sock.sendto(pkt, dest_addr[0])
                        else:
                            sock.sendto(pkt, ("255.255.255.255", port))
                    except Exception:
                        pass

        # 3. Audio Callbacks (Pürüzsüz Hece Akışı - Sıfır Hece Takılması)
        def playback_callback(outdata, frames, time_info, status):
            with queue_lock:
                if len(play_queue) > 0:
                    far_frame = play_queue.pop(0)
                    self.current_latency = int(len(play_queue) * 10) + 10
                else:
                    far_frame = np.zeros(frames, dtype=np.float32)
                    self.current_latency = 0

            outdata[:, 0] = far_frame
            with queue_lock:
                aec_ref_queue.append(far_frame.copy())
                if len(aec_ref_queue) > 10:
                    aec_ref_queue.pop(0)

        def capture_callback(indata, frames, time_info, status):
            near_frame = indata[:, 0].copy()

            peak = np.max(np.abs(near_frame)) * 100.0
            self.current_peak = peak

            with queue_lock:
                if len(aec_ref_queue) > 0:
                    far_frame = aec_ref_queue.pop(0)
                else:
                    far_frame = np.zeros(frames, dtype=np.float32)

            near_int16 = np.clip(near_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)
            far_int16 = np.clip(far_frame * 32767.0, -32768.0, 32767.0).astype(np.int16)

            # Takılmasız Canlı AEC İşleme
            cleaned_int16 = self.active_aec[0].process(near_int16, far_int16)

            # Transmit Koşulu
            should_transmit = not self.ptt_var.get() or self.ptt_active

            if should_transmit:
                compressed = self.active_codec[0].encode(cleaned_int16)
                timestamp = int(time.time() * 16000)
                rtp_pkt = make_rtp_packet(seq_num[0], timestamp, self.current_ssrc_ref[0], compressed)
                seq_num[0] += 1

                with queue_lock:
                    send_queue.append(rtp_pkt)
                    while len(send_queue) > 3:
                        send_queue.pop(0)
                send_event.set()
            else:
                with queue_lock:
                    send_queue.clear()

        t_recv = threading.Thread(target=receiver_loop, daemon=True)
        t_send = threading.Thread(target=sender_loop, daemon=True)
        t_recv.start()
        t_send.start()

        try:
            input_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=capture_callback,
                blocksize=frame_size,
                dtype="float32",
                latency="low"
            )
            output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                callback=playback_callback,
                blocksize=frame_size,
                dtype="float32",
                latency="low"
            )
            input_stream.start()
            output_stream.start()

            self.log("🎙️ Tam dupleks canlı ses akışı başlatıldı.")

            while not self.stop_event.is_set():
                time.sleep(0.1)

            try:
                input_stream.stop()
                input_stream.close()
            except Exception:
                pass
            try:
                output_stream.stop()
                output_stream.close()
            except Exception:
                pass

        except Exception as e:
            self.log(f"Ses Cihazı Hatası: {e}")

        try:
            sock.close()
        except Exception:
            pass
        self.log("Yayın başarıyla sonlandırıldı.")

    def on_close(self):
        self.app_closing = True
        self.stop_intercom()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SimCleverIntercomGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
