import json
import logging
import socket
import struct
import sys
import threading
import time
import shutil
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np

from .core.aec import AECProcessor
from .core.codec import IntercomCodec
from .core.denoiser import DenoiserProcessor
from .core.vad import VADProcessor
from .events.event_bus import EventBus
from .recording.recorder import Recorder


def _is_private_ip(ip_str):
    try:
        import ipaddress
        return ipaddress.ip_address(ip_str).is_private
    except Exception:
        return False


def get_local_ips():
    """Bilgisayardaki tüm aktif yerel IP adreslerini listeler (Özel LAN IP'leri öncelikli)."""
    ips = set()
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        logging.getLogger(__name__).debug("Hostname IP lookup failed", exc_info=True)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        main_ip = s.getsockname()[0]
        s.close()
        if not main_ip.startswith("127."):
            ips.add(main_ip)
    except OSError:
        logging.getLogger(__name__).debug("Route-based local IP lookup failed", exc_info=True)

    # Özel LAN IP'lerini (192.168.x.x, 10.x.x.x, 172.16-31.x.x) en başa al
    private_ips = [ip for ip in ips if _is_private_ip(ip)]
    public_ips = [ip for ip in ips if not _is_private_ip(ip)]
    res = private_ips + public_ips
    return res if res else ["127.0.0.1"]


def get_all_discovery_targets(local_ips=None):
    """Sadece yerel ağ (LAN / Hotspot) ve yerel makine broadcast hedeflerini üretir."""
    if local_ips is None:
        local_ips = get_local_ips()
    targets = {"255.255.255.255", "127.0.0.1", "224.0.0.1", "224.0.0.251"}
    for lip in local_ips:
        if lip.startswith("127.") or not _is_private_ip(lip):
            continue
        parts = lip.split(".")
        if len(parts) == 4:
            subnet_prefix = f"{parts[0]}.{parts[1]}.{parts[2]}."
            targets.add(f"{subnet_prefix}255")
            for host_id in [1, 2, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]:
                targets.add(f"{subnet_prefix}{host_id}")
    return list(targets)


def is_self_ip(ip):
    """Verilen IP yerel makineye mi ait kontrol eder."""
    if not ip:
        return True
    clean_ip = str(ip).strip()
    if clean_ip in ("", "127.0.0.1", "0.0.0.0", "localhost"):
        return True
    return clean_ip in get_local_ips()


def is_self_endpoint(ip, port=None, my_port=None):
    """Verilen IP ve port kendi soketimiz mi kontrol eder.

    Port farklıysa (örn. 4999 hoca -> 5001 öğrenci), aynı bilgisayarda olsalar bile
    farklı süreçler olduğundan False döner (bağlantıya ve ses akışına izin verir).
    """
    if not ip:
        return True
    if port is not None and my_port is not None:
        try:
            if int(port) != int(my_port):
                return False
        except (ValueError, TypeError):
            pass
    return is_self_ip(ip)


def query_wasapi_endpoints(data_flow):
    """Windows WASAPI COM arabirimini kullanarak canlı ses uç noktalarını 0 ms'de çeker.
    data_flow: 0 = Render (Hoparlörler/Kulaklıklar), 1 = Capture (Mikrofonlar)
    """
    if sys.platform != "win32":
        return []
    com_initialized = False
    ole32 = None
    try:
        import ctypes
        from ctypes import HRESULT, POINTER, Structure, c_wchar_p, byref
        from ctypes.wintypes import DWORD, WORD, BYTE

        ole32 = ctypes.windll.ole32
        init_result = ole32.CoInitializeEx(None, 0)
        RPC_E_CHANGED_MODE = -2147417850  # 0x80010106 (COM already initialized on thread with different mode)
        if init_result in (0, 1):
            com_initialized = True
        elif init_result == RPC_E_CHANGED_MODE:
            com_initialized = False
        elif init_result < 0:
            return []

        class GUID(Structure):
            _fields_ = [('Data1', DWORD), ('Data2', WORD), ('Data3', WORD), ('Data4', BYTE * 8)]

        class PROPERTYKEY(Structure):
            _fields_ = [('fmtid', GUID), ('pid', DWORD)]

        CLSID_MMDeviceEnumerator = GUID(0xBCDE0395, 0xE52F, 0x467C, (BYTE * 8)(0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E))
        IID_IMMDeviceEnumerator  = GUID(0xA95664D2, 0x9614, 0x4F35, (BYTE * 8)(0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6))
        PKEY_Device_FriendlyName = PROPERTYKEY(
            GUID(0xa45c254e, 0xdf1c, 0x4efd, (BYTE * 8)(0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0)), 14
        )

        class IMMDeviceEnumeratorVtbl(Structure):
            pass
        class IMMDeviceEnumerator(Structure):
            _fields_ = [('lpVtbl', POINTER(IMMDeviceEnumeratorVtbl))]
        class IMMDeviceCollectionVtbl(Structure):
            pass
        class IMMDeviceCollection(Structure):
            _fields_ = [('lpVtbl', POINTER(IMMDeviceCollectionVtbl))]
        class IMMDeviceVtbl(Structure):
            pass
        class IMMDevice(Structure):
            _fields_ = [('lpVtbl', POINTER(IMMDeviceVtbl))]
        class IPropertyStoreVtbl(Structure):
            pass
        class IPropertyStore(Structure):
            _fields_ = [('lpVtbl', POINTER(IPropertyStoreVtbl))]
        class PROPVARIANT(Structure):
            _fields_ = [
                ('vt', WORD), ('wReserved1', WORD), ('wReserved2', WORD), ('wReserved3', WORD), ('pwszVal', c_wchar_p)
            ]

        Release_t            = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
        EnumAudioEndpoints_t = ctypes.WINFUNCTYPE(HRESULT, POINTER(IMMDeviceEnumerator), DWORD, DWORD, POINTER(POINTER(IMMDeviceCollection)))
        GetCount_t           = ctypes.WINFUNCTYPE(HRESULT, POINTER(IMMDeviceCollection), POINTER(DWORD))
        Item_t               = ctypes.WINFUNCTYPE(HRESULT, POINTER(IMMDeviceCollection), DWORD, POINTER(POINTER(IMMDevice)))
        OpenPropertyStore_t  = ctypes.WINFUNCTYPE(HRESULT, POINTER(IMMDevice), DWORD, POINTER(POINTER(IPropertyStore)))
        GetValue_t           = ctypes.WINFUNCTYPE(HRESULT, POINTER(IPropertyStore), POINTER(PROPERTYKEY), POINTER(PROPVARIANT))

        IMMDeviceEnumeratorVtbl._fields_ = [
            ('QueryInterface', ctypes.c_void_p), ('AddRef', ctypes.c_void_p), ('Release', Release_t),
            ('EnumAudioEndpoints', EnumAudioEndpoints_t), ('GetDefaultAudioEndpoint', ctypes.c_void_p),
            ('GetDevice', ctypes.c_void_p), ('RegisterEndpointNotificationCallback', ctypes.c_void_p),
            ('UnregisterEndpointNotificationCallback', ctypes.c_void_p)
        ]
        IMMDeviceCollectionVtbl._fields_ = [
            ('QueryInterface', ctypes.c_void_p), ('AddRef', ctypes.c_void_p), ('Release', Release_t),
            ('GetCount', GetCount_t), ('Item', Item_t)
        ]
        IMMDeviceVtbl._fields_ = [
            ('QueryInterface', ctypes.c_void_p), ('AddRef', ctypes.c_void_p), ('Release', Release_t),
            ('Activate', ctypes.c_void_p), ('OpenPropertyStore', OpenPropertyStore_t),
            ('GetId', ctypes.c_void_p), ('GetState', ctypes.c_void_p)
        ]
        IPropertyStoreVtbl._fields_ = [
            ('QueryInterface', ctypes.c_void_p), ('AddRef', ctypes.c_void_p), ('Release', Release_t),
            ('GetCount', ctypes.c_void_p), ('GetAt', ctypes.c_void_p), ('GetValue', GetValue_t),
            ('SetValue', ctypes.c_void_p), ('Commit', ctypes.c_void_p)
        ]

        def release(interface):
            if interface:
                try:
                    interface.contents.lpVtbl.contents.Release(ctypes.cast(interface, ctypes.c_void_p))
                except Exception:
                    logging.getLogger(__name__).debug("COM interface release failed", exc_info=True)

        enumerator = POINTER(IMMDeviceEnumerator)()
        collection = POINTER(IMMDeviceCollection)()
        try:
            if ole32.CoCreateInstance(byref(CLSID_MMDeviceEnumerator), None, 1, byref(IID_IMMDeviceEnumerator), byref(enumerator)) != 0 or not enumerator:
                return []

            if enumerator.contents.lpVtbl.contents.EnumAudioEndpoints(enumerator, data_flow, 1, byref(collection)) != 0 or not collection:
                return []

            count = DWORD()
            if collection.contents.lpVtbl.contents.GetCount(collection, byref(count)) != 0:
                return []

            names = []
            for i in range(count.value):
                device = POINTER(IMMDevice)()
                store = POINTER(IPropertyStore)()
                prop = PROPVARIANT()
                try:
                    if collection.contents.lpVtbl.contents.Item(collection, i, byref(device)) != 0 or not device:
                        continue
                    if device.contents.lpVtbl.contents.OpenPropertyStore(device, 0, byref(store)) != 0 or not store:
                        continue
                    if store.contents.lpVtbl.contents.GetValue(store, byref(PKEY_Device_FriendlyName), byref(prop)) == 0 and prop.pwszVal:
                        names.append(str(prop.pwszVal))
                finally:
                    try:
                        ole32.PropVariantClear(byref(prop))
                    except Exception:
                        logging.getLogger(__name__).debug("PropVariantClear failed", exc_info=True)
                    release(store)
                    release(device)
            return names
        finally:
            release(collection)
            release(enumerator)
    except Exception:
        logging.getLogger(__name__).debug("WASAPI endpoint query failed", exc_info=True)
        return []
    finally:
        if com_initialized and ole32 is not None:
            try:
                ole32.CoUninitialize()
            except Exception:
                logging.getLogger(__name__).debug("CoUninitialize failed", exc_info=True)


def _rtp_packet(sequence, timestamp, ssrc, payload):
    return struct.pack("!BBHII", 0x80, 96, sequence & 0xFFFF, timestamp & 0xFFFFFFFF, ssrc) + payload


def _rtp_payload(packet):
    if len(packet) < 12:
        return None, None, packet
    header = struct.unpack("!BBHII", packet[:12])
    seq_num = header[2]
    ssrc_val = header[4]
    payload = packet[12:]
    return seq_num, ssrc_val, payload


class IntercomEngine:
    """Modüler, UI-bağımsız tam dupleks ses ve UDP aktarım çekirdeği."""

    DEFAULTS = {
        "schema": 2,
        "role": "teacher",
        "network": {"bind_ip": "0.0.0.0", "target_ip": "", "port": 5000},
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "frame_size": 160,
            "denoise": "webrtc_mild",
            "aec": "webrtc",
            "vad": "none",
            "codec": "pcm",
            "input_device": None,
            "output_device": None,
            "input_gain": 100,
            "output_level": 100,
            "agc": True,
        },
        "recording": {"enabled": True, "folder": "recordings"},
        "ptt": {"mode": "continuous", "muted": False, "talk_warning": True},
        "advanced": {"prevent_sleep": False},
        "profiles": [],
    }

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.event_bus = EventBus()
        self._config_migrated = False
        self.config = self._load_config()
        if self._config_migrated:
            self.save_config()
        self._local_ips = frozenset(get_local_ips())
        self._agc_preference_gain = float(self.config["audio"].get("input_gain", 100)) / 100.0
        self._agc_gain = self._agc_preference_gain
        self._state_lock = threading.RLock()
        self._queue_lock = threading.Lock()
        self._send_event = threading.Event()
        self._stop = threading.Event()
        self._play = deque(maxlen=10)
        self._send = deque(maxlen=10)
        self._far_ref = deque(maxlen=10)
        self._threads = []
        self._socket = None
        self._input_stream = None
        self._output_stream = None
        self._running = False
        self._connected = False
        self._manual_disconnect = False
        self._remote = None
        self._sequence = 0
        self._ssrc = 12345 if self.config["role"] == "teacher" else 54321
        self._sent = 0
        self._received = 0
        self._last_rx = 0.0
        self._last_rx_seq = None
        self._rx_next_expected = None  # Sonraki beklenen RTP sequence (reorder için)
        self._rx_reorder = {}           # {seq_num: decoded_float_frame} bekleyen paketler
        self._last_capture = None
        self._last_level_emit = 0.0
        self._last_talk_warning = 0.0
        self._last_send_warning = 0.0
        self._restarting_input = False
        self._restarting_output = False
        self._muted = bool(self.config.get("ptt", {}).get("muted", False))
        self._ptt_active = False
        self._monitor_mic = False
        self._monitor_frame = None
        self._build_processors()
        self.recorder = Recorder(self._record_cfg(), self.event_bus)

    def _merge(self, base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._merge(base[key], value)
            else:
                base[key] = value

    def _load_config(self):
        cfg = deepcopy(self.DEFAULTS)
        if self.config_path.is_file():
            try:
                with self.config_path.open("r", encoding="utf-8-sig") as stream:
                    loaded = json.load(stream)
                self._migrate_config(loaded)
                self._merge(cfg, loaded)
            except Exception as exc:
                try:
                    shutil.copy2(
                        self.config_path,
                        self.config_path.with_suffix(f".corrupt-{int(time.time())}.json"),
                    )
                except OSError as _e:
                    logging.getLogger(__name__).debug("Corrupt config backup failed", exc_info=True)
                self.event_bus.emit(
                    "warning",
                    {
                        "message": "Ayar dosyası bozuk; yedeklendi ve güvenli varsayılanlar kullanılıyor.",
                        "technical_detail": str(exc),
                    },
                )
        cfg.setdefault("audio", {}).setdefault("denoise", "webrtc_mild")
        cfg.setdefault("audio", {}).setdefault("agc", True)
        cfg.setdefault("ptt", {}).setdefault("talk_warning", True)
        return cfg

    def _migrate_config(self, loaded):
        """Migrate persisted settings to the current name-based device schema."""
        try:
            schema = int(loaded.get("schema", 1))
        except (TypeError, ValueError):
            schema = 1

        if schema >= 2:
            return

        audio = loaded.setdefault("audio", {})
        reset_devices = []
        for key in ("input_device", "output_device"):
            value = audio.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                audio[key] = None
                reset_devices.append(key)

        loaded["schema"] = 2
        self._config_migrated = True
        if reset_devices:
            logging.getLogger(__name__).warning(
                "Legacy numeric audio device settings were reset; select devices again: %s",
                ", ".join(reset_devices),
            )

    def _record_cfg(self):
        cfg = dict(self.config.get("recording", {}))
        folder = Path(cfg.get("folder", "recordings"))
        cfg["folder"] = str(
            folder if folder.is_absolute() else self.config_path.parent.parent / folder
        )
        cfg["sample_rate"] = int(self.config["audio"].get("sample_rate", 16000))
        return cfg

    def save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.config_path.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(self.config, stream, ensure_ascii=False, indent=2)
        temp.replace(self.config_path)

    def _safe_processor(self, factory, fallback, name):
        try:
            return factory()
        except Exception as exc:
            self.event_bus.emit(
                "warning",
                {
                    "message": f"{name} başlatılamadı; güvenli moda geçildi.",
                    "technical_detail": str(exc),
                },
            )
            return fallback()

    def _build_processors(self):
        a = self.config["audio"]
        rate = int(a.get("sample_rate", 16000))
        frame = int(a.get("frame_size", 160))
        self.codec = self._safe_processor(
            lambda: IntercomCodec(method=a.get("codec", "pcm"), sample_rate=rate),
            lambda: IntercomCodec(method="pcm", sample_rate=rate),
            "Codec",
        )
        self.aec = self._safe_processor(
            lambda: AECProcessor(method=a.get("aec", "webrtc"), sample_rate=rate, frame_size=frame),
            lambda: AECProcessor(method="none", sample_rate=rate, frame_size=frame),
            "AEC",
        )
        self.vad = self._safe_processor(
            lambda: VADProcessor(method=a.get("vad", "none"), sample_rate=rate, frame_size=frame),
            lambda: VADProcessor(method="none", sample_rate=rate, frame_size=frame),
            "VAD",
        )
        self.denoiser = DenoiserProcessor(
            method=a.get("denoise", "webrtc_mild"), frame_size=frame, sample_rate=rate
        )

    def update_config(self, changes, save=True):
        old_audio = deepcopy(self.config.get("audio", {}))
        self._merge(self.config, changes)
        audio_changed = old_audio != self.config.get("audio", {})
        if audio_changed:
            with self._state_lock:
                self._build_processors()
        if save:
            self.save_config()
        self.event_bus.emit("config_changed", {"config": deepcopy(self.config)})

    def reload_config(self):
        was_recording = self.recorder.active
        if was_recording:
            self.recorder.stop()
        self.config = self._load_config()
        self._agc_preference_gain = float(self.config["audio"].get("input_gain", 100)) / 100.0
        self._agc_gain = self._agc_preference_gain
        with self._state_lock:
            self._build_processors()
        self.recorder = Recorder(self._record_cfg(), self.event_bus)
        self.set_prevent_sleep(bool(self.config.get("advanced", {}).get("prevent_sleep", False)))
        self.event_bus.emit("config_changed", {"config": deepcopy(self.config)})

    def reset_rx_sequence(self):
        """Alıcı tarafındaki RTP sequence takibini ve reorder buffer'ı sıfırlar."""
        with self._state_lock:
            self._last_rx_seq = None
            self._rx_next_expected = None
            self._rx_reorder.clear()

    def _accept_rx_sequence(self, seq_num):
        """16-bit RTP sequence numarası takibi ve duplicate/eski paket eleme filtresi.
        Kabul edilirse True döner ve _last_rx_seq güncellenir; aksi halde False döner.

        Not: Bu metot orijinal strict monotonic sözleşmesini korur. Reorder
        mantığı _push_to_reorder() / _drain_reorder_to_play() üzerinden yürür;
        _accept_rx_sequence() yalnızca duplicate/stale classification için kullanılır.
        """
        if seq_num is None:
            return False
        with self._state_lock:
            if self._last_rx_seq is not None:
                diff = (seq_num - self._last_rx_seq) & 0xFFFF
                if diff == 0 or diff >= 32768:
                    return False
            self._last_rx_seq = seq_num
            return True

    # ------------------------------------------------------------------
    # Bounded RTP Reorder Buffer (maks 3 paket / ~30 ms)
    # ------------------------------------------------------------------

    _RX_REORDER_WINDOW = 3  # Maksimum bekletilecek paket sayısı

    @staticmethod
    def _seq16_diff(a, b):
        """16-bit modular forward distance: a'dan b'ye kaç adım (0..32767).
        a == b ise 0, a'dan b ileriyse pozitif, a'dan b gerideyse 32769..65535.
        """
        return (b - a) & 0xFFFF

    def _push_to_reorder(self, seq_num, decoded_frame):
        """Gelen decode edilmiş kareyi reorder buffer'a ekler;
        sonra buffer'dan sıradaki kareleri _play kuyruğuna drainlar.
        Caller _state_lock ve _queue_lock'u elinde tutmamalıdır.
        """
        with self._state_lock:
            # İlk paket: beklenen seq'i başlat
            if self._rx_next_expected is None:
                self._rx_next_expected = seq_num

            next_exp = self._rx_next_expected

            # Stale / çok eski paket: pencere başlangıcından 32768+ ileride değil
            dist_from_next = self._seq16_diff(next_exp, seq_num)
            if dist_from_next >= 32768:
                # Geriye doğru paket; duplicate veya çok eski -> sil
                return

            # Zaten buffer'da varsa (duplicate) yoksay
            if seq_num in self._rx_reorder:
                return

            # Buffer'a ekle
            self._rx_reorder[seq_num] = decoded_frame

            # Buffer doluluk sınırı aşıldıysa en eskiyi flush et (kayip paketi atla).
            # Koşul: Buffer pencere boyutuna ulaştı VE beklenen seq buffer'da yok
            # (yani bir sonraki paket gelmemiş, sonsuza kadar beklememeye karar veriyoruz).
            if (len(self._rx_reorder) >= self._RX_REORDER_WINDOW
                    and self._rx_next_expected not in self._rx_reorder):
                # Penceredeki en küçük seq'i bul (16-bit modular)
                oldest = min(
                    self._rx_reorder.keys(),
                    key=lambda s: self._seq16_diff(self._rx_next_expected, s)
                )
                # Beklenen seq'i oldest'e ilerlet; kayip paket(leri) atla
                self._rx_next_expected = oldest

            frames_to_play = self._drain_in_order()

        # _state_lock bırakıldıktan sonra _play'e yaz
        if frames_to_play:
            with self._queue_lock:
                for f in frames_to_play:
                    self._enqueue_play(f)

    def _enqueue_send(self, packet):
        """_queue_lock altinda cagrilir. _send'e ekler ve bounded trim uygular (maks 3 paket)."""
        self._send.append(packet)
        while len(self._send) > 3:
            self._send.popleft()

    def _enqueue_play(self, frame):
        """_queue_lock altinda cagrilir. _play'e ekler ve bounded trim uygular (maks 6 frame)."""
        self._play.append(frame)
        while len(self._play) > 6:
            self._play.popleft()

    def _drain_in_order(self):
        """_state_lock altında çağrılır. _rx_reorder'dan ardışık sıradaki
        kareleri çıkarır ve liste olarak döner; _rx_next_expected güncellenir.
        """
        result = []
        while self._rx_next_expected in self._rx_reorder:
            frame = self._rx_reorder.pop(self._rx_next_expected)
            result.append(frame)
            self._rx_next_expected = (self._rx_next_expected + 1) & 0xFFFF
        return result

    def start(self):
        """YAYINI BAŞLAT: Ses akışlarını ve UDP haberleşmesini açar."""
        if self._running:
            return
        self.reset_rx_sequence()
        self._stop.clear()
        self._manual_disconnect = False
        self._threads.clear()
        with self._queue_lock:
            self._send.clear()
            self._play.clear()
            self._far_ref.clear()
        try:
            self._open_socket()
            self._open_audio()
            self._running = True
            for target, name in (
                (self._receiver, "udp-receiver"),
                (self._sender, "udp-sender"),
                (self._heartbeat, "udp-heartbeat"),
                (self._monitor, "engine-monitor"),
                (self._bg_discovery, "bg-discovery"),
            ):
                thread = threading.Thread(target=target, name=name, daemon=True)
                thread.start()
                self._threads.append(thread)
            self.event_bus.emit("started", {"message": "Gerçek ses ve UDP motoru başlatıldı."})
        except Exception as exc:
            self._close_resources()
            self._running = False
            self.event_bus.emit(
                "error",
                {"message": "Sistem başlatılamadı.", "technical_detail": str(exc)},
            )
            raise

    def stop(self):
        """YAYINI DURDUR: Akışları ve motoru anında durdurur; PTT modunu sıfırlar."""
        self._ptt_active = False
        self.reset_rx_sequence()
        self.recorder.stop()
        self._stop.set()
        self._running = False
        self._send_event.set()
        with self._queue_lock:
            self._send.clear()
            self._play.clear()
            self._far_ref.clear()
        self._close_resources()
        for thread in self._threads:
            try:
                thread.join(timeout=0.5)
            except RuntimeError:
                logging.getLogger(__name__).debug("Engine thread join failed", exc_info=True)
        self._threads.clear()
        self.event_bus.emit("stopped", {"message": "Sistem güvenli biçimde durduruldu."})

    def set_port(self, port_num):
        """UDP Portunu günceller ve soketi rebind eder."""
        port_num = int(port_num)
        self.config.setdefault("network", {})["port"] = port_num
        if self._remote:
            self._remote = (self._remote[0], port_num)
        if self._running or self._socket:
            try:
                self._open_socket()
                self.event_bus.emit("config_changed", {"message": f"UDP Port {port_num} olarak güncellendi."})
            except Exception as exc:
                self.event_bus.emit("warning", {"message": f"Port değiştirilemedi: {exc}"})

    def _open_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                logging.getLogger(__name__).debug("Old socket close failed", exc_info=True)
        n = self.config["network"]
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        except OSError:
            logging.getLogger(__name__).debug("SO_RCVBUF/SO_SNDBUF setsockopt failed (platform limit)", exc_info=True)

        port_num = int(n.get("port", 5000))
        self._socket.bind((n.get("bind_ip", "0.0.0.0"), port_num))
        self._socket.settimeout(0.5)
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            try:
                self._socket.ioctl(socket.SIO_UDP_CONNRESET, False)
            except OSError:
                logging.getLogger(__name__).debug("SIO_UDP_CONNRESET ioctl failed", exc_info=True)
        target = n.get("target_ip", "").strip()
        self._remote = (target, port_num) if target and not self._is_self_endpoint(target, port_num) else None

    def _is_self_ip(self, ip):
        """Check loopback and the discovery-maintained local IP cache."""
        if not ip:
            return True
        clean_ip = str(ip).strip()
        if clean_ip in ("", "127.0.0.1", "0.0.0.0", "localhost"):
            return True
        return clean_ip in self._local_ips

    def _is_self_endpoint(self, ip, port=None):
        """Check if IP and port point to our own listening socket."""
        if not ip:
            return True
        my_port = int(self.config["network"].get("port", 5000))
        if port is not None and int(port) != my_port:
            return False
        return self._is_self_ip(ip)

    def _resolve_device_index(self, dev_name, is_input=True):
        """Cihaz adını KESİN OLARAK kanallarıyla beraber PortAudio indeksine dönüştürür.
        Mikrofon için max_input_channels > 0, Hoparlör/Kulaklık için max_output_channels > 0 şarttır.
        Bulunamadıysa None (Sistem Varsayılanı) döner.
        """
        if not dev_name or dev_name in ("Sistem Varsayılanı", "None", ""):
            return None
        try:
            import sounddevice as sd
            dev_str = str(dev_name).strip()
            
            # 1. Tam isim ve kanal sayısı kontrolü
            for i, d in enumerate(sd.query_devices()):
                chans = d["max_input_channels"] if is_input else d["max_output_channels"]
                if chans > 0 and d["name"].strip() == dev_str:
                    return i
                    
            # 2. Kısmi isim ve kanal sayısı kontrolü
            for i, d in enumerate(sd.query_devices()):
                chans = d["max_input_channels"] if is_input else d["max_output_channels"]
                if chans > 0 and (dev_str in d["name"] or d["name"] in dev_str):
                    return i
        except Exception:
            logging.getLogger(__name__).debug("Audio device resolution failed", exc_info=True)
        return None

    def _open_input_audio(self):
        import sounddevice as sd
        a = self.config["audio"]
        rate = int(a.get("sample_rate", 16000))
        frame = int(a.get("frame_size", 160))
        common = {
            "samplerate": rate,
            "channels": 1,
            "blocksize": frame,
            "dtype": "float32",
            "latency": "low"
        }

        raw_in = a.get("input_device")
        input_index = self._resolve_device_index(raw_in, is_input=True)

        try:
            self._input_stream = sd.InputStream(callback=self._capture_callback, device=input_index, **common)
            self._input_stream.start()
        except Exception as exc:
            self.event_bus.emit("warning", {"message": f"Seçilen mikrofon açılamadı ({exc}); Sistem Varsayılanına geçiliyor."})
            try:
                self._input_stream = sd.InputStream(callback=self._capture_callback, device=None, **common)
                self._input_stream.start()
                self.config["audio"]["input_device"] = None
            except Exception as exc2:
                self.event_bus.emit("error", {"message": f"Mikrofon başlatılamadı: {exc2}"})
                raise

    def _open_output_audio(self):
        import sounddevice as sd
        a = self.config["audio"]
        rate = int(a.get("sample_rate", 16000))
        frame = int(a.get("frame_size", 160))
        common = {
            "samplerate": rate,
            "channels": 1,
            "blocksize": frame,
            "dtype": "float32",
            "latency": "low"
        }

        raw_out = a.get("output_device")
        output_index = self._resolve_device_index(raw_out, is_input=False)

        try:
            self._output_stream = sd.OutputStream(callback=self._playback_callback, device=output_index, **common)
            self._output_stream.start()
        except Exception as exc:
            self.event_bus.emit("warning", {"message": f"Seçilen hoparlör/kulaklık açılamadı ({exc}); Sistem Varsayılanına geçiliyor."})
            try:
                self._output_stream = sd.OutputStream(callback=self._playback_callback, device=None, **common)
                self._output_stream.start()
                self.config["audio"]["output_device"] = None
            except Exception as exc2:
                self.event_bus.emit("error", {"message": f"Hoparlör/Kulaklık başlatılamadı: {exc2}"})
                raise

    def _open_audio(self):
        self._open_input_audio()
        self._open_output_audio()

    def _capture_callback(self, indata, frames, time_info, status):
        if not self._running or self._stop.is_set():
            return
        try:
            raw = indata[:, 0].copy()

            # Otomatik Kazanç Kontrolü (AGC - Dynamic Gain Adaptation)
            gain = self._calculate_input_gain(raw)

            # Kazanç Ölçekleme & Kırpma Önleme
            raw_scaled = np.clip(raw * gain, -1.0, 1.0)
            self._monitor_frame = raw_scaled.copy()

            level = min(100, int(np.max(np.abs(raw_scaled)) * 200))
            now = time.monotonic()
            if now - self._last_level_emit >= 0.05:
                self._last_level_emit = now
                self.event_bus.emit("audio_level", {
                    "input": level,
                    "clipping": level >= 98,
                    "gain": int(round(gain * 100.0))
                })

            with self._queue_lock:
                far = self._far_ref.popleft() if self._far_ref else np.zeros(frames, dtype=np.float32)

            near_i16 = np.clip(raw_scaled * 32767.0, -32768.0, 32767.0).astype(np.int16)
            far_i16 = np.clip(far * 32767.0, -32768.0, 32767.0).astype(np.int16)

            with self._state_lock:
                cleaned_i16 = self.aec.process(near_i16, far_i16)

            transmitting = not self._muted and (
                self.config.get("ptt", {}).get("mode") != "ptt" or self._ptt_active
            )

            if transmitting:
                payload = self.codec.encode(cleaned_i16)
                if payload is not None:
                    timestamp = int(time.monotonic() * 16000)
                    packet = _rtp_packet(self._sequence, timestamp, self._ssrc, payload)
                    self._sequence += 1

                    with self._queue_lock:
                        self._enqueue_send(packet)
                    self._send_event.set()
            else:
                with self._queue_lock:
                    self._send.clear()

                # "Konuşuyor musunuz?" uyarısı tespiti (VAD veya Genlik Eşiği + 4sn Debounce)
                with self._state_lock:
                    is_speech = self.vad.is_speech(cleaned_i16) if self.config["audio"].get("vad") != "none" else (np.max(np.abs(raw_scaled)) > 0.08)
                if is_speech:
                    talk_warn_enabled = bool(self.config.get("ptt", {}).get("talk_warning", True))
                    if talk_warn_enabled and (now - self._last_talk_warning >= 4.0):
                        self._last_talk_warning = now
                        self.event_bus.emit(
                            "talk_while_muted", {"message": "Mikrofonunuz kapalı, sizi duyamıyorlar!"}
                        )

            cleaned_f = cleaned_i16.astype(np.float32) / 32768.0
            self.recorder.write(raw=raw, reference=far, processed=cleaned_f)

        except Exception as exc:
            if not self._restarting_input:
                self._restarting_input = True
                self.config["audio"]["input_device"] = None
                threading.Thread(target=self._auto_fallback_recover_input, daemon=True).start()

    def _calculate_input_gain(self, raw):
        """Return runtime gain without mutating the persisted user preference."""
        preference = float(self.config["audio"].get("input_gain", 100)) / 100.0
        if preference != self._agc_preference_gain:
            self._agc_preference_gain = preference
            self._agc_gain = preference

        if not self.config["audio"].get("agc", True):
            self._agc_gain = preference
            return preference

        rms = np.sqrt(np.mean(raw ** 2))
        if rms > 0.004:
            target_rms = 0.12
            desired = max(0.2, min(1.5, target_rms / (rms + 1e-6)))
            self._agc_gain = 0.95 * self._agc_gain + 0.05 * desired
        return self._agc_gain

    def _playback_callback(self, outdata, frames, time_info, status):
        if not self._running or self._stop.is_set():
            outdata.fill(0)
            return

        try:
            with self._queue_lock:
                if len(self._play) > 0:
                    frame = self._play.popleft()
                else:
                    frame = np.zeros(frames, dtype=np.float32)

            if self._monitor_mic and self._monitor_frame is not None and len(self._monitor_frame) == frames:
                frame = self._monitor_frame

            level = float(self.config["audio"].get("output_level", 100)) / 100.0
            frame = np.clip(frame * level, -1.0, 1.0)
            outdata[:, 0] = frame

            with self._queue_lock:
                self._far_ref.append(frame.copy())
                while len(self._far_ref) > 10:
                    self._far_ref.popleft()
        except Exception:
            if not self._restarting_output:
                self._restarting_output = True
                self.config["audio"]["output_device"] = None
                threading.Thread(target=self._auto_fallback_recover_output, daemon=True).start()

    def _auto_fallback_recover_input(self):
        try:
            self.event_bus.emit("warning", {"message": "Mikrofon söküldü; Sistem Varsayılanına geçiliyor..."})
            self.restart_input_stream()
        finally:
            self._restarting_input = False

    def _auto_fallback_recover_output(self):
        try:
            self.event_bus.emit("warning", {"message": "Hoparlör/Kulaklık söküldü; Sistem Varsayılanına geçiliyor..."})
            self.restart_output_stream()
        finally:
            self._restarting_output = False

    def _receiver(self):
        frame_size = int(self.config["audio"].get("frame_size", 160))

        while not self._stop.is_set():
            try:
                if not self._socket:
                    time.sleep(0.1)
                    continue
                data, addr = self._socket.recvfrom(65535)
                self._last_rx = time.monotonic()

                my_port = int(self.config["network"].get("port", 5000))
                # Sadece ayni soketten gelen kendi yankisini ele
                if addr[0] in self._local_ips and addr[1] == my_port:
                    continue

                # 1. Ping paketi
                if data == b"ping":
                    self._set_connected(True, addr)
                    continue

                # 2. Keşif İsteği
                if data.startswith(b"SC_INTERCOM_DISCOVER") or data == b"SIMCLEVER_DISCOVERY_PING":
                    parts = data.split(b":")
                    sender_port = int(parts[2]) if len(parts) > 2 else addr[1]
                    reply = f"SC_INTERCOM_NODE:{self.config['role']}:{addr[0]}:{my_port}".encode()
                    try:
                        reply_ip = "127.0.0.1" if addr[0] in self._local_ips else addr[0]
                        self._socket.sendto(reply, (reply_ip, sender_port))
                    except OSError:
                        logging.getLogger(__name__).debug("Discovery reply send failed", exc_info=True)
                    # Gelen keşif paketinden karşı tarafı otomatik öğren ve bağla
                    if not self._remote or self._is_self_endpoint(self._remote[0], self._remote[1]):
                        self._remote = (addr[0], sender_port)
                        self.event_bus.emit(
                            "discovered", {
                                "message": f"Ağda cihaz bulundu: {addr[0]}:{sender_port}",
                                "ip": addr[0],
                                "port": sender_port,
                            }
                        )
                    continue

                # 3. Keşif Yanıtı
                if data.startswith(b"SC_INTERCOM_NODE:") or data == b"SC_INTERCOM_TEACHER":
                    parts = data.split(b":")
                    remote_port = int(parts[3]) if len(parts) > 3 else addr[1]
                    self._remote = (addr[0], remote_port)
                    self.event_bus.emit(
                        "discovered", {
                            "message": f"Ağda cihaz bulundu: {addr[0]}:{remote_port}",
                            "ip": addr[0],
                            "port": remote_port,
                        }
                    )
                    continue

                if not self._running:
                    continue

                seq_num_in, pkt_ssrc, payload = _rtp_payload(data)
                if seq_num_in is None:
                    continue

                if pkt_ssrc == self._ssrc:
                    continue

                # Reorder buffer: duplicate ve stale eleme ile bounded reordering.
                # _accept_rx_sequence yerine _push_to_reorder kullanılır;
                # buffer'dan sıralı drainlama _push_to_reorder içinde yapılır.
                with self._state_lock:
                    next_exp = self._rx_next_expected
                    if next_exp is not None:
                        dist = self._seq16_diff(next_exp, seq_num_in)
                        # Geriye doğru (stale) veya buffer penceresi çok büyük
                        if dist >= 32768:
                            continue
                        # Zaten bekleyen listede var (duplicate)
                        if seq_num_in in self._rx_reorder:
                            continue

                with self._state_lock:
                    pcm_frame = self.codec.decode(payload, frame_size)

                if pcm_frame is None or len(pcm_frame) != frame_size:
                    continue

                pcm_float = pcm_frame.astype(np.float32) / 32768.0

                with self._state_lock:
                    if self.denoiser.method not in ["none", "passthrough"]:
                        proc_float = self.denoiser.process(pcm_float)
                        out_float = np.clip(proc_float, -1.0, 1.0)
                    else:
                        out_float = pcm_float

                # Reorder buffer'a ekle ve sıradaki kareleri _play'e drainla
                self._push_to_reorder(seq_num_in, out_float)

                self._received += 1
                self._set_connected(True, addr)

            except socket.timeout:
                continue
            except (ConnectionResetError, OSError) as exc:
                # 10054 = WSAECONNRESET (Windows UDP ICMP Port Unreachable bildirimi - normal)
                if getattr(exc, "winerror", None) == 10054 or getattr(exc, "errno", None) == 10054:
                    continue
                if not self._stop.is_set():
                    self.event_bus.emit("warning", {"message": "UDP alımı durdu."})
            except Exception as exc:
                self.event_bus.emit(
                    "warning", {"message": "Paket çözülemedi.", "technical_detail": str(exc)}
                )

    def _sender(self):
        while not self._stop.is_set():
            self._send_event.wait(timeout=0.01)
            self._send_event.clear()
            while len(self._send) > 0 and self._running and not self._stop.is_set():
                with self._queue_lock:
                    packet = self._send.popleft() if self._send else None
                if packet and self._socket:
                    try:
                        if self._remote and not self._is_self_endpoint(self._remote[0], self._remote[1]):
                            dest_ip = "127.0.0.1" if (self._remote[0] in self._local_ips) else self._remote[0]
                            self._socket.sendto(packet, (dest_ip, self._remote[1]))
                        else:
                            port_num = int(self.config["network"].get("port", 5000))
                            self._socket.sendto(packet, ("255.255.255.255", port_num))
                        self._sent += 1
                    except OSError as exc:
                        _log = logging.getLogger(__name__)
                        _log.debug("Audio packet send failed: %s", exc, exc_info=True)
                        _now = time.monotonic()
                        if _now - self._last_send_warning >= 5.0:
                            self._last_send_warning = _now
                            self.event_bus.emit(
                                "warning",
                                {
                                    "message": "Ses paketi gonderilemiyor.",
                                    "technical_detail": str(exc),
                                },
                            )

    def _heartbeat(self):
        while not self._stop.wait(2.0):
            if self._socket and self._remote and not self._is_self_endpoint(self._remote[0], self._remote[1]):
                try:
                    dest_ip = "127.0.0.1" if (self._remote[0] in self._local_ips) else self._remote[0]
                    self._socket.sendto(b"ping", (dest_ip, self._remote[1]))
                except OSError:
                    logging.getLogger(__name__).debug("Heartbeat ping failed", exc_info=True)

    def _monitor(self):
        while not self._stop.wait(1.0):
            if self._connected and time.monotonic() - self._last_rx > 5:
                self._set_connected(False, None)
            self.event_bus.emit(
                "metrics",
                {
                    "sent": self._sent,
                    "received": self._received,
                    "connected": self._connected,
                    "remote": self._remote,
                },
            )

    def _bg_discovery(self):
        last_ips = set(self._local_ips)
        while not self._stop.wait(3.0):
            try:
                curr_ips = get_local_ips()
                if set(curr_ips) != set(last_ips):
                    last_ips = set(curr_ips)
                    self._local_ips = frozenset(curr_ips)
                    self.event_bus.emit("local_ips_changed", {"ips": curr_ips})

                if self._socket:
                    curr_port = int(self.config["network"].get("port", 5000))
                    probe_ports = {4999, 5000, 5001, curr_port}
                    if self._remote and self._remote[1]:
                        probe_ports.add(int(self._remote[1]))
                    targets = get_all_discovery_targets(curr_ips)
                    msg = f"SC_INTERCOM_DISCOVER:{self.config['role']}:{curr_port}".encode()
                    for target in targets:
                        for p in probe_ports:
                            if target in curr_ips and p == curr_port:
                                continue
                            try:
                                self._socket.sendto(msg, (target, p))
                            except OSError:
                                pass
            except (OSError, ValueError, TypeError):
                logging.getLogger(__name__).debug("Background discovery failed", exc_info=True)

    def _set_connected(self, value, addr):
        if value != self._connected:
            self._connected = value
            if not value:
                self.reset_rx_sequence()
            self.event_bus.emit(
                "connected" if value else "disconnected",
                {
                    "message": "Bağlantı kuruldu." if value else "Bağlantı koptu.",
                    "remote": addr,
                },
            )

    def connect(self, target_ip=None, port=None):
        self.reset_rx_sequence()
        my_port = int(self.config["network"].get("port", 5000))
        target = target_ip.strip() if target_ip else self.config["network"].get("target_ip", "").strip()
        port_num = int(port or self.config["network"].get("port", 5000))

        if not target or self._is_self_endpoint(target, port_num):
            self._remote = None
            if target and self._is_self_endpoint(target, port_num):
                self.event_bus.emit(
                    "warning",
                    {
                        "message": f"Kendi soket adresinize ({target}:{port_num}) bağlanamazsınız. Lütfen farklı bir port veya cihaz seçin."
                    },
                )
            return

        self.config["network"]["target_ip"] = target
        self._remote = (target, port_num)
        self._manual_disconnect = False
        if not self._socket:
            try:
                self._open_socket()
            except OSError as exc:
                self.event_bus.emit(
                    "warning",
                    {"message": "Ağ soketi açılamadı.", "technical_detail": str(exc)},
                )
        self.event_bus.emit("connecting", {"message": "Karşı taraf bekleniyor..."})

    def disconnect(self, manual=True):
        self._manual_disconnect = manual
        self._remote = None
        self._set_connected(False, None)

    def reconnect(self):
        self.connect(
            self.config["network"].get("target_ip"), self.config["network"].get("port", 5000)
        )

    def discover(self):
        if not self._socket:
            try:
                self._open_socket()
            except OSError as exc:
                self.event_bus.emit(
                    "warning",
                    {"message": "Ağ soketi açılamadı.", "technical_detail": str(exc)},
                )
        curr_port = int(self.config["network"].get("port", 5000))
        probe_ports = set(range(4995, 5016)) | {curr_port}
        curr_ips = get_local_ips()
        targets = get_all_discovery_targets(curr_ips)
        msg = b"SC_INTERCOM_DISCOVER"
        for target in targets:
            for p in probe_ports:
                if target in curr_ips and p == curr_port:
                    continue
                try:
                    if self._socket:
                        self._socket.sendto(msg, (target, p))
                except OSError:
                    logging.getLogger(__name__).debug(
                        "Manual discovery packet send failed: %s:%d", target, p, exc_info=True
                    )
        self.event_bus.emit("connecting", {"message": "Yerel ağda cihazlar aranıyor..."})

    def start_recording(self):
        if self.config["role"] != "teacher":
            self.event_bus.emit(
                "warning", {"message": "Kayıt yalnızca hoca rolünde kullanılabilir."}
            )
            return
        self.recorder.enabled = True
        self.recorder.start()

    def stop_recording(self):
        self.recorder.stop()

    def set_muted(self, value):
        self._muted = bool(value)
        self.event_bus.emit("mute_changed", {"muted": self._muted})

    def set_ptt(self, value):
        self._ptt_active = bool(value)
        self.event_bus.emit("ptt_changed", {"active": self._ptt_active})

    def set_agc(self, value):
        self.config.setdefault("audio", {})["agc"] = bool(value)
        self.event_bus.emit("config_changed", {"message": f"AGC {'etkinleştirildi' if value else 'devre dışı'}"})

    def set_monitor_mic(self, value):
        self._monitor_mic = bool(value)
        self.event_bus.emit("monitor_changed", {"active": self._monitor_mic})

    def set_prevent_sleep(self, enabled):
        if sys.platform == "win32":
            try:
                import ctypes
                flags = 0x80000000 | (0x00000003 if enabled else 0)
                ctypes.windll.kernel32.SetThreadExecutionState(flags)
            except (AttributeError, OSError):
                logging.getLogger(__name__).debug(
                    "Prevent-sleep request failed", exc_info=True
                )
        self.config.setdefault("advanced", {})["prevent_sleep"] = bool(enabled)

    def _close_audio(self):
        for stream in (self._input_stream, self._output_stream):
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    logging.getLogger(__name__).debug("Audio stream close failed", exc_info=True)
        self._input_stream = self._output_stream = None

    def restart_input_stream(self):
        """Sadece mikrofon akışını milisaniyeler içinde yeniler; yayın veya hoparlör etkilenmez."""
        with self._state_lock:
            if self._input_stream:
                try:
                    self._input_stream.stop()
                    self._input_stream.close()
                except Exception:
                    logging.getLogger(__name__).debug("Input stream close failed", exc_info=True)
                self._input_stream = None
            if self._running:
                self._open_input_audio()
        self.event_bus.emit("device_changed", {"message": "Mikrofon akışı yenilendi."})

    def restart_output_stream(self):
        """Sadece hoparlör/kulaklık akışını milisaniyeler içinde yeniler; yayın veya mikrofon etkilenmez."""
        with self._state_lock:
            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:
                    logging.getLogger(__name__).debug("Output stream close failed", exc_info=True)
                self._output_stream = None
            if self._running:
                self._open_output_audio()
        self.event_bus.emit("device_changed", {"message": "Hoparlör/Kulaklık akışı yenilendi."})

    def restart_audio(self):
        """Ses akışlarını güvenle kapatıp PortAudio cihaz önbelleğini temizleyerek yeniden açar."""
        with self._state_lock:
            self._close_audio()
            try:
                import sounddevice as sd
                sd._terminate()
                sd._initialize()
            except Exception:
                logging.getLogger(__name__).debug("PortAudio reset failed", exc_info=True)
            if self._running:
                self._open_audio()
        self.event_bus.emit("device_changed", {"message": "Ses cihazları yenilendi."})

    def _close_resources(self):
        self._close_audio()
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                logging.getLogger(__name__).debug("Socket close on cleanup failed", exc_info=True)
        self._socket = None

    def get_status(self):
        return {
            "running": self._running,
            "connected": self._connected,
            "muted": self._muted,
            "ptt_active": self._ptt_active,
            "sent": self._sent,
            "received": self._received,
            "remote": self._remote,
            "role": self.config["role"],
        }

    def list_audio_devices(self):
        """Windows WASAPI COM arabirimi ve sounddevice ile 0 ms'de canlı ses cihazı isimlerini listeler."""
        result = {"inputs": [(None, "Sistem Varsayılanı")], "outputs": [(None, "Sistem Varsayılanı")]}
        try:
            # 1. Windows WASAPI COM uç noktalarını çek (0 ms canlı sorgu)
            wasapi_mics = query_wasapi_endpoints(1) # Capture (Mikrofonlar)
            wasapi_spks = query_wasapi_endpoints(0) # Render (Hoparlörler/Kulaklıklar)

            seen_in  = set()
            seen_out = set()

            for name in wasapi_mics:
                clean = name.strip()
                if clean and clean not in seen_in:
                    seen_in.add(clean)
                    result["inputs"].append((clean, clean))

            for name in wasapi_spks:
                clean = name.strip()
                if clean and clean not in seen_out:
                    seen_out.add(clean)
                    result["outputs"].append((clean, clean))

            # 2. Yedek sounddevice sorgusu (Kanal sayılarına göre tam filtrele!)
            import sounddevice as sd
            for d in sd.query_devices():
                name = d["name"].strip()
                if "Ses Eşleştiricisi" in name or "Birincil Ses" in name:
                    continue
                if d["max_input_channels"] > 0 and name not in seen_in:
                    seen_in.add(name)
                    result["inputs"].append((name, name))
                if d["max_output_channels"] > 0 and name not in seen_out:
                    seen_out.add(name)
                    result["outputs"].append((name, name))

        except Exception as exc:
            self.event_bus.emit(
                "warning",
                {"message": "Ses cihazları okunamadı.", "technical_detail": str(exc)},
            )
        return result
