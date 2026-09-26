"""
paket5/ui_main.py – SimClever Intercom Paket 5 Ana Pencere
Google Meet ve Windows WASAPI stili anlık ses cihazı takılma/çıkarılma algılaması, otomatik cihaz seçimi, AGC ve canlı kontroller.
"""
import ctypes
import datetime
import logging
import pathlib
import sys
import time

from PyQt5 import QtCore, QtGui, QtWidgets

from shared.intercom_engine import IntercomEngine, get_local_ips, is_self_ip, is_self_endpoint, query_wasapi_endpoints
from paket5.ui_settings import SettingsDialog


# ─────────────────────────── Stil ────────────────────────────────────────────
STYLE = """
/* Genel arka plan */
QMainWindow, QWidget          { background:#12121e; color:#dce1f0; font:10pt 'Segoe UI'; }
QScrollArea                   { border:none; background:#12121e; }

/* Kartlar */
QGroupBox {
    background:#1a1a2e;
    border:1px solid #2e2e4a;
    border-radius:8px;
    margin-top:14px;
    padding:14px 12px 12px;
    font-weight:700;
    color:#f9e2af;
}
QGroupBox::title {
    subcontrol-origin:margin;
    left:12px;
    padding:0 6px;
    font-size:10pt;
}

/* Giriş bileşenleri */
QLineEdit, QComboBox, QSpinBox {
    background:#0d0d1a;
    border:1px solid #3a3a5c;
    border-radius:5px;
    padding:6px 8px;
    color:#f0f0f8;
    min-height:24px;
}
QComboBox::drop-down    { border:none; }
QComboBox::down-arrow   { image:none; width:12px; }
QLineEdit:focus, QComboBox:focus { border-color:#89b4fa; }

/* Slider Stili */
QSlider::groove:horizontal { background:#1a1a2e; height:6px; border-radius:3px; border:1px solid #2e2e4a; }
QSlider::handle:horizontal { background:#89b4fa; width:14px; height:14px; border-radius:7px; margin:-4px 0; }
QSlider::handle:horizontal:hover { background:#b4befe; }

/* CheckBox Stili */
QCheckBox { color:#dce1f0; font-weight:600; }
QCheckBox::indicator { width:16px; height:16px; border-radius:3px; border:1px solid #3a3a5c; background:#0d0d1a; }
QCheckBox::indicator:checked { background:#89b4fa; border-color:#89b4fa; }

/* Butonlar */
QPushButton {
    background:#252540;
    border:1px solid #3a3a5c;
    border-radius:6px;
    padding:8px 14px;
    font-weight:600;
    color:#dce1f0;
    min-height:28px;
}
QPushButton:hover   { background:#30304e; border-color:#585b70; }
QPushButton:pressed { background:#1e1e36; }
QPushButton:disabled{ color:#4a4a6a; background:#1a1a2e; border-color:#2a2a44; }
QPushButton:checked { background:#45475a; border-color:#89b4fa; color:#89b4fa; }

/* Özel butonlar */
QPushButton#startBtn  { background:#1e4a2e; border-color:#a6e3a1; color:#a6e3a1; font-size:12pt; padding:10px 20px; }
QPushButton#startBtn:hover  { background:#2a6040; }
QPushButton#stopBtn   { background:#4a1e2e; border-color:#f38ba8; color:#f38ba8; font-size:12pt; padding:10px 20px; }
QPushButton#stopBtn:hover   { background:#602a3a; }
QPushButton#muteBtn   { background:#3a2a0a; border-color:#f9e2af; color:#f9e2af; font-size:11pt; }
QPushButton#muteBtn:checked { background:#4a1e2e; border-color:#f38ba8; color:#f38ba8; }
QPushButton#pttBtn    { background:#1e2e4a; border-color:#89b4fa; color:#89b4fa; font-size:11pt; }
QPushButton#pttBtn:checked  { background:#2a4a6a; }
QPushButton#accentBtn { background:#1e304a; border-color:#89b4fa; color:#89b4fa; }
QPushButton#accentBtn:hover { background:#264060; }
QPushButton#warnBtn   { background:#3a2000; border-color:#f9a825; color:#f9a825; }

/* Etiketler */
QLabel#titleLbl  { color:#89b4fa; font-size:16pt; font-weight:700; }
QLabel#roleLbl   { font-size:11pt; font-weight:700; border-radius:5px; padding:4px 10px; }
QLabel#connLbl   { font-size:9pt; padding:3px 8px; border-radius:4px; }
QLabel#statusLbl { font-size:20pt; font-weight:700; letter-spacing:2px; }
QLabel#warningBanner {
    background:#3a2000;
    border:1px solid #f9a825;
    border-radius:6px;
    color:#f9e2af;
    padding:8px 14px;
    font-weight:600;
}

/* İlerleme çubuğu */
QProgressBar {
    background:#0d0d1a;
    border:1px solid #3a3a5c;
    border-radius:4px;
    text-align:center;
    color:#dce1f0;
    max-height:18px;
}
QProgressBar::chunk { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1e8449,stop:0.7 #a6e3a1,stop:1 #f38ba8); border-radius:4px; }

/* Log alanı */
QPlainTextEdit {
    background:#08080f;
    color:#a6e3a1;
    border:1px solid #2a2a44;
    border-radius:6px;
    font:9pt Consolas;
    padding:4px;
}

/* Çizgi */
QFrame[frameShape="4"], QFrame[frameShape="5"] { color:#2e2e4a; }

/* Scroll çubukları */
QScrollBar:vertical { background:#1a1a2e; width:8px; }
QScrollBar::handle:vertical { background:#3a3a5c; border-radius:4px; min-height:30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
"""

# Durum renkleri
CONN_COLORS = {
    "Hazır":             ("#4a4a6a", "#dce1f0"),
    "Bağlanıyor":        ("#3a3000", "#f9e2af"),
    "Bağlı":             ("#1e4a2e", "#a6e3a1"),
    "Koptu":             ("#4a1e1e", "#f38ba8"),
    "Yeniden bağlanıyor":("#3a3000", "#f9e2af"),
    "Hata":              ("#4a1e1e", "#f38ba8"),
}

DENOISE_HELP = {
    "Kapalı": "Ses işlenmeden iletilir.",
    "Düşük":  "Hafif ortam gürültüleri için — önerilen günlük kullanım.",
    "Orta":   "Gürültülü sınıf ortamları için.",
    "Yüksek": "Yoğun gürültü için; kısık sesler zayıflayabilir.",
}
DENOISE_MAP  = {"Kapalı": "none", "Düşük": "webrtc_mild", "Orta": "webrtc_aggressive", "Yüksek": "webrtc_max"}
DENOISE_RMAP = {v: k for k, v in DENOISE_MAP.items()}


# ─────────────────────── Event Bridge ────────────────────────────────────────
class EventBridge(QtCore.QObject):
    event = QtCore.pyqtSignal(str, dict)


# ─────────────────────── Ana Pencere ─────────────────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SimClever Intercom — Paket 5")
        self.resize(1140, 740)
        self.setMinimumSize(940, 640)
        self.setStyleSheet(STYLE)

        self.engine = IntercomEngine(config_path)
        self.bridge = EventBridge()
        self.bridge.event.connect(self._handle_event)
        self.engine.event_bus.on("*", lambda name, payload: self.bridge.event.emit(name, payload))

        self._previous_wasapi_inputs  = set(query_wasapi_endpoints(1)) # Capture
        self._previous_wasapi_outputs = set(query_wasapi_endpoints(0)) # Render
        self._last_error_time   = 0.0
        self._last_error_msg    = ""
        self._banner_timer      = QtCore.QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self._hide_banner)

        self._build_ui()
        self._load_config_into_ui()
        self.refresh_devices(auto_switch=False)
        self.engine.set_prevent_sleep(
            bool(self.engine.config.get("advanced", {}).get("prevent_sleep", False))
        )
        self._register_global_hotkey()

        # Windows WASAPI stili anlık donanım kontrolü (200 ms periyot - Milisaniyelik anlık tepki!)
        self._device_timer = QtCore.QTimer(self)
        self._device_timer.timeout.connect(self._poll_devices)
        self._device_timer.start(200)

        self._append_log("Sistem hazır. Mikrofon/hoparlör/kulaklık takıldığında anında seçilir, çıkarıldığında varsayılana düşer.")

    # ──────────────────────── Windows Donanım Olayları ────────────────────────
    def nativeEvent(self, eventType, message):
        """Windows WM_DEVICECHANGE (0x0219) donanım takılma/çıkarılma olaylarını 0 ms'de yakalar."""
        retval, result = super().nativeEvent(eventType, message)
        if eventType == b"windows_generic_MSG" and message:
            try:
                import ctypes.wintypes
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == 0x0219:  # WM_DEVICECHANGE
                    QtCore.QTimer.singleShot(50, lambda: self.refresh_devices(auto_switch=True))
            except (AttributeError, TypeError, ValueError, OSError):
                logging.getLogger(__name__).debug(
                    "WM_DEVICECHANGE handling failed", exc_info=True
                )
        return retval, result

    # ──────────────────────── UI İnşaası ─────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        root.addLayout(self._build_header())
        root.addWidget(self._build_banner())

        # İki sütunlu ana alan
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left_scroll  = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_w = QtWidgets.QWidget()
        self._left = QtWidgets.QVBoxLayout(left_w)
        self._left.setSpacing(8)
        left_scroll.setWidget(left_w)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_w = QtWidgets.QWidget()
        self._right = QtWidgets.QVBoxLayout(right_w)
        self._right.setSpacing(8)
        right_scroll.setWidget(right_w)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_scroll)
        splitter.setSizes([600, 500])
        root.addWidget(splitter, 1)

        # Sol sütun kartları
        self._build_connection_card()
        self._build_devices_card()
        self._build_denoise_card()
        self._build_ptt_card()
        self._left.addStretch()

        # Sağ sütun kartları
        self._build_status_card()
        self._build_recording_card()
        self._build_log_card()
        self._right.addStretch()

    def _build_header(self):
        row = QtWidgets.QHBoxLayout()

        # Başlık
        title = QtWidgets.QLabel("SimClever Intercom")
        title.setObjectName("titleLbl")
        row.addWidget(title)

        # Rol rozeti
        self.role_badge = QtWidgets.QLabel("HOCA")
        self.role_badge.setObjectName("roleLbl")
        self.role_badge.setStyleSheet("background:#1e304a; color:#89b4fa; border-radius:5px; padding:4px 10px;")
        row.addWidget(self.role_badge)

        # Rol değiştir combo
        self.role_combo = QtWidgets.QComboBox()
        self.role_combo.addItems(["Hoca Paneli", "Öğrenci Düğümü"])
        self.role_combo.setFixedWidth(150)
        self.role_combo.currentTextChanged.connect(self._role_changed)
        row.addWidget(self.role_combo)

        # Yerel IP Etiketi
        self.local_ip_label = QtWidgets.QLabel("Yerel IP: Algılanıyor...")
        self.local_ip_label.setStyleSheet("color:#a6e3a1; font-size:9pt; font-weight:600; padding:2px 8px; background:#181826; border-radius:4px; border:1px solid #2e2e4a;")
        row.addWidget(self.local_ip_label)

        row.addStretch()

        # Bağlantı durumu
        self.conn_badge = QtWidgets.QLabel("Hazır")
        self.conn_badge.setObjectName("connLbl")
        self._set_conn_badge("Hazır")
        row.addWidget(self.conn_badge)

        # Ayarlar düğmesi
        settings = QtWidgets.QPushButton("⚙  Ayarlar")
        settings.setObjectName("accentBtn")
        settings.clicked.connect(self.open_settings)
        row.addWidget(settings)

        return row

    def _build_banner(self):
        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("warningBanner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        return self.banner

    # ── Hızlı Bağlantı Kartı ─────────────────────────────────────────────────
    def _build_connection_card(self):
        inner = QtWidgets.QGroupBox("Bağlantı ve Ağ")
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(6)
        self._left.addWidget(inner)

        form = QtWidgets.QFormLayout()
        
        # Profil ComboBox
        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        form.addRow("Kayıtlı Profil:", self.profile_combo)

        # Hedef IP ComboBox
        self.ip_combo = QtWidgets.QComboBox()
        self.ip_combo.setEditable(True)
        self.ip_combo.setPlaceholderText("Örn: 192.168.1.45")
        self.ip_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        form.addRow("Hedef IP (Ağ/Cihaz):", self.ip_combo)

        # UDP Port SpinBox
        self.port_spin = QtWidgets.QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5000)
        form.addRow("UDP Port:", self.port_spin)

        il.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.discover_btn    = QtWidgets.QPushButton("🔍 Otomatik Bul")
        self.connect_btn     = QtWidgets.QPushButton("🔗 Bağlan")
        self.connect_btn.setObjectName("accentBtn")
        self.disconnect_btn  = QtWidgets.QPushButton("✖ Kes")
        self.reconnect_btn   = QtWidgets.QPushButton("↻ Yeniden")
        for b in (self.discover_btn, self.connect_btn, self.disconnect_btn, self.reconnect_btn):
            btns.addWidget(b)
        il.addLayout(btns)

        tip = QtWidgets.QLabel(
            '<a style="color:#89b4fa;" href="settings">Detaylı IP/Port ve profil yönetimi için Ayarlar &gt; Bağlantı</a>'
        )
        tip.setOpenExternalLinks(False)
        tip.linkActivated.connect(lambda _: self.open_settings(tab=1))
        il.addWidget(tip)

        self.discover_btn.clicked.connect(self._on_discover_clicked)
        self.connect_btn.clicked.connect(self._connect_from_ui)
        self.disconnect_btn.clicked.connect(lambda: self.engine.disconnect(True))
        self.reconnect_btn.clicked.connect(self.engine.reconnect)
        self.profile_combo.currentIndexChanged.connect(self._load_profile)
        self.ip_combo.currentTextChanged.connect(self._on_ip_combo_changed)
        self.port_spin.valueChanged.connect(self._on_port_changed)

    # ── Ses Cihazları Kartı ───────────────────────────────────────────────────
    def _build_devices_card(self):
        inner = QtWidgets.QGroupBox("Ses Cihazları ve Kazanç")
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(6)
        self._left.addWidget(inner)

        form = QtWidgets.QFormLayout()
        self.input_combo  = QtWidgets.QComboBox()
        self.output_combo = QtWidgets.QComboBox()
        form.addRow("Mikrofon / Ses Girişi:", self.input_combo)
        form.addRow("Hoparlör / Kulaklık:", self.output_combo)

        # Otomatik Kazanç Kontrolü (AGC)
        self.agc_checkbox = QtWidgets.QCheckBox("⚡ Otomatik Kazanç (AGC)")
        self.agc_checkbox.setChecked(True)
        form.addRow(self.agc_checkbox)

        # Giriş kazancı slider
        self.input_gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.input_gain_slider.setRange(0, 150)
        self.input_gain_slider.setValue(100)
        self.input_gain_val_lbl = QtWidgets.QLabel("100%")
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.input_gain_slider)
        gain_row.addWidget(self.input_gain_val_lbl)
        form.addRow("Giriş Kazancı (%):", gain_row)

        # Çıkış seviyesi slider
        self.output_level_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.output_level_slider.setRange(0, 100)
        self.output_level_slider.setValue(100)
        self.output_level_val_lbl = QtWidgets.QLabel("100%")
        vol_row = QtWidgets.QHBoxLayout()
        vol_row.addWidget(self.output_level_slider)
        vol_row.addWidget(self.output_level_val_lbl)
        form.addRow("Çıkış Seviyesi (%):", vol_row)

        il.addLayout(form)

        # Seviye çubuğu
        self.mic_meter = QtWidgets.QProgressBar()
        self.mic_meter.setRange(0, 100)
        self.mic_meter.setTextVisible(False)
        self.mic_meter.setFixedHeight(14)
        il.addWidget(self.mic_meter)

        btns = QtWidgets.QHBoxLayout()
        self.refresh_dev_btn = QtWidgets.QPushButton("↻ Yenile")
        self.mic_test_btn    = QtWidgets.QPushButton("🎙 Mikrofon Testi")
        self.mic_test_btn.setCheckable(True)
        btns.addWidget(self.refresh_dev_btn)
        btns.addWidget(self.mic_test_btn)
        btns.addStretch()
        il.addLayout(btns)

        self.refresh_dev_btn.clicked.connect(lambda: self.refresh_devices(auto_switch=False))
        self.mic_test_btn.toggled.connect(self._mic_test_changed)
        self.input_combo.currentIndexChanged.connect(self._input_device_changed)
        self.output_combo.currentIndexChanged.connect(self._output_device_changed)
        self.agc_checkbox.toggled.connect(self._on_agc_toggled)
        self.input_gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self.output_level_slider.valueChanged.connect(self._on_vol_slider_changed)

    # ── Gürültü Kontrol Kartı ─────────────────────────────────────────────────
    def _build_denoise_card(self):
        inner = QtWidgets.QGroupBox("Gürültü Kontrolü")
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(8)
        self._left.addWidget(inner)

        self.denoise_combo = QtWidgets.QComboBox()
        self.denoise_combo.addItems(["Kapalı", "Düşük", "Orta", "Yüksek"])
        il.addWidget(self.denoise_combo)

        self.denoise_help_lbl = QtWidgets.QLabel()
        self.denoise_help_lbl.setWordWrap(True)
        self.denoise_help_lbl.setStyleSheet("color:#a0a8c0; font-size:9pt;")
        il.addWidget(self.denoise_help_lbl)

        tip = QtWidgets.QLabel(
            '<a style="color:#89b4fa;" href="settings">Codec, AEC, VAD için Ayarlar &gt; Ses İşleme</a>'
        )
        tip.linkActivated.connect(lambda _: self.open_settings(tab=4))
        il.addWidget(tip)

        self.denoise_combo.currentTextChanged.connect(self._denoise_changed)

    # ── PTT / Yayın Kartı ─────────────────────────────────────────────────────
    def _build_ptt_card(self):
        inner = QtWidgets.QGroupBox("Yayın ve PTT")
        il = QtWidgets.QVBoxLayout(inner)
        il.setSpacing(8)
        self._left.addWidget(inner)

        # Mod seçimi
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mod:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Sürekli Açık", "Bas-Konuş"])
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        self.shortcut_info = QtWidgets.QLabel("")
        self.shortcut_info.setStyleSheet("color:#a0a8c0; font-size:9pt;")
        mode_row.addWidget(self.shortcut_info)
        il.addLayout(mode_row)

        # Büyük durum etiketi
        self.broadcast_label = QtWidgets.QLabel("MİKROFON KAPALI")
        self.broadcast_label.setObjectName("statusLbl")
        self.broadcast_label.setAlignment(QtCore.Qt.AlignCenter)
        self.broadcast_label.setStyleSheet("color:#f38ba8; font-size:18pt; font-weight:700; padding:6px; letter-spacing:1px;")
        il.addWidget(self.broadcast_label)

        # Başlat / Durdur
        action_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶  YAYINI BAŞLAT")
        self.start_btn.setObjectName("startBtn")
        self.stop_btn  = QtWidgets.QPushButton("■  YAYINI DURDUR")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        action_row.addWidget(self.start_btn, 2)
        action_row.addWidget(self.stop_btn, 1)
        il.addLayout(action_row)

        # Sessize al + PTT
        ctrl_row = QtWidgets.QHBoxLayout()
        self.mute_btn = QtWidgets.QPushButton("🔇 Sessize Al")
        self.mute_btn.setObjectName("muteBtn")
        self.mute_btn.setCheckable(True)
        self.ptt_btn  = QtWidgets.QPushButton("🎙 BAS-KONUŞ (BASILI TUT)")
        self.ptt_btn.setObjectName("pttBtn")
        self.ptt_btn.setEnabled(False)
        ctrl_row.addWidget(self.mute_btn)
        ctrl_row.addWidget(self.ptt_btn)
        il.addLayout(ctrl_row)

        self.start_btn.clicked.connect(self.start_engine)
        self.stop_btn.clicked.connect(self.stop_engine)
        self.mute_btn.toggled.connect(self._mute_changed)
        self.ptt_btn.pressed.connect(lambda: self.engine.set_ptt(True))
        self.ptt_btn.released.connect(lambda: self.engine.set_ptt(False))
        self.mode_combo.currentTextChanged.connect(self._ptt_mode_changed)

    # ── Canlı Durum Kartı (Sağ) ───────────────────────────────────────────────
    def _build_status_card(self):
        inner = QtWidgets.QGroupBox("Canlı Durum")
        il = QtWidgets.QVBoxLayout(inner)
        self._right.addWidget(inner)

        form = QtWidgets.QFormLayout()
        self.status_conn     = QtWidgets.QLabel("Hazır")
        self.status_denoise  = QtWidgets.QLabel("Düşük / webrtc_mild")
        self.status_codec    = QtWidgets.QLabel("PCM")
        self.status_delay    = QtWidgets.QLabel("—")
        self.status_record   = QtWidgets.QLabel("Kapalı")

        form.addRow("Bağlantı:",         self.status_conn)
        form.addRow("Aktif filtre:",     self.status_denoise)
        form.addRow("Codec:",            self.status_codec)
        form.addRow("Gecikme:",          self.status_delay)
        form.addRow("Kayıt:",            self.status_record)
        il.addLayout(form)

    # ── Kayıt Kartı (Sağ – yalnızca hoca) ────────────────────────────────────
    def _build_recording_card(self):
        self.record_group = QtWidgets.QGroupBox("Kayıt — Yalnızca Hoca")
        il = QtWidgets.QVBoxLayout(self.record_group)
        il.setSpacing(6)
        self._right.addWidget(self.record_group)

        form = QtWidgets.QFormLayout()
        self.record_status  = QtWidgets.QLabel("Durduruldu")
        self.record_timer   = QtWidgets.QLabel("00:00:00")
        self.record_file    = QtWidgets.QLabel("—")
        self.record_file.setWordWrap(True)
        form.addRow("Durum:",       self.record_status)
        form.addRow("Süre:",        self.record_timer)
        form.addRow("Dosya:",       self.record_file)
        il.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.record_start_btn = QtWidgets.QPushButton("⏺  Kaydı Başlat")
        self.record_start_btn.setObjectName("startBtn")
        self.record_stop_btn  = QtWidgets.QPushButton("⏹  Durdur")
        self.record_stop_btn.setObjectName("stopBtn")
        self.record_stop_btn.setEnabled(False)
        self.open_record_btn  = QtWidgets.QPushButton("📂 Klasörü Aç")
        btns.addWidget(self.record_start_btn)
        btns.addWidget(self.record_stop_btn)
        btns.addWidget(self.open_record_btn)
        il.addLayout(btns)

        self.record_start_btn.clicked.connect(self._start_recording)
        self.record_stop_btn.clicked.connect(self._stop_recording)
        self.open_record_btn.clicked.connect(self._open_recordings)

        # Kayıt kronometresi
        self._record_elapsed = 0
        self._rec_clock = QtCore.QTimer(self)
        self._rec_clock.setInterval(1000)
        self._rec_clock.timeout.connect(self._tick_record_timer)

    # ── Log Kartı (Sağ) ───────────────────────────────────────────────────────
    def _build_log_card(self):
        inner = QtWidgets.QGroupBox("Sistem Günlüğü")
        il = QtWidgets.QVBoxLayout(inner)
        self._right.addWidget(inner)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(160)
        self.log_view.setMaximumBlockCount(400)
        il.addWidget(self.log_view)

        row = QtWidgets.QHBoxLayout()
        clear_btn    = QtWidgets.QPushButton("Temizle")
        open_log_btn = QtWidgets.QPushButton("📂 Log Klasörü")
        clear_btn.clicked.connect(self.log_view.clear)
        open_log_btn.clicked.connect(self._open_logs)
        row.addWidget(clear_btn)
        row.addWidget(open_log_btn)
        row.addStretch()
        il.addLayout(row)

    # ──────────────────────── Config → UI ────────────────────────────────────

    def _load_config_into_ui(self):
        cfg   = self.engine.config
        audio = cfg.get("audio", {})
        net   = cfg.get("network", {})
        ptt   = cfg.get("ptt", {})

        # Local IPs
        local_ips = get_local_ips()
        self.local_ip_label.setText(f"Yerel IP: {', '.join(local_ips)}")

        # Rol
        role = cfg.get("role", "teacher")
        self.role_combo.blockSignals(True)
        self.role_combo.setCurrentText("Hoca Paneli" if role == "teacher" else "Öğrenci Düğümü")
        self.role_combo.blockSignals(False)
        self._apply_role(role)

        # Profiller & Target IP Combobox
        self.profile_combo.clear()
        self.ip_combo.blockSignals(True)
        self.ip_combo.clear()

        my_port = int(net.get("port", 5000))
        profiles = cfg.get("profiles", [])
        if not profiles:
            self.profile_combo.addItem("Manuel bağlantı", None)
        for p in profiles:
            self.profile_combo.addItem(p.get("name", "Profil"), p)
            pip = p.get("ip", "").strip()
            pport = int(p.get("port", 5000))
            if pip and not is_self_endpoint(pip, pport, my_port):
                if self.ip_combo.findText(pip) < 0:
                    self.ip_combo.addItem(pip)

        target_ip = net.get("target_ip", "").strip()
        port_num = int(net.get("port", 5000))
        if target_ip and not is_self_endpoint(target_ip, port_num, my_port):
            if self.ip_combo.findText(target_ip) < 0:
                self.ip_combo.addItem(target_ip)
            self.ip_combo.setCurrentText(target_ip)
        self.ip_combo.blockSignals(False)

        # UDP Port
        port_num = int(net.get("port", 5000))
        self.port_spin.setValue(port_num)

        # AGC & Kazançlar
        self.agc_checkbox.setChecked(bool(audio.get("agc", True)))
        gain_val = int(audio.get("input_gain", 100))
        vol_val  = int(audio.get("output_level", 100))
        self.input_gain_slider.setValue(gain_val)
        self.input_gain_val_lbl.setText(f"{gain_val}%")
        self.output_level_slider.setValue(vol_val)
        self.output_level_val_lbl.setText(f"{vol_val}%")

        # Denoise
        label = DENOISE_RMAP.get(audio.get("denoise", "webrtc_mild"), "Düşük")
        self.denoise_combo.blockSignals(True)
        self.denoise_combo.setCurrentText(label)
        self.denoise_combo.blockSignals(False)
        self._denoise_changed(label)

        # PTT modu
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(
            "Bas-Konuş" if ptt.get("mode") == "ptt" else "Sürekli Açık"
        )
        self.mode_combo.blockSignals(False)
        self._ptt_mode_changed(self.mode_combo.currentText())

        # Kısayol bilgisi
        shortcut = ptt.get("shortcut", "Space")
        self.shortcut_info.setText(f"Kısayol: [{shortcut}]")

        # Codec durumu
        codec_val = audio.get("codec", "pcm")
        self.status_codec.setText("Opus" if "opus" in codec_val else "PCM")

    def refresh_devices(self, auto_switch: bool = True):
        """Google Meet ve Windows WASAPI stili anlık ses cihazı güncellemesi, eşzamanlı takılma ve varsayılana düşüş."""
        devices = self.engine.list_audio_devices()

        # Mevcut seçili cihaz adları
        prev_in_name  = self.input_combo.currentText()
        prev_out_name = self.output_combo.currentText()

        # Canlı WASAPI cihaz kümelerinden yeni takılan ve sökülenleri bul
        cur_wasapi_in  = set(query_wasapi_endpoints(1)) # Capture (Mikrofonlar)
        cur_wasapi_out = set(query_wasapi_endpoints(0)) # Render (Hoparlörler/Kulaklıklar)

        new_in_names      = cur_wasapi_in - self._previous_wasapi_inputs
        new_out_names     = cur_wasapi_out - self._previous_wasapi_outputs
        removed_in_names  = self._previous_wasapi_inputs - cur_wasapi_in
        removed_out_names = self._previous_wasapi_outputs - cur_wasapi_out

        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)
        self.input_combo.clear()
        self.output_combo.clear()

        for val, label in devices["inputs"]:
            self.input_combo.addItem(label, val)
        for val, label in devices["outputs"]:
            self.output_combo.addItem(label, val)

        selected_in_name  = prev_in_name
        selected_out_name = prev_out_name

        banner_msgs = []
        restart_mic = False
        restart_spk = False

        # 1. Yeni MİKROFON takıldıysa YALNIZCA MİKROFON KISMINA ATAMA YAP!
        if auto_switch and new_in_names:
            new_in_name = list(new_in_names)[0]
            selected_in_name = new_in_name
            banner_msgs.append(f"🎧 Yeni Mikrofon Algılandı ve Seçildi: {new_in_name}")
            self._append_log(f"🎧 Yeni Mikrofon Takıldı -> Otomatik Seçildi: {new_in_name}")
            restart_mic = True

        # 2. Yeni HOPARLÖR / KULAKLIK takıldıysa YALNIZCA HOPARLÖR / KULAKLIK KISMINA ATAMA YAP!
        if auto_switch and new_out_names:
            new_out_name = list(new_out_names)[0]
            selected_out_name = new_out_name
            banner_msgs.append(f"🔊 Yeni Hoparlör/Kulaklık Algılandı ve Seçildi: {new_out_name}")
            self._append_log(f"🔊 Yeni Hoparlör/Kulaklık Takıldı -> Otomatik Seçildi: {new_out_name}")
            restart_spk = True

        # 3. Seçili olan mikrofon söküldüyse Otomatik Varsayılana Düş!
        if prev_in_name in removed_in_names and not new_in_names:
            selected_in_name = "Sistem Varsayılanı"
            banner_msgs.append(f"⚠️ Mikrofon ({prev_in_name}) çıkarıldı; Sistem Varsayılanına geçildi.")
            self._append_log(f"⚠️ Mikrofon çıkarıldı ({prev_in_name}) -> Otomatik Varsayılana geçildi.")
            restart_mic = True

        # 4. Seçili olan hoparlör/kulaklık söküldüyse Otomatik Varsayılana Düş!
        if prev_out_name in removed_out_names and not new_out_names:
            selected_out_name = "Sistem Varsayılanı"
            banner_msgs.append(f"⚠️ Hoparlör/Kulaklık ({prev_out_name}) çıkarıldı; Sistem Varsayılanına geçildi.")
            self._append_log(f"⚠️ Hoparlör/Kulaklık çıkarıldı ({prev_out_name}) -> Otomatik Varsayılana geçildi.")
            restart_spk = True

        idx_in  = self.input_combo.findText(selected_in_name)
        idx_out = self.output_combo.findText(selected_out_name)
        self.input_combo.setCurrentIndex(max(0, idx_in))
        self.output_combo.setCurrentIndex(max(0, idx_out))

        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

        val_in  = self.input_combo.currentData()
        val_out = self.output_combo.currentData()

        self.engine.config.setdefault("audio", {}).update({
            "input_device":  val_in,
            "output_device": val_out,
        })

        # Donanım isim kümelerini sakla
        self._previous_wasapi_inputs  = cur_wasapi_in
        self._previous_wasapi_outputs = cur_wasapi_out

        if banner_msgs:
            self._show_banner(" | ".join(banner_msgs), duration_ms=6000)

        # Bağımsız ve milisaniyelik akış güncellemeleri
        if self.engine.get_status().get("running"):
            if restart_mic:
                self.engine.restart_input_stream()
            if restart_spk:
                self.engine.restart_output_stream()

    # ──────────────────────── Yardımcılar ────────────────────────────────────

    def _set_conn_badge(self, state: str):
        bg, fg = CONN_COLORS.get(state, ("#4a4a6a", "#dce1f0"))
        self.conn_badge.setText(state)
        self.conn_badge.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:4px; padding:3px 8px; font-size:9pt; font-weight:700;"
        )

    def _show_banner(self, text: str, duration_ms: int = 0):
        self.banner.setText(text)
        self.banner.setVisible(True)
        if duration_ms > 0:
            self._banner_timer.start(duration_ms)

    def _hide_banner(self):
        self.banner.setVisible(False)

    def _append_log(self, message: str):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{stamp}] {message}")
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _apply_role(self, role: str):
        is_teacher = role == "teacher"
        self.record_group.setVisible(is_teacher)
        if is_teacher:
            self.role_badge.setText("HOCA")
            self.role_badge.setStyleSheet(
                "background:#1e304a; color:#89b4fa; border-radius:5px; padding:4px 10px; font-weight:700;"
            )
        else:
            self.role_badge.setText("ÖĞRENCİ")
            self.role_badge.setStyleSheet(
                "background:#1e3a1e; color:#a6e3a1; border-radius:5px; padding:4px 10px; font-weight:700;"
            )

    # ──────────────────────── Slot Bağlantıları ───────────────────────────────

    def _role_changed(self, label: str):
        new_role = "teacher" if label == "Hoca Paneli" else "student"
        if self.engine.get_status().get("running") and self.engine.config.get("role") != new_role:
            self._show_banner(
                "Rol değiştirmek için önce yayını durdurun.", duration_ms=4000
            )
            self.role_combo.blockSignals(True)
            self.role_combo.setCurrentText(
                "Hoca Paneli" if self.engine.config.get("role") == "teacher" else "Öğrenci Düğümü"
            )
            self.role_combo.blockSignals(False)
            return
        self.engine.config["role"] = new_role
        self.engine._ssrc = 12345 if new_role == "teacher" else 54321
        curr_p = int(self.engine.config.get("network", {}).get("port", 5000))
        if new_role == "teacher" and curr_p == 5001:
            self.port_spin.setValue(4999)
        elif new_role == "student" and curr_p == 4999:
            self.port_spin.setValue(5001)
        self._apply_role(new_role)

    def _denoise_changed(self, label: str):
        internal = DENOISE_MAP.get(label, "webrtc_mild")
        self.engine.update_config({"audio": {"denoise": internal}}, save=False)
        self.denoise_help_lbl.setText(DENOISE_HELP.get(label, ""))
        self.status_denoise.setText(f"{label} / {internal}")

    def _ptt_mode_changed(self, label: str):
        is_ptt = label == "Bas-Konuş"
        self.engine.config.setdefault("ptt", {})["mode"] = "ptt" if is_ptt else "continuous"
        self.ptt_btn.setEnabled(is_ptt)

    def _mute_changed(self, muted: bool):
        self.engine.set_muted(muted)
        if muted:
            self.mute_btn.setText("🔊 Sesi Aç")
            self.broadcast_label.setText("MİKROFON KAPALI")
            self.broadcast_label.setStyleSheet(
                "color:#f38ba8; font-size:18pt; font-weight:700; padding:6px;"
            )
        else:
            self.mute_btn.setText("🔇 Sessize Al")
            if self.engine.get_status().get("running"):
                self.broadcast_label.setText("YAYINDA")
                self.broadcast_label.setStyleSheet(
                    "color:#a6e3a1; font-size:18pt; font-weight:700; padding:6px;"
                )

    def _input_device_changed(self):
        if not hasattr(self, "input_combo"):
            return
        val_in = self.input_combo.currentData()
        self.engine.config.setdefault("audio", {})["input_device"] = val_in
        if self.engine.get_status().get("running"):
            try:
                self.engine.restart_input_stream()
                self._show_banner("Mikrofon cihazı değiştirildi.", duration_ms=2000)
            except Exception as exc:
                self._show_banner(f"Mikrofon değiştirilemedi: {exc}")

    def _output_device_changed(self):
        if not hasattr(self, "output_combo"):
            return
        val_out = self.output_combo.currentData()
        self.engine.config.setdefault("audio", {})["output_device"] = val_out
        if self.engine.get_status().get("running"):
            try:
                self.engine.restart_output_stream()
                self._show_banner("Hoparlör/Kulaklık cihazı değiştirildi.", duration_ms=2000)
            except Exception as exc:
                self._show_banner(f"Hoparlör/Kulaklık değiştirilemedi: {exc}")

    def _on_agc_toggled(self, checked: bool):
        self.engine.set_agc(checked)
        self._append_log(f"Otomatik Kazanç (AGC) {'açıldı' if checked else 'kapatıldı'}.")

    def _on_gain_slider_changed(self, val: int):
        self.input_gain_val_lbl.setText(f"{val}%")
        self.engine.config.setdefault("audio", {})["input_gain"] = val

    def _on_vol_slider_changed(self, val: int):
        self.output_level_slider.setValue(val)
        self.output_level_val_lbl.setText(f"{val}%")
        self.engine.config.setdefault("audio", {})["output_level"] = val

    def _poll_devices(self):
        """Windows WASAPI COM arabirimi ile 0 ms'de canlı ses cihazı takılma/çıkarılma kontrolü (200 ms periyot)."""
        cur_wasapi_in  = set(query_wasapi_endpoints(1)) # Capture
        cur_wasapi_out = set(query_wasapi_endpoints(0)) # Render

        if cur_wasapi_in != self._previous_wasapi_inputs or cur_wasapi_out != self._previous_wasapi_outputs:
            self.refresh_devices(auto_switch=True)

    def _mic_test_changed(self, enabled: bool):
        if enabled and not self.engine.get_status().get("running"):
            try:
                self.engine.start()
            except Exception as exc:
                self._show_banner(f"Mikrofon testi başlatılamadı: {exc}", duration_ms=4000)
                self.mic_test_btn.setChecked(False)
                return
        self.engine.set_monitor_mic(enabled)
        self.mic_test_btn.setText(
            "⏹ Testi Durdur" if enabled else "🎙 Mikrofon Testi"
        )
        self._append_log(
            "Mikrofon testi açıldı; kulaklık kullanın." if enabled
            else "Mikrofon testi kapatıldı."
        )

    def _on_discover_clicked(self):
        self.engine.discover()

    def _connect_from_ui(self):
        target_ip = self.ip_combo.currentText().strip()
        port_num  = self.port_spin.value()
        my_port   = int(self.engine.config.get("network", {}).get("port", 5000))

        if not target_ip:
            self._show_banner("Lütfen bir Hedef IP adresi girin veya listeden seçin.", duration_ms=4000)
            return

        if is_self_endpoint(target_ip, port_num, my_port):
            self._show_banner(f"⚠ Kendi IP adresinize ({target_ip}:{port_num}) bağlanamazsınız! Farklı bir cihaz veya port seçin.", duration_ms=5000)
            self._append_log(f"Engellendi: Kendi yerel IP ve soketinize ({target_ip}:{port_num}) bağlantı kurulamaz.")
            return

        self.engine.connect(target_ip, port_num)
        self._append_log(f"Hedef IP/Port ayarlandı → {target_ip}:{port_num}")

    def _on_ip_combo_changed(self, text: str):
        target_ip = text.strip()
        port_num  = self.port_spin.value()
        my_port   = int(self.engine.config.get("network", {}).get("port", 5000))
        if target_ip and not is_self_endpoint(target_ip, port_num, my_port):
            self.engine.config.setdefault("network", {})["target_ip"] = target_ip

    def _on_port_changed(self, value: int):
        self.engine.set_port(value)
        self._append_log(f"UDP Port güncellendi → {value}")

    def _load_profile(self):
        data = self.profile_combo.currentData()
        if data:
            target_ip = data.get("ip", "").strip()
            port_num  = int(data.get("port", 5000))
            my_port   = int(self.engine.config.get("network", {}).get("port", 5000))
            if target_ip and not is_self_endpoint(target_ip, port_num, my_port):
                if self.ip_combo.findText(target_ip) < 0:
                    self.ip_combo.addItem(target_ip)
                self.ip_combo.setCurrentText(target_ip)
            self.port_spin.setValue(port_num)
            self.engine.connect(target_ip, port_num)

    def _register_global_hotkey(self):
        if getattr(self, "_keyboard", None):
            try:
                self._keyboard.unhook_all()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Global hotkey cleanup failed", exc_info=True
                )
        self._keyboard = None
        try:
            import keyboard
            key = str(self.engine.config.get("ptt", {}).get("shortcut", "space")).lower()
            keyboard.on_press_key(
                key, lambda _: self.bridge.event.emit("global_ptt_press", {}), suppress=False
            )
            keyboard.on_release_key(
                key, lambda _: self.bridge.event.emit("global_ptt_release", {}), suppress=False
            )
            self._keyboard = keyboard
            self.shortcut_info.setText(f"Kısayol: [{key}]")
        except Exception as exc:
            self._append_log(f"Global PTT kısayolu etkinleştirilemedi: {exc}")

    # ──────────────────────── Engine Aksiyonları ─────────────────────────────

    def start_engine(self):
        self._save_quick_settings()
        try:
            self.engine.start()
        except Exception as exc:
            self._show_banner(f"Sistem başlatılamadı: {exc}", duration_ms=6000)
            self._append_log(f"Başlatma hatası: {exc}")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.broadcast_label.setText("YAYINDA")
        self.broadcast_label.setStyleSheet("color:#a6e3a1; font-size:18pt; font-weight:700; padding:6px;")

    def stop_engine(self):
        self.engine.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.broadcast_label.setText("MİKROFON KAPALI")
        self.broadcast_label.setStyleSheet("color:#f38ba8; font-size:18pt; font-weight:700; padding:6px;")

    def _start_recording(self):
        self.engine.start_recording()
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self._record_elapsed = 0
        self._rec_clock.start()
        self.status_record.setText("Kaydediliyor")

    def _stop_recording(self):
        self.engine.stop_recording()
        self._rec_clock.stop()
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        self.status_record.setText("Durduruldu")

    def _tick_record_timer(self):
        self._record_elapsed += 1
        h, rem = divmod(self._record_elapsed, 3600)
        m, s   = divmod(rem, 60)
        self.record_timer.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _save_quick_settings(self):
        audio = self.engine.config.setdefault("audio", {})
        audio["input_device"]   = self.input_combo.currentData()
        audio["output_device"]  = self.output_combo.currentData()
        audio["input_gain"]     = self.input_gain_slider.value()
        audio["output_level"]   = self.output_level_slider.value()
        audio["agc"]            = self.agc_checkbox.isChecked()
        self.engine.save_config()

    def _open_recordings(self):
        path = pathlib.Path(self.engine.recorder.folder).resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _open_logs(self):
        path = pathlib.Path("logs").resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ──────────────────────── Event Handler ──────────────────────────────────

    SILENT = {
        "audio_level", "metrics", "config_changed",
        "ptt_changed", "mute_changed", "monitor_changed",
        "global_ptt_press", "global_ptt_release",
    }

    def _handle_event(self, name: str, payload: dict):
        msg = payload.get("message")
        if msg and name not in self.SILENT:
            self._append_log(msg)

        if name == "connecting":
            self._set_conn_badge("Bağlanıyor")
            self.status_conn.setText("Bağlanıyor")

        elif name == "connected":
            self._set_conn_badge("Bağlı")
            self.status_conn.setText("Bağlı")
            self._hide_banner()

        elif name == "disconnected":
            self._set_conn_badge("Koptu")
            self.status_conn.setText("Koptu")

        elif name == "error":
            tech = payload.get("technical_detail", "")
            now  = time.monotonic()
            if tech != self._last_error_msg or now - self._last_error_time > 5:
                self._last_error_msg  = tech
                self._last_error_time = now
                self._show_banner(msg or "Bir hata oluştu.", duration_ms=6000)

        elif name == "warning":
            self._show_banner(msg or "Uyarı.", duration_ms=4000)

        elif name == "audio_level":
            self.mic_meter.setValue(int(payload.get("input", 0)))
            if payload.get("clipping"):
                self._show_banner("⚠ Ses kırpılıyor — kazancı azaltın.", duration_ms=2000)
            # Slider kullanıcı tercihidir; anlık AGC değeri yalnız etikette gösterilir.
            if self.agc_checkbox.isChecked() and "gain" in payload:
                new_gain = payload["gain"]
                preference = self.input_gain_slider.value()
                self.input_gain_val_lbl.setText(f"{preference}% (AGC: {new_gain}%)")

        elif name == "recording_started":
            self.record_status.setText("Kaydediliyor")
            self.status_record.setText("Kaydediliyor")
            wav = payload.get("wav_path", "")
            if wav:
                self.record_file.setText(pathlib.Path(wav).name)

        elif name == "recording_stopped":
            self.record_status.setText("Durduruldu")
            self.status_record.setText("Durduruldu")

        elif name == "talk_while_muted":
            self._show_banner(
                "🔇 Mikrofonunuz kapalı — sizi duyamıyorlar!", duration_ms=3000
            )

        elif name == "discovered":
            ip = payload.get("ip", "")
            remote_port = payload.get("port") or (self.engine._remote[1] if self.engine._remote else None)
            my_port = int(self.engine.config.get("network", {}).get("port", 5000))
            if ip and not is_self_endpoint(ip, remote_port, my_port):
                if self.ip_combo.findText(ip) < 0:
                    self.ip_combo.addItem(ip)
                self.ip_combo.setCurrentText(ip)
                if remote_port:
                    self.port_spin.setValue(remote_port)
                self.conn_badge.setText(f"Ağda cihaz bulundu: {ip}")
                self._set_conn_badge("Bağlanıyor")
                self._append_log(f"🔗 Ağda Cihaz Algılandı → Hedef IP Listesine Eklendi: {ip}:{remote_port or ''}")

        elif name == "local_ips_changed":
            ips = payload.get("ips", [])
            self.local_ip_label.setText(f"Yerel IP: {', '.join(ips)}")
            self._append_log(f"🌐 Yerel Ağ IP Değişimi Algılandı: {', '.join(ips)}")

        elif name == "global_ptt_press" and self.mode_combo.currentText() == "Bas-Konuş":
            self.engine.set_ptt(True)
            self.ptt_btn.setDown(True)

        elif name == "global_ptt_release" and self.mode_combo.currentText() == "Bas-Konuş":
            self.engine.set_ptt(False)
            self.ptt_btn.setDown(False)

        elif name == "device_changed":
            self._show_banner(msg or "Ses cihazı değiştirildi.", duration_ms=2000)

    # ──────────────────────── Ayarlar ────────────────────────────────────────

    def open_settings(self, *, tab: int = 0):
        dlg = SettingsDialog(self.engine.config_path, self, initial_tab=tab)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.engine.reload_config()
            self._load_config_into_ui()
            self.refresh_devices(auto_switch=False)
            self._register_global_hotkey()

    # ──────────────────────── Pencere Kapanışı ────────────────────────────────

    def closeEvent(self, event):
        try:
            if getattr(self, "_keyboard", None):
                self._keyboard.unhook_all()
            self.engine.set_ptt(False)
            self._save_quick_settings()
            self.engine.set_prevent_sleep(False)
            self.engine.stop()
        finally:
            event.accept()


# ─────────────────────────── main() ──────────────────────────────────────────
def main():
    app = QtWidgets.QApplication(sys.argv)
    config = pathlib.Path(__file__).resolve().parents[1] / "config" / "teacher.json"
    window = MainWindow(str(config))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
