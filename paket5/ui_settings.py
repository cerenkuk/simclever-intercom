"""
paket5/ui_settings.py – SimClever Intercom Paket 5 Ayarlar Penceresi
7 sekmeli detaylı ayarlar diyaloğu.
"""
import json
import os
import pathlib
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

DENOISE_LABELS = ["Düşük", "Orta", "Yüksek", "Kapalı"]
DENOISE_HELP = {
    "Düşük":  "Hafif ortam gürültüleri için — önerilen günlük kullanım.",
    "Orta":   "Gürültülü sınıf ortamları için.",
    "Yüksek": "Yoğun gürültü için; kısık sesler zayıflayabilir.",
    "Kapalı": "Ses işlenmeden iletilir.",
}
DENOISE_MAP  = {"Düşük": "webrtc_mild", "Orta": "webrtc_aggressive", "Yüksek": "webrtc_max", "Kapalı": "none"}
DENOISE_RMAP = {v: k for k, v in DENOISE_MAP.items()}

EXPERIMENTAL = ["rnnoise", "deepfilter", "speex", "wiener", "spectral_subtraction"]

STYLE_SETTINGS = """
QDialog { background:#12121e; color:#dce1f0; font:10pt 'Segoe UI'; }
QTabWidget::pane { border:1px solid #2e2e4a; background:#1a1a2e; border-radius:6px; }
QTabBar::tab {
    background:#181828; color:#a0a8c0; border:1px solid #2e2e4a;
    padding:8px 14px; margin-right:2px; border-top-left-radius:5px; border-top-right-radius:5px;
    font-weight:600;
}
QTabBar::tab:selected { background:#1a1a2e; color:#89b4fa; border-bottom-color:#1a1a2e; }
QTabBar::tab:hover    { background:#222238; color:#dce1f0; }

QGroupBox {
    background:#161628; border:1px solid #2e2e4a; border-radius:6px;
    margin-top:16px; padding:16px 12px 12px 12px; font-weight:700; color:#f9e2af;
}
QGroupBox::title { subcontrol-origin:margin; subcontrol-position:top left; left:12px; padding:0 6px; }

QLineEdit, QComboBox, QSpinBox {
    background:#0d0d1a; border:1px solid #3a3a5c; border-radius:4px;
    padding:4px 8px; color:#f0f0f8; min-height:22px;
}
QLineEdit:focus, QComboBox:focus { border-color:#89b4fa; }

QPushButton {
    background:#252540; border:1px solid #3a3a5c; border-radius:5px;
    padding:6px 12px; font-weight:600; color:#dce1f0;
}
QPushButton:hover   { background:#30304e; border-color:#585b70; }
QPushButton:pressed { background:#1e1e36; }
QPushButton#saveBtn { background:#1e4a2e; border-color:#a6e3a1; color:#a6e3a1; font-weight:700; }
QPushButton#saveBtn:hover { background:#2a6040; }

QSlider::groove:horizontal { background:#1a1a2e; height:6px; border-radius:3px; }
QSlider::handle:horizontal { background:#89b4fa; width:14px; height:14px; border-radius:7px; margin:-4px 0; }

QPlainTextEdit, QListWidget {
    background:#08080f; color:#a6e3a1; border:1px solid #2a2a44; border-radius:4px; font:9pt Consolas;
}

QCheckBox { color:#dce1f0; font-weight:600; }
QCheckBox::indicator { width:16px; height:16px; border-radius:3px; border:1px solid #3a3a5c; background:#0d0d1a; }
QCheckBox::indicator:checked { background:#89b4fa; border-color:#89b4fa; }
"""


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config_path, parent=None, *, initial_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("SimClever Intercom — Ayarlar")
        self.resize(760, 560)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(STYLE_SETTINGS)

        self.config_path = Path(config_path)
        self.config      = self._load()
        self._profiles   = list(self.config.get("profiles", []))

        root = QtWidgets.QVBoxLayout(self)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        self._tab_audio()
        self._tab_connection()
        self._tab_ptt()
        self._tab_recording()
        self._tab_processing()
        self._tab_diagnostics()
        self._tab_advanced()

        self.tabs.setCurrentIndex(initial_tab)

        # Butonlar
        btn_row = QtWidgets.QHBoxLayout()
        restore_btn = QtWidgets.QPushButton("Varsayılanlara Dön")
        restore_btn.clicked.connect(self._restore_defaults)
        btn_row.addWidget(restore_btn)
        btn_row.addStretch()

        apply_btn = QtWidgets.QPushButton("Uygula")
        save_btn  = QtWidgets.QPushButton("Kaydet")
        save_btn.setObjectName("saveBtn")
        cancel_btn= QtWidgets.QPushButton("İptal")

        apply_btn.clicked.connect(self._apply)
        save_btn.clicked.connect(self._save_and_close)
        cancel_btn.clicked.connect(self.reject)

        for b in (apply_btn, save_btn, cancel_btn):
            btn_row.addWidget(b)

        root.addLayout(btn_row)

        self._populate()

    # ─────────────────────── Yardımcılar ─────────────────────────────────────

    def _page(self, title: str) -> QtWidgets.QFormLayout:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        w = QtWidgets.QWidget()
        w.setStyleSheet("background:transparent;")
        form = QtWidgets.QFormLayout(w)
        form.setSpacing(12)
        form.setContentsMargins(16, 16, 16, 16)
        scroll.setWidget(w)
        self.tabs.addTab(scroll, title)
        return form

    def _page_vbox(self, title: str) -> QtWidgets.QVBoxLayout:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("background:transparent;")
        w = QtWidgets.QWidget()
        w.setStyleSheet("background:transparent;")
        vbox = QtWidgets.QVBoxLayout(w)
        vbox.setSpacing(12)
        vbox.setContentsMargins(16, 16, 16, 16)
        scroll.setWidget(w)
        self.tabs.addTab(scroll, title)
        return vbox

    def _help(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#808098; font-size:9pt;")
        return lbl

    # ─────────────────────── Sekmeler ────────────────────────────────────────

    # 0 – Ses ve Cihazlar
    def _tab_audio(self):
        form = self._page("Ses ve Cihazlar")

        self.s_input_combo  = QtWidgets.QComboBox()
        self.s_output_combo = QtWidgets.QComboBox()
        form.addRow("Mikrofon:",  self.s_input_combo)
        form.addRow("Hoparlör:",  self.s_output_combo)
        form.addRow(self._help("Cihazları listelemek için Yenile'ye basın. Yeni cihaz takıldığında otomatik seçilir."))

        refresh = QtWidgets.QPushButton("↻ Cihazları Yenile")
        refresh.clicked.connect(self._refresh_devices_settings)
        form.addRow(refresh)

        form.addRow(QtWidgets.QLabel(""))

        self.s_agc_checkbox = QtWidgets.QCheckBox("Otomatik Kazanç Kontrolü (AGC)")
        form.addRow(self.s_agc_checkbox)

        self.s_input_gain   = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s_input_gain.setRange(0, 150)
        self.s_input_gain.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.s_input_gain.setTickInterval(25)
        self.s_gain_val_lbl = QtWidgets.QLabel("100")
        self.s_input_gain.valueChanged.connect(lambda v: self.s_gain_val_lbl.setText(str(v)))
        gain_row = QtWidgets.QHBoxLayout()
        gain_row.addWidget(self.s_input_gain)
        gain_row.addWidget(self.s_gain_val_lbl)
        form.addRow("Giriş kazancı (%):", gain_row)

        self.s_output_level = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.s_output_level.setRange(0, 100)
        self.s_output_level.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.s_output_level.setTickInterval(25)
        self.s_vol_val_lbl  = QtWidgets.QLabel("100")
        self.s_output_level.valueChanged.connect(lambda v: self.s_vol_val_lbl.setText(str(v)))
        vol_row = QtWidgets.QHBoxLayout()
        vol_row.addWidget(self.s_output_level)
        vol_row.addWidget(self.s_vol_val_lbl)
        form.addRow("Çıkış seviyesi (%):", vol_row)

        form.addRow(self._help(
            "Giriş kazancı 100'ün üzerinde ses amplifikasyonu yapar; AGC açıkken otomatik ayarlanır."
        ))

    # 1 – Bağlantı ve Profiller
    def _tab_connection(self):
        vbox = self._page_vbox("Bağlantı ve Profiller")

        # Profil listesi
        profile_grp = QtWidgets.QGroupBox("Profiller")
        pg = QtWidgets.QVBoxLayout(profile_grp)
        self.s_profile_list = QtWidgets.QListWidget()
        self.s_profile_list.setMaximumHeight(120)
        pg.addWidget(self.s_profile_list)

        profile_btns = QtWidgets.QHBoxLayout()
        self.s_profile_add_btn = QtWidgets.QPushButton("+ Ekle")
        self.s_profile_del_btn = QtWidgets.QPushButton("✖ Sil")
        self.s_profile_set_btn = QtWidgets.QPushButton("✓ Varsayılan Yap")
        for b in (self.s_profile_add_btn, self.s_profile_del_btn, self.s_profile_set_btn):
            profile_btns.addWidget(b)
        profile_btns.addStretch()
        pg.addLayout(profile_btns)
        vbox.addWidget(profile_grp)

        self.s_profile_add_btn.clicked.connect(self._add_profile)
        self.s_profile_del_btn.clicked.connect(self._del_profile)
        self.s_profile_list.currentRowChanged.connect(self._profile_selected)

        # Profil detay formu
        detail_grp = QtWidgets.QGroupBox("Profil Detayı")
        dg = QtWidgets.QFormLayout(detail_grp)
        self.s_profile_name = QtWidgets.QLineEdit()
        self.s_profile_ip   = QtWidgets.QLineEdit()
        self.s_profile_ip.setPlaceholderText("192.168.1.xxx")
        self.s_profile_port = QtWidgets.QSpinBox()
        self.s_profile_port.setRange(1, 65535)
        self.s_profile_port.setValue(5000)
        dg.addRow("Profil adı:", self.s_profile_name)
        dg.addRow("Hedef IP:",   self.s_profile_ip)
        dg.addRow("UDP port:",   self.s_profile_port)
        save_profile_btn = QtWidgets.QPushButton("Profili Güncelle")
        save_profile_btn.clicked.connect(self._update_profile)
        dg.addRow(save_profile_btn)
        vbox.addWidget(detail_grp)

        # Ağ bilgisi
        net_grp = QtWidgets.QGroupBox("Ağ Bilgisi")
        ng = QtWidgets.QFormLayout(net_grp)
        self.s_local_ip   = QtWidgets.QLabel("—")
        self.s_active_iface = QtWidgets.QLabel("—")
        ng.addRow("Yerel IP:",           self.s_local_ip)
        ng.addRow("Aktif ağ arayüzü:",   self.s_active_iface)
        ng.addRow(self._help(
            "Windows Güvenlik Duvarı 5000/UDP portuna izin verilmezse bağlantı kurulamaz. "
            "Güvenlik duvarı ayarları için: Denetim Masası → Windows Defender Güvenlik Duvarı → "
            "Gelişmiş ayarlar → Gelen Kurallar → Yeni Kural (UDP 5000)"
        ))
        vbox.addWidget(net_grp)
        vbox.addStretch()

        try:
            import socket as _s
            hostname = _s.gethostname()
            local_ip = _s.gethostbyname(hostname)
            self.s_local_ip.setText(local_ip)
        except Exception:
            self.s_local_ip.setText("Alınamadı")

    # 2 – PTT ve Kısayollar
    def _tab_ptt(self):
        form = self._page("PTT ve Kısayollar")

        self.s_ptt_mode = QtWidgets.QComboBox()
        self.s_ptt_mode.addItems(["Sürekli Açık", "Bas-Konuş"])
        form.addRow("Yayın modu:", self.s_ptt_mode)
        form.addRow(self._help(
            "Sürekli Açık: mikrofon her zaman aktif.\n"
            "Bas-Konuş: yalnızca kısayol tuşuna basılıyken ses iletilir."
        ))

        form.addRow(QtWidgets.QLabel(""))
        self.s_ptt_shortcut = QtWidgets.QLineEdit()
        self.s_ptt_shortcut.setPlaceholderText("Örn: space, ctrl+m")
        form.addRow("Kısayol tuşu:", self.s_ptt_shortcut)
        form.addRow(self._help(
            "keyboard kütüphanesi yüklüyse global kısayol etkindir. "
            "Değiştirmek için burada yazın, Kaydet'e basın ve uygulamayı yeniden başlatın."
        ))

        form.addRow(QtWidgets.QLabel(""))
        self.s_talk_warning = QtWidgets.QCheckBox(
            "Mikrofon kapalıyken konuşulduğunda uyarı göster"
        )
        form.addRow(self.s_talk_warning)
        form.addRow(self._help(
            "VAD veya sinyal tespiti ile kapalı mikrofonda konuşma algılandığında ekranda sönümlenen uyarı çıkar."
        ))

        self.s_mute_behavior = QtWidgets.QComboBox()
        self.s_mute_behavior.addItems([
            "Yalnızca gönderimi durdur",
            "Gönderimi ve kaydı durdur",
        ])
        form.addRow("Sessize alma davranışı:", self.s_mute_behavior)

    # 3 – Kayıt
    def _tab_recording(self):
        form = self._page("Kayıt")

        self.s_rec_enabled = QtWidgets.QCheckBox("Kayıt etkin (yalnızca hoca rolünde)")
        form.addRow(self.s_rec_enabled)

        folder_row = QtWidgets.QHBoxLayout()
        self.s_rec_folder = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Klasör Seç")
        browse_btn.clicked.connect(self._choose_folder)
        open_btn   = QtWidgets.QPushButton("Klasörü Aç")
        open_btn.clicked.connect(self._open_rec_folder)
        folder_row.addWidget(self.s_rec_folder)
        folder_row.addWidget(browse_btn)
        folder_row.addWidget(open_btn)
        form.addRow("Kayıt klasörü:", folder_row)

        form.addRow(self._help(
            "Kayıtlar ham PCM WAV formatında ve her oturum için ayrı JSON sidecar dosyasıyla saklanır."
        ))

        self.s_sidecar_interval = QtWidgets.QSpinBox()
        self.s_sidecar_interval.setRange(1, 60)
        self.s_sidecar_interval.setValue(5)
        self.s_sidecar_interval.setSuffix(" sn")
        form.addRow("Sidecar güncelleme aralığı:", self.s_sidecar_interval)

        self.s_min_disk = QtWidgets.QSpinBox()
        self.s_min_disk.setRange(50, 10000)
        self.s_min_disk.setValue(200)
        self.s_min_disk.setSuffix(" MB")
        form.addRow("Minimum boş disk alanı:", self.s_min_disk)

    # 4 – Ses İşleme
    def _tab_processing(self):
        vbox = self._page_vbox("Ses İşleme")

        dn_grp = QtWidgets.QGroupBox("Gürültü Bastırma")
        dg = QtWidgets.QFormLayout(dn_grp)
        self.s_denoise = QtWidgets.QComboBox()
        self.s_denoise.addItems(DENOISE_LABELS)
        self.s_denoise_help = QtWidgets.QLabel()
        self.s_denoise_help.setWordWrap(True)
        self.s_denoise_help.setStyleSheet("color:#808098; font-size:9pt;")
        self.s_denoise.currentTextChanged.connect(
            lambda t: self.s_denoise_help.setText(DENOISE_HELP.get(t, ""))
        )
        dg.addRow("Seviye:", self.s_denoise)
        dg.addRow(self.s_denoise_help)
        vbox.addWidget(dn_grp)

        codec_grp = QtWidgets.QGroupBox("Ses Kodlama (Codec)")
        cg = QtWidgets.QFormLayout(codec_grp)
        self.s_codec = QtWidgets.QComboBox()
        self.s_codec.addItems(["PCM (Güvenli Varsayılan)", "Opus (kullanılabilirse)"])
        cg.addRow("Codec:", self.s_codec)
        vbox.addWidget(codec_grp)

        aec_grp = QtWidgets.QGroupBox("Yankı Engelleme (AEC)")
        ag = QtWidgets.QFormLayout(aec_grp)
        self.s_aec = QtWidgets.QComboBox()
        self.s_aec.addItems(["WebRTC AEC", "Kapalı"])
        ag.addRow("AEC:", self.s_aec)
        vbox.addWidget(aec_grp)

        vad_grp = QtWidgets.QGroupBox("Konuşma Algılama (VAD)")
        vg = QtWidgets.QFormLayout(vad_grp)
        self.s_vad = QtWidgets.QComboBox()
        self.s_vad.addItems(["Kapalı", "RMS (Basit)", "WebRTC VAD", "Silero (Derin Öğrenme)"])
        vg.addRow("VAD:", self.s_vad)
        vbox.addWidget(vad_grp)

        exp_grp = QtWidgets.QGroupBox("Deneysel / Gelişmiş Filtreler")
        eg = QtWidgets.QFormLayout(exp_grp)
        self.s_experimental = QtWidgets.QComboBox()
        self.s_experimental.addItems(["(Hiçbiri)"] + EXPERIMENTAL)
        eg.addRow("Ek filtre:", self.s_experimental)
        vbox.addWidget(exp_grp)
        vbox.addStretch()

    # 5 – Tanılama ve Loglar
    def _tab_diagnostics(self):
        vbox = self._page_vbox("Tanılama ve Loglar")

        stat_grp = QtWidgets.QGroupBox("Anlık Durum")
        sg = QtWidgets.QFormLayout(stat_grp)
        self.diag_sent     = QtWidgets.QLabel("—")
        self.diag_received = QtWidgets.QLabel("—")
        self.diag_remote   = QtWidgets.QLabel("—")
        self.diag_codec    = QtWidgets.QLabel("—")
        self.diag_aec      = QtWidgets.QLabel("—")
        self.diag_vad      = QtWidgets.QLabel("—")
        self.diag_denoise  = QtWidgets.QLabel("—")
        sg.addRow("Gönderilen paket:",  self.diag_sent)
        sg.addRow("Alınan paket:",      self.diag_received)
        sg.addRow("Uzak adres:",        self.diag_remote)
        sg.addRow("Aktif codec:",       self.diag_codec)
        sg.addRow("Aktif AEC:",         self.diag_aec)
        sg.addRow("Aktif VAD:",         self.diag_vad)
        sg.addRow("Aktif denoise:",     self.diag_denoise)
        vbox.addWidget(stat_grp)

        log_grp = QtWidgets.QGroupBox("Teknik Log")
        lg = QtWidgets.QVBoxLayout(log_grp)
        self.diag_log = QtWidgets.QPlainTextEdit()
        self.diag_log.setReadOnly(True)
        self.diag_log.setMinimumHeight(160)
        self.diag_log.setMaximumBlockCount(500)
        self.diag_log.setFont(QtGui.QFont("Consolas", 9))
        lg.addWidget(self.diag_log)
        btn_row = QtWidgets.QHBoxLayout()
        clear_btn  = QtWidgets.QPushButton("Temizle")
        export_btn = QtWidgets.QPushButton("Panoya Kopyala")
        open_btn   = QtWidgets.QPushButton("Log Klasörü")
        clear_btn.clicked.connect(self.diag_log.clear)
        export_btn.clicked.connect(
            lambda: QtWidgets.QApplication.clipboard().setText(self.diag_log.toPlainText())
        )
        open_btn.clicked.connect(self._open_log_folder)
        for b in (clear_btn, export_btn, open_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        lg.addLayout(btn_row)
        vbox.addWidget(log_grp)
        vbox.addStretch()

    # 6 – Gelişmiş
    def _tab_advanced(self):
        form = self._page("Gelişmiş")

        self.s_prevent_sleep = QtWidgets.QCheckBox(
            "Seans sırasında bilgisayarın uyumasını engelle (Windows)"
        )
        form.addRow(self.s_prevent_sleep)

        form.addRow(QtWidgets.QLabel(""))
        self.s_single_instance = QtWidgets.QCheckBox("Tek örnek kilidi (önerilir)")
        self.s_single_instance.setChecked(True)
        self.s_single_instance.setEnabled(False)
        form.addRow(self.s_single_instance)

        form.addRow(QtWidgets.QLabel(""))
        config_lbl = QtWidgets.QLabel()
        config_lbl.setText(f"Ayar dosyası: {self.config_path}")
        config_lbl.setWordWrap(True)
        config_lbl.setStyleSheet("color:#808098; font-size:9pt;")
        form.addRow(config_lbl)

    # ─────────────────────── Populate ────────────────────────────────────────

    def _populate(self):
        audio     = self.config.get("audio", {})
        net       = self.config.get("network", {})
        ptt       = self.config.get("ptt", {})
        recording = self.config.get("recording", {})
        advanced  = self.config.get("advanced", {})

        # Ses ve Cihazlar
        self.s_agc_checkbox.setChecked(bool(audio.get("agc", True)))
        self.s_input_gain.setValue(int(audio.get("input_gain", 100)))
        self.s_output_level.setValue(int(audio.get("output_level", 100)))
        self._refresh_devices_settings()

        # Profiller
        self._refresh_profile_list()

        # PTT
        self.s_ptt_mode.setCurrentText(
            "Bas-Konuş" if ptt.get("mode") == "ptt" else "Sürekli Açık"
        )
        self.s_ptt_shortcut.setText(ptt.get("shortcut", "Space"))
        self.s_talk_warning.setChecked(bool(ptt.get("talk_warning", True)))

        # Kayıt
        self.s_rec_enabled.setChecked(bool(recording.get("enabled", True)))
        self.s_rec_folder.setText(recording.get("folder", "recordings"))
        self.s_sidecar_interval.setValue(int(recording.get("sidecar_update_interval_sec", 5)))

        # Ses İşleme
        denoise_label = DENOISE_RMAP.get(audio.get("denoise", "webrtc_mild"), "Düşük")
        self.s_denoise.setCurrentText(denoise_label)
        self.s_denoise_help.setText(DENOISE_HELP.get(denoise_label, ""))
        codec_val = audio.get("codec", "pcm")
        self.s_codec.setCurrentText(
            "Opus (kullanılabilirse)" if "opus" in codec_val else "PCM (Güvenli Varsayılan)"
        )
        aec_val = audio.get("aec", "webrtc")
        self.s_aec.setCurrentText("Kapalı" if aec_val == "none" else "WebRTC AEC")
        vad_val = audio.get("vad", "none")
        vad_map  = {"none": "Kapalı", "rms": "RMS (Basit)", "webrtc": "WebRTC VAD", "silero": "Silero (Derin Öğrenme)"}
        self.s_vad.setCurrentText(vad_map.get(vad_val, "Kapalı"))

        # Gelişmiş
        self.s_prevent_sleep.setChecked(bool(advanced.get("prevent_sleep", False)))

        # Tanılama bilgileri
        self.diag_codec.setText(codec_val)
        self.diag_aec.setText(aec_val)
        self.diag_vad.setText(vad_val)
        self.diag_denoise.setText(audio.get("denoise", "webrtc_mild"))

    # ─────────────────────── Cihazlar ────────────────────────────────────────

    def _refresh_devices_settings(self):
        try:
            import sounddevice as sd
            devices = sd.query_devices()
        except Exception:
            devices = []

        prev_in  = self.config.get("audio", {}).get("input_device")
        prev_out = self.config.get("audio", {}).get("output_device")

        self.s_input_combo.blockSignals(True)
        self.s_output_combo.blockSignals(True)
        self.s_input_combo.clear()
        self.s_output_combo.clear()
        self.s_input_combo.addItem("Sistem Varsayılanı", None)
        self.s_output_combo.addItem("Sistem Varsayılanı", None)

        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                self.s_input_combo.addItem(d["name"], i)
            if d.get("max_output_channels", 0) > 0:
                self.s_output_combo.addItem(d["name"], i)

        for combo, val in ((self.s_input_combo, prev_in), (self.s_output_combo, prev_out)):
            idx = combo.findData(val)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

    # ─────────────────────── Profil İşlemleri ────────────────────────────────

    def _refresh_profile_list(self):
        self.s_profile_list.clear()
        for p in self._profiles:
            self.s_profile_list.addItem(f"{p.get('name','')}  —  {p.get('ip','')}:{p.get('port',5000)}")

    def _profile_selected(self, row: int):
        if 0 <= row < len(self._profiles):
            p = self._profiles[row]
            self.s_profile_name.setText(p.get("name", ""))
            self.s_profile_ip.setText(p.get("ip", ""))
            self.s_profile_port.setValue(int(p.get("port", 5000)))

    def _add_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Yeni Profil", "Profil adı:")
        if ok and name.strip():
            self._profiles.append({"name": name.strip(), "ip": "", "port": 5000})
            self._refresh_profile_list()
            self.s_profile_list.setCurrentRow(len(self._profiles) - 1)

    def _del_profile(self):
        row = self.s_profile_list.currentRow()
        if 0 <= row < len(self._profiles):
            del self._profiles[row]
            self._refresh_profile_list()

    def _update_profile(self):
        row = self.s_profile_list.currentRow()
        if 0 <= row < len(self._profiles):
            self._profiles[row] = {
                "name": self.s_profile_name.text().strip(),
                "ip":   self.s_profile_ip.text().strip(),
                "port": self.s_profile_port.value(),
            }
            self._refresh_profile_list()
            self.s_profile_list.setCurrentRow(row)

    # ─────────────────────── Kayıt Yardımcıları ──────────────────────────────

    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Kayıt Klasörü", self.s_rec_folder.text() or "."
        )
        if folder:
            self.s_rec_folder.setText(folder)

    def _open_rec_folder(self):
        path = pathlib.Path(self.s_rec_folder.text() or "recordings").resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _open_log_folder(self):
        path = pathlib.Path("logs").resolve()
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    # ─────────────────────── Config Yükle ────────────────────────────────────

    def _load(self) -> dict:
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    # ─────────────────────── Config → Kaydet ─────────────────────────────────

    def _collect(self) -> dict:
        """UI → config dict."""
        cfg = self.config

        # Ses
        audio = cfg.setdefault("audio", {})
        audio["input_device"]   = self.s_input_combo.currentData()
        audio["output_device"]  = self.s_output_combo.currentData()
        audio["input_gain"]     = self.s_input_gain.value()
        audio["output_level"]   = self.s_output_level.value()
        audio["agc"]            = self.s_agc_checkbox.isChecked()
        audio["denoise"]        = DENOISE_MAP[self.s_denoise.currentText()]
        audio["codec"]          = (
            "opus_64" if self.s_codec.currentText().startswith("Opus") else "pcm"
        )
        audio["aec"]  = "none" if self.s_aec.currentText() == "Kapalı" else "webrtc"
        vad_rmap = {
            "Kapalı": "none",
            "RMS (Basit)": "rms",
            "WebRTC VAD": "webrtc",
            "Silero (Derin Öğrenme)": "silero",
        }
        audio["vad"] = vad_rmap.get(self.s_vad.currentText(), "none")

        # PTT
        ptt = cfg.setdefault("ptt", {})
        ptt["mode"]         = "ptt" if self.s_ptt_mode.currentText() == "Bas-Konuş" else "continuous"
        ptt["shortcut"]     = self.s_ptt_shortcut.text().strip() or "Space"
        ptt["talk_warning"] = self.s_talk_warning.isChecked()

        # Kayıt
        rec = cfg.setdefault("recording", {})
        rec["enabled"]                   = self.s_rec_enabled.isChecked()
        rec["folder"]                    = self.s_rec_folder.text().strip() or "recordings"
        rec["sidecar_update_interval_sec"] = self.s_sidecar_interval.value()

        # Gelişmiş
        cfg.setdefault("advanced", {})["prevent_sleep"] = self.s_prevent_sleep.isChecked()

        # Profiller
        cfg["profiles"] = self._profiles

        return cfg

    def _write_config(self):
        cfg = self._collect()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        tmp.replace(self.config_path)

    def _apply(self):
        """Kaydetmeden canlı uygula."""
        self._collect()

    def _save_and_close(self):
        self._write_config()
        super().accept()

    def _restore_defaults(self):
        self.s_denoise.setCurrentText("Düşük")
        self.s_codec.setCurrentText("PCM (Güvenli Varsayılan)")
        self.s_aec.setCurrentText("WebRTC AEC")
        self.s_vad.setCurrentText("Kapalı")
        self.s_agc_checkbox.setChecked(True)
        self.s_input_gain.setValue(100)
        self.s_output_level.setValue(100)
        self.s_profile_port.setValue(5000)
        self.s_ptt_mode.setCurrentText("Sürekli Açık")
        self.s_ptt_shortcut.setText("Space")
        self.s_rec_folder.setText("recordings")
        self.s_prevent_sleep.setChecked(False)
