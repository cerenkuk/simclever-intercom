"""tests/paket5/test_ui_smoke.py

PyQt5 offscreen smoke testleri – yeni arayüz ve ağ özellikleri için güncellenmiş.
"""
import json
import os
import pathlib
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="module")
def qt_app():
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    yield app


@pytest.fixture
def config_path(tmp_path):
    cfg = {
        "role": "teacher",
        "audio": {
            "sample_rate": 16000,
            "frame_size": 160,
            "denoise": "webrtc_mild",
            "aec": "none",
            "vad": "none",
            "codec": "pcm",
            "input_gain": 100,
            "output_level": 100,
            "agc": True,
        },
        "network": {"bind_ip": "0.0.0.0", "port": 5000, "target_ip": ""},
        "ptt": {"mode": "continuous", "shortcut": "Space", "talk_warning": True},
        "recording": {"enabled": False, "folder": str(tmp_path / "recordings")},
        "profiles": [],
    }
    p = tmp_path / "teacher.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return str(p)


@pytest.fixture
def win(qt_app, config_path, fake_sounddevice_streams):
    from paket5.ui_main import MainWindow
    w = MainWindow(config_path)
    yield w
    w.close()


# ─── Temel pencere testleri ───────────────────────────────────────────────────

def test_main_window_creates(qt_app, config_path):
    """MainWindow hata vermeden oluşturulabilmeli."""
    from paket5.ui_main import MainWindow
    win = MainWindow(config_path)
    assert win is not None


def test_main_window_title(win):
    """Pencere başlığı 'SimClever' içermeli."""
    assert "SimClever" in win.windowTitle()


def test_main_window_shows(qt_app, config_path):
    """MainWindow.show() exception üretmemeli ve pencere görünür olmalı."""
    from paket5.ui_main import MainWindow
    win = MainWindow(config_path)
    win.show()
    assert win.isVisible()
    win.close()


def test_engine_accessible(win):
    """win.engine IntercomEngine nesnesi döndürmeli."""
    from shared.intercom_engine import IntercomEngine
    assert isinstance(win.engine, IntercomEngine)


# ─── Varsayılan denoise ────────────────────────────────────────────────────────

def test_default_denoise_is_webrtc_mild(win):
    """Engine config'de varsayılan denoise webrtc_mild olmalı."""
    denoise = win.engine.config.get("audio", {}).get("denoise")
    assert denoise == "webrtc_mild", f"Beklenen 'webrtc_mild', alınan '{denoise}'"


def test_denoise_combo_default_is_dusuk(win):
    """Ana ekranda denoise combo 'Düşük' olarak başlamalı."""
    assert win.denoise_combo.currentText() == "Düşük"


def test_denoise_map_dusuk(win):
    """Düşük seçilince engine config'e webrtc_mild yazılmalı."""
    win.denoise_combo.setCurrentText("Düşük")
    assert win.engine.config["audio"]["denoise"] == "webrtc_mild"


def test_denoise_map_orta(win):
    """Orta seçilince engine config'e webrtc_aggressive yazılmalı."""
    win.denoise_combo.setCurrentText("Orta")
    assert win.engine.config["audio"]["denoise"] == "webrtc_aggressive"


def test_denoise_map_kapali(win):
    """Kapalı seçilince engine config'e none yazılmalı."""
    win.denoise_combo.setCurrentText("Kapalı")
    assert win.engine.config["audio"]["denoise"] == "none"


# ─── Rol kontrolü ─────────────────────────────────────────────────────────────

def test_teacher_role_shows_recording(win):
    """Hoca rolünde kayıt kartı gizli olmamalı."""
    win.role_combo.setCurrentText("Öğrenci Düğümü")
    win.role_combo.setCurrentText("Hoca Paneli")
    assert not win.record_group.isHidden()


def test_student_role_hides_recording(win):
    """Öğrenci rolünde kayıt kartı gizli olmalı."""
    win.role_combo.setCurrentText("Öğrenci Düğümü")
    assert win.record_group.isHidden()
    win.role_combo.setCurrentText("Hoca Paneli")


# ─── Bağlantı & Ağ Widget Testleri ───────────────────────────────────────────

def test_connection_badge_default(win):
    """Başlangıçta bağlantı rozeti 'Hazır' göstermeli."""
    assert "Hazır" in win.conn_badge.text()


def test_local_ip_label_exists(win):
    """Yerel IP etiketi başlıkta mevcut ve dolu olmalı."""
    assert win.local_ip_label is not None
    assert "Yerel IP:" in win.local_ip_label.text()


def test_ip_combo_exists(win):
    """Target IP combo kutusu mevcut ve düzenlenebilir olmalı."""
    assert win.ip_combo is not None
    assert win.ip_combo.isEditable()


def test_port_spin_exists(win):
    """UDP Port spinbox mevcut ve değeri 5000 olmalı."""
    assert win.port_spin is not None
    assert win.port_spin.value() == 5000


def test_self_connection_blocked(win):
    """Kendine (127.0.0.1 veya yerel IP) bağlanma engellenmeli."""
    from shared.intercom_engine import get_local_ips
    local_ips = get_local_ips()
    my_ip = local_ips[0]
    win.ip_combo.setCurrentText(my_ip)
    win._connect_from_ui()
    assert win.engine._remote is None or win.engine._remote[0] != my_ip
    assert not win.banner.isHidden()
    assert "Kendi IP" in win.banner.text()


def test_gain_sliders_and_agc_exist(win):
    """Ana ekranda kazanç slider'ları ve AGC checkbox'ı olmalı."""
    assert win.input_gain_slider is not None
    assert win.output_level_slider is not None
    assert win.agc_checkbox is not None
    assert win.agc_checkbox.isChecked()


def test_agc_event_does_not_replace_user_gain(win):
    win.input_gain_slider.setValue(75)

    win._handle_event("audio_level", {"input": 10, "gain": 120})

    assert win.input_gain_slider.value() == 75
    assert "AGC: 120%" in win.input_gain_val_lbl.text()


# ─── Kritik widget varlıkları ─────────────────────────────────────────────────

def test_start_button_exists(win):
    assert win.start_btn is not None


def test_stop_button_disabled_initially(win):
    assert not win.stop_btn.isEnabled()


def test_start_stop_broadcast_flow(win):
    """YAYINI BAŞLAT ve DURDUR düğmeleri akışı doğru değiştirmeli."""
    assert not win.engine._running
    win.start_engine()
    assert win.engine._running
    assert win.stop_btn.isEnabled()
    assert not win.start_btn.isEnabled()
    assert "YAYINDA" in win.broadcast_label.text()

    win.stop_engine()
    assert not win.engine._running
    assert win.start_btn.isEnabled()
    assert not win.stop_btn.isEnabled()
    assert "MİKROFON KAPALI" in win.broadcast_label.text()


def test_ptt_button_disabled_in_continuous(win):
    """Sürekli Açık modunda PTT butonu devre dışı olmalı."""
    win.mode_combo.setCurrentText("Sürekli Açık")
    assert not win.ptt_btn.isEnabled()


def test_ptt_button_enabled_in_ptt_mode(win):
    """Bas-Konuş modunda PTT butonu etkin olmalı."""
    win.mode_combo.setCurrentText("Bas-Konuş")
    assert win.ptt_btn.isEnabled()
    win.mode_combo.setCurrentText("Sürekli Açık")


def test_device_combos_exist(win):
    """Mikrofon ve hoparlör combo'ları mevcut olmalı."""
    assert win.input_combo is not None
    assert win.output_combo is not None


def test_device_combos_have_default(win):
    """Her cihaz combo'su en az 'Sistem Varsayılanı' içermeli."""
    assert win.input_combo.count() >= 1
    assert win.output_combo.count() >= 1


def test_mic_meter_exists(win):
    """Canlı mikrofon seviye çubuğu mevcut olmalı."""
    assert win.mic_meter is not None


def test_profile_combo_exists(win):
    """Profil combo'su mevcut olmalı."""
    assert win.profile_combo is not None


def test_mute_button_toggles(win):
    """Mute düğmesi toggle edilebilmeli."""
    win.mute_btn.setChecked(True)
    assert win.engine._muted
    win.mute_btn.setChecked(False)
    assert not win.engine._muted


def test_log_area_exists(win):
    """Log alanı mevcut olmalı."""
    assert win.log_view is not None


def test_record_timer_starts_at_zero(win):
    """Kayıt kronometresi başlangıçta sıfır göstermeli."""
    assert win.record_timer.text() == "00:00:00"


# ─── Event handler ses seviyesi log yazmıyor ─────────────────────────────────

def test_audio_level_event_does_not_write_log(win):
    """audio_level olayı log alanına satır yazmamalı."""
    initial_count = win.log_view.document().blockCount()
    win.bridge.event.emit("audio_level", {"input": 50, "clipping": False})
    assert win.log_view.document().blockCount() == initial_count


def test_metrics_event_does_not_write_log(win):
    """metrics olayı log alanına satır yazmamalı."""
    initial_count = win.log_view.document().blockCount()
    win.bridge.event.emit("metrics", {"sent": 10, "received": 5})
    assert win.log_view.document().blockCount() == initial_count


# ─── Ayarlar penceresi ────────────────────────────────────────────────────────

def test_settings_dialog_creates(qt_app, config_path):
    """SettingsDialog hata vermeden oluşturulabilmeli."""
    from paket5.ui_settings import SettingsDialog
    dlg = SettingsDialog(config_path)
    assert dlg is not None
    dlg.close()


def test_settings_dialog_tabs(qt_app, config_path):
    """Ayarlar penceresinde 7 sekme bulunmalı."""
    from paket5.ui_settings import SettingsDialog
    dlg = SettingsDialog(config_path)
    assert dlg.tabs.count() == 7
    dlg.close()


def test_settings_default_denoise_dusuk(qt_app, config_path):
    """Ayarlar penceresinde varsayılan denoise 'Düşük' olmalı."""
    from paket5.ui_settings import SettingsDialog
    dlg = SettingsDialog(config_path)
    assert dlg.s_denoise.currentText() == "Düşük"
    dlg.close()


def test_settings_restore_defaults(qt_app, config_path):
    """Varsayılanlara Dön'de denoise 'Düşük', mod 'Sürekli Açık' olmalı."""
    from paket5.ui_settings import SettingsDialog
    dlg = SettingsDialog(config_path)
    dlg.s_denoise.setCurrentText("Yüksek")
    dlg._restore_defaults()
    assert dlg.s_denoise.currentText() == "Düşük"
    assert dlg.s_ptt_mode.currentText() == "Sürekli Açık"
    dlg.close()


# ─── Pencere kapanışı ─────────────────────────────────────────────────────────

def test_close_event_calls_engine_stop(qt_app, config_path):
    """Pencere kapanırken engine.stop() çağrılmalı."""
    from paket5.ui_main import MainWindow
    from PyQt5.QtGui import QCloseEvent
    win = MainWindow(config_path)
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()
