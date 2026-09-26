import argparse
import pathlib
import sys
import logging
from logging.handlers import RotatingFileHandler
import shutil

_ROOT = pathlib.Path(__file__).resolve().parent
_BUNDLE_ROOT = pathlib.Path(getattr(sys, "_MEIPASS", _ROOT))
_APP_DIR = pathlib.Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5 import QtCore, QtWidgets
from paket5.ui_main import MainWindow


def main():
    parser = argparse.ArgumentParser(description="SimClever Intercom — Paket 5")
    parser.add_argument("--role", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = pathlib.Path(args.config) if args.config else _APP_DIR / "config" / f"{args.role}.json"
    if not config.is_file() and not args.config:
        bundled = _BUNDLE_ROOT / "config" / f"{args.role}.json"
        if bundled.is_file():
            config.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(bundled, config)
    if not config.is_file():
        parser.error(f"Config dosyası bulunamadı: {config}")
    log_dir = _APP_DIR / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "paket5.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = QtWidgets.QApplication(sys.argv); app.setApplicationName("SimClever Intercom Paket 5")
    lock = QtCore.QLockFile(str(pathlib.Path.home() / f".simclever_intercom_paket5_{args.role}.lock"))
    lock.setStaleLockTime(10000)
    if not lock.tryLock(100):
        QtWidgets.QMessageBox.warning(None, "Uygulama Açık", "SimClever Intercom zaten çalışıyor."); return 2
    window = MainWindow(str(config))
    if not args.config and window.engine.config.get("role") != args.role:
        window.engine.config["role"] = args.role
        window.role_combo.setCurrentText("Hoca Paneli" if args.role == "teacher" else "Öğrenci Düğümü")
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
