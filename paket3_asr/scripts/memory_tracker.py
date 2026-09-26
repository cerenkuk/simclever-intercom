"""
Paket 3 - Process Tree Peak RAM Measurement Module (psutil)
SimClever Intercom - 100 ms RSS Sampling & Delta Tracking
"""

import time
import threading
import psutil
from contextlib import contextmanager

class ProcessTreeMemoryTracker:
    """
    Windows ve diğer işletim sistemlerinde psutil kullanarak mevcut Python işleminin ve
    varsa yaşayan tüm alt süreçlerinin (child processes) toplam RSS (Resident Set Size)
    belleğini yaklaşık 100 ms aralıklarla örnekleyen güvenilir tepe RAM ölçücüdür.
    CTranslate2 ve PyTorch yerel (native) bellek tahsislerini de RSS üzerinden yakalar.
    """
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.running = False
        self.thread = None
        self.initial_ram_bytes = 0
        self.peak_ram_bytes = 0
        self._lock = threading.Lock()
        try:
            self.main_process = psutil.Process()
        except Exception:
            self.main_process = None

    def get_current_rss_bytes(self) -> int:
        """Ana işlem ve yaşayan alt süreçlerin toplam RSS baytını döndürür."""
        if self.main_process is None:
            return 0

        total_rss = 0
        try:
            total_rss += self.main_process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        try:
            children = self.main_process.children(recursive=True)
            for child in children:
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return total_rss

    def start(self):
        """Örnekleyiciyi arka plan daemon thread olarak başlatır."""
        self.initial_ram_bytes = self.get_current_rss_bytes()
        self.peak_ram_bytes = self.initial_ram_bytes
        self.running = True

        def _sample_loop():
            while self.running:
                try:
                    current_rss = self.get_current_rss_bytes()
                    with self._lock:
                        if current_rss > self.peak_ram_bytes:
                            self.peak_ram_bytes = current_rss
                except Exception:
                    pass
                time.sleep(self.interval)

        self.thread = threading.Thread(target=_sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        """
        Örnekleyiciyi güvenli biçimde durdurur ve tepe RAM ile artış (delta) değerlerini
        MB cinsinden döndürür.
        """
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Son bir kez daha kontrol et
        final_rss = self.get_current_rss_bytes()
        with self._lock:
            if final_rss > self.peak_ram_bytes:
                self.peak_ram_bytes = final_rss

        initial_mb = round(self.initial_ram_bytes / (1024 * 1024), 2)
        peak_mb = round(self.peak_ram_bytes / (1024 * 1024), 2)
        delta_mb = round(max(0.0, peak_mb - initial_mb), 2)

        return {
            "initial_ram_mb": initial_mb,
            "peak_ram_mb": peak_mb,
            "ram_delta_mb": delta_mb
        }

@contextmanager
def track_process_memory(interval: float = 0.1):
    """
    try/finally garantisi ile tepe RAM ölçümü yapan context manager.
    Hata durumunda da thread'in güvenli kapanmasını garanti eder.
    """
    tracker = ProcessTreeMemoryTracker(interval=interval)
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
