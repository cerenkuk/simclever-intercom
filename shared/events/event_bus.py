import logging
import threading
from collections import defaultdict


class EventBus:
    """Instance-scoped, thread-safe event dispatcher."""

    def __init__(self):
        self._listeners = defaultdict(list)
        self._lock = threading.RLock()

    def on(self, event_name, callback):
        with self._lock:
            if callback not in self._listeners[event_name]:
                self._listeners[event_name].append(callback)
        return callback

    def off(self, event_name, callback):
        with self._lock:
            listeners = self._listeners.get(event_name, [])
            if callback in listeners:
                listeners.remove(callback)

    def emit(self, event_name, payload=None):
        data = payload or {}
        with self._lock:
            listeners = list(self._listeners.get(event_name, []))
            wildcard = list(self._listeners.get("*", []))
        for callback in listeners:
            try:
                callback(data)
            except Exception:
                logging.getLogger(__name__).exception("Event listener failed: %s", event_name)
        for callback in wildcard:
            try:
                callback(event_name, data)
            except Exception:
                logging.getLogger(__name__).exception("Wildcard listener failed: %s", event_name)
