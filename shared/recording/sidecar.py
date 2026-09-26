import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SidecarWriter:
    """Writes a JSON side‑car file describing the current recording session.
    
    The side‑car is updated periodically by :class:`Recorder`.  It contains
    metadata such as the recording folder, timestamps and a status flag.
    """

    def __init__(self, base_folder: str):
        self.base_folder = Path(base_folder)
        self.sidecar_path = self.base_folder / "sidecar.json"
        self.base_folder.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self):
        initial = {"status": "initialized", "records": []}
        self._write(initial)
        logger.debug("Sidecar initialized at %s", self.sidecar_path)

    def update(self, recorder):
        """Update the side‑car with the latest recorder state.
        
        ``recorder`` is an instance of :class:`SessionRecorder`.  We record the
        absolute paths of the three WAV files if they exist.
        """
        data = {
            "status": "recording",
            "paths": {
                "raw_mic": str(recorder.raw_mic_path) if hasattr(recorder, "raw_mic_path") else None,
                "reference_spk": str(recorder.reference_spk_path) if hasattr(recorder, "reference_spk_path") else None,
                "processed_mic": str(recorder.processed_mic_path) if hasattr(recorder, "processed_mic_path") else None,
            },
        }
        self._write(data)
        logger.debug("Sidecar updated with recorder state")

    def finalize(self):
        data = {"status": "finished"}
        self._write(data)
        logger.debug("Sidecar finalized at %s", self.sidecar_path)

    def _write(self, data: dict):
        try:
            with self.sidecar_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.exception("Failed to write sidecar file: %s", e)
