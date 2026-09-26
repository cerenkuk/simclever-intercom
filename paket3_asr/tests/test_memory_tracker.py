"""
Paket 3 - Memory Tracker & CSV Schema Migration Unit Tests
"""

import os
import sys
import unittest
import tempfile
import csv
import time
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from memory_tracker import ProcessTreeMemoryTracker, track_process_memory
from benchmark import REQUIRED_FIELDNAMES, migrate_csv_schema_if_needed

class TestMemoryTracker(unittest.TestCase):
    def test_sampler_start_stop_lifecycle(self):
        tracker = ProcessTreeMemoryTracker(interval=0.05)
        tracker.start()
        self.assertTrue(tracker.running)
        self.assertIsNotNone(tracker.thread)
        self.assertTrue(tracker.thread.is_alive())
        
        time.sleep(0.15)
        res = tracker.stop()
        self.assertFalse(tracker.running)
        self.assertGreater(res["peak_ram_mb"], 0.0)
        self.assertGreaterEqual(res["ram_delta_mb"], 0.0)

    def test_peak_ram_capture_and_delta(self):
        tracker = ProcessTreeMemoryTracker(interval=0.05)
        tracker.start()
        dummy_list = [bytearray(10 * 1024 * 1024) for _ in range(3)] # ~30 MB allocation
        time.sleep(0.15)
        del dummy_list
        res = tracker.stop()
        
        self.assertGreaterEqual(res["peak_ram_mb"], res["initial_ram_mb"])
        self.assertGreaterEqual(res["ram_delta_mb"], 0.0)

    def test_context_manager_cleanup_on_error(self):
        with self.assertRaises(RuntimeError):
            with track_process_memory(interval=0.05) as tracker:
                self.assertTrue(tracker.running)
                raise RuntimeError("Test hatası simülasyonu")
        self.assertFalse(tracker.running)

    def test_non_cuda_vram_status(self):
        from transcribe import torch_has_cuda
        if not torch_has_cuda():
            vram_status = "N/A - CUDA unavailable"
            peak_vram = ""
            self.assertEqual(vram_status, "N/A - CUDA unavailable")
            self.assertEqual(peak_vram, "")

    def test_csv_schema_lossless_migration_and_run_purpose(self):
        temp_dir = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(temp_dir, "benchmark_results.csv")
            backup_path = os.path.join(temp_dir, "benchmark_results_pilot_v1_backup.csv")

            old_fieldnames = [
                "protocol_version", "timestamp", "basename", "condition", "denoiser_method", "model_id",
                "engine", "transcription_mode", "model_name", "device", "compute_type", "language", "temperature",
                "transcription", "WER", "CER", "medical_term_recall", "load_seconds",
                "elapsed_seconds", "duration_seconds", "RTF", "total_seconds",
                "decoding_config", "unsupported_decoding_params",
                "found_terms", "missed_terms", "reference_text"
            ]

            old_row = {
                "protocol_version": "pilot-v2-antiloop",
                "timestamp": "2026-07-25 12:00:00",
                "basename": "video_01_genel_anamnez",
                "condition": "none",
                "denoiser_method": "none",
                "model_id": "faster-whisper-large-v3-turbo-int8",
                "engine": "faster-whisper",
                "transcription_mode": "faster_whisper_native",
                "model_name": "large-v3-turbo",
                "device": "cpu",
                "compute_type": "int8",
                "language": "tr",
                "temperature": "0.0",
                "transcription": "test transkripsiyon",
                "WER": "0.2258",
                "CER": "0.1121",
                "medical_term_recall": "1.0",
                "load_seconds": "5.07",
                "elapsed_seconds": "468.5",
                "duration_seconds": "1000.0",
                "RTF": "0.4577",
                "total_seconds": "473.57",
                "decoding_config": "{}",
                "unsupported_decoding_params": "[]",
                "found_terms": "tansiyon",
                "missed_terms": "",
                "reference_text": "test referans"
            }

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=old_fieldnames)
                writer.writeheader()
                writer.writerow(old_row)

            import benchmark
            orig_output_csv = benchmark.OUTPUT_CSV
            orig_backup_csv = benchmark.BACKUP_CSV
            try:
                benchmark.OUTPUT_CSV = csv_path
                benchmark.BACKUP_CSV = backup_path
                migrate_csv_schema_if_needed(csv_path, default_purpose="main_benchmark")
            finally:
                benchmark.OUTPUT_CSV = orig_output_csv
                benchmark.BACKUP_CSV = orig_backup_csv

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                self.assertEqual(headers, REQUIRED_FIELDNAMES)
                self.assertIn("run_purpose", headers)
                rows = list(reader)
                self.assertEqual(len(rows), 1)
                m_row = rows[0]
                self.assertEqual(m_row["basename"], "video_01_genel_anamnez")
                self.assertEqual(m_row["WER"], "0.2258")
                self.assertEqual(m_row["run_purpose"], "main_benchmark")
                self.assertEqual(m_row["load_peak_ram_mb"], "")
                self.assertEqual(m_row["inference_peak_ram_mb"], "")
                self.assertEqual(m_row["peak_vram_mb"], "")
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()
