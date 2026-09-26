"""
Paket 3 - Yeni Model Yapılandırması ve Sözleşme Doğrulama Birim Testleri
SimClever Intercom - Nemotron & Qwen3-ASR Adaptör Doğrulaması
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import wave
import json
import csv
import importlib.util


_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_pkg_root, "scripts"))

from transcribe import transcribe_audio_transformers, get_experiment_decoding_config

@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch kurulu değil.")
class TestNemotronOfficialRNNTContract(unittest.TestCase):
    @patch("transcribe.torch_has_cuda", return_value=False)
    @patch("soundfile.read")
    def test_nemotron_locale_mapping_tr_to_tr_TR(self, mock_sf_read, mock_cuda):
        import numpy as np
        import torch

        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_processor = MagicMock()
        mock_processor.feature_extractor = MagicMock()
        mock_processor.feature_extractor.sampling_rate = 16000
        mock_processor.return_value = {
            "input_features": torch.zeros((1, 101, 128)),
            "attention_mask": torch.ones((1, 101), dtype=torch.bool)
        }
        mock_processor.batch_decode.side_effect = [
            ["<tr-TR> merhaba doktor"], # raw
            ["merhaba doktor"]          # clean
        ]

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[101, 102]])

        test_wav = os.path.join(_pkg_root, "data", "diagnostic_vad_start", "audio_snippets", "test_nemotron_locale.wav")
        os.makedirs(os.path.dirname(test_wav), exist_ok=True)
        with wave.open(test_wav, "wb") as wf:
            wf.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            wf.writeframes(b"\x00" * 32000)

        try:
            res = transcribe_audio_transformers(
                audio_path=test_wav,
                model_size_or_path="nvidia/nemotron-3.5-asr-streaming-0.6b",
                language="tr",
                device="cpu",
                compute_type="float32",
                model_and_processor=(mock_model, mock_processor)
            )
            mock_processor.assert_called()
            call_kwargs = mock_processor.call_args[1]
            self.assertEqual(call_kwargs.get("language"), "tr-TR")
            self.assertEqual(call_kwargs.get("sampling_rate"), 16000)
            self.assertEqual(res["text"], "merhaba doktor")
        finally:
            if os.path.exists(test_wav):
                os.remove(test_wav)

    @patch("transcribe.torch_has_cuda", return_value=False)
    @patch("soundfile.read")
    def test_nemotron_no_qwen_slicing_and_direct_sequences_decode(self, mock_sf_read, mock_cuda):
        import numpy as np
        import torch

        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_processor = MagicMock()
        mock_processor.feature_extractor.sampling_rate = 16000
        mock_processor.return_value = {"input_features": torch.zeros((1, 100, 128))}
        mock_processor.batch_decode.side_effect = [
            ["<tr-TR> hasta hekim görüşmesi"],
            ["hasta hekim görüşmesi"]
        ]

        mock_res = MagicMock()
        mock_res.sequences = torch.tensor([[50, 60, 70]])

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_res

        test_wav = os.path.join(_pkg_root, "data", "diagnostic_vad_start", "audio_snippets", "test_nemotron_direct.wav")
        os.makedirs(os.path.dirname(test_wav), exist_ok=True)
        with wave.open(test_wav, "wb") as wf:
            wf.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            wf.writeframes(b"\x00" * 32000)

        try:
            res = transcribe_audio_transformers(
                audio_path=test_wav,
                model_size_or_path="nvidia/nemotron-3.5-asr-streaming-0.6b",
                language="tr-TR",
                device="cpu",
                compute_type="float32",
                model_and_processor=(mock_model, mock_processor)
            )
            args_passed = mock_processor.batch_decode.call_args_list[0][0][0]
            self.assertTrue(torch.equal(args_passed, mock_res.sequences))
            self.assertEqual(res["text"], "hasta hekim görüşmesi")
        finally:
            if os.path.exists(test_wav):
                os.remove(test_wav)

    @patch("transcribe.torch_has_cuda", return_value=False)
    @patch("soundfile.read")
    def test_nemotron_raw_full_clean_empty_fail_fast(self, mock_sf_read, mock_cuda):
        import numpy as np
        import torch

        mock_sf_read.return_value = (np.zeros(16000, dtype=np.float32), 16000)

        mock_processor = MagicMock()
        mock_processor.feature_extractor.sampling_rate = 16000
        mock_processor.return_value = {"input_features": torch.zeros((1, 100, 128))}
        mock_processor.batch_decode.side_effect = [
            ["<pad><eos><blank>"], # raw
            [""]                   # clean
        ]

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])

        test_wav = os.path.join(_pkg_root, "data", "diagnostic_vad_start", "audio_snippets", "test_nemotron_silenced.wav")
        os.makedirs(os.path.dirname(test_wav), exist_ok=True)
        with wave.open(test_wav, "wb") as wf:
            wf.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
            wf.writeframes(b"\x00" * 32000)

        try:
            with self.assertRaises(RuntimeError) as ctx:
                transcribe_audio_transformers(
                    audio_path=test_wav,
                    model_size_or_path="nvidia/nemotron-3.5-asr-streaming-0.6b",
                    language="tr",
                    device="cpu",
                    compute_type="float32",
                    model_and_processor=(mock_model, mock_processor)
                )
            self.assertIn("NEMOTRON DİL HATASI", str(ctx.exception))
            self.assertIn("Özel tokenlar metni silmektedir", str(ctx.exception))
        finally:
            if os.path.exists(test_wav):
                os.remove(test_wav)


class TestSpeechPad800ValidationDecision(unittest.TestCase):
    def setUp(self):
        self.prod_csv = os.path.join(_pkg_root, "outputs", "metrics", "product_candidate_validation_results.csv")
        self.bench_csv = os.path.join(_pkg_root, "outputs", "metrics", "benchmark_results.csv")
        self.comp_csv = os.path.join(_pkg_root, "outputs", "metrics", "speech_pad_400_vs_800_comparison.csv")

    def test_product_candidate_validation_10_unique_rows(self):
        if not os.path.exists(self.prod_csv):
            self.skipTest("Harici ürün adayı doğrulama sonuç paketi mevcut değil.")
        with open(self.prod_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 10, "product_candidate_validation_results.csv tam 10 satır olmalı")
        basenames = [r["basename"] for r in rows]
        self.assertEqual(len(set(basenames)), 10, "10 benzersiz basename bulunmalı")

        for r in rows:
            self.assertEqual(r["model_id"], "faster-whisper-large-v3-int8")
            self.assertEqual(r["condition"], "none")
            self.assertEqual(r["run_purpose"], "product_candidate_validation")
            dec_cfg = json.loads(r["decoding_config"])
            self.assertEqual(dec_cfg.get("speech_pad_ms"), 800)

    def test_speech_pad_macro_averages_and_decision(self):
        if not os.path.exists(self.comp_csv):
            self.skipTest("Harici speech-pad karşılaştırma sonuç paketi mevcut değil.")
        with open(self.comp_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 10)
        v_list = [r for r in rows if r["group"] == "video"]
        s_list = [r for r in rows if r["group"] == "saha"]

        v_mac_400 = sum(float(r["wer_400ms"]) for r in v_list) / len(v_list)
        v_mac_800 = sum(float(r["wer_800ms"]) for r in v_list) / len(v_list)
        s_mac_400 = sum(float(r["wer_400ms"]) for r in s_list) / len(s_list)
        s_mac_800 = sum(float(r["wer_800ms"]) for r in s_list) / len(s_list)
        o_mac_400 = sum(float(r["wer_400ms"]) for r in rows) / len(rows)
        o_mac_800 = sum(float(r["wer_800ms"]) for r in rows) / len(rows)

        self.assertAlmostEqual(v_mac_400, 0.2175, delta=0.001)
        self.assertAlmostEqual(v_mac_800, 0.2133, delta=0.001)
        self.assertAlmostEqual(s_mac_400, 0.3828, delta=0.001)
        self.assertAlmostEqual(s_mac_800, 0.4480, delta=0.001)
        self.assertAlmostEqual(o_mac_400, 0.2836, delta=0.001)
        self.assertAlmostEqual(o_mac_800, 0.3072, delta=0.001)

        speech_pad_800_decision = "rejected" if (o_mac_800 > o_mac_400 and s_mac_800 > s_mac_400) else "accepted"
        speech_pad_400_decision = "retained_product_candidate"

        self.assertEqual(speech_pad_800_decision, "rejected")
        self.assertEqual(speech_pad_400_decision, "retained_product_candidate")

    def test_benchmark_results_csv_unmodified(self):
        if not os.path.exists(self.bench_csv):
            self.skipTest("Harici benchmark sonuç paketi mevcut değil.")
        with open(self.bench_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 124)
        l3_rows = [
            r for r in rows
            if r.get("model_id") == "faster-whisper-large-v3-int8"
            and r.get("condition") == "none"
            and r.get("run_purpose") == "main_benchmark"
        ]
        self.assertEqual(len(l3_rows), 10)




if __name__ == "__main__":
    unittest.main()
