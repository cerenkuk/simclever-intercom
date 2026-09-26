import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock, patch

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_pkg_root, "scripts"))

from cloud_security import validate_cloud_upload_permission, CloudUploadDeniedError
from run_commercial_api import run_commercial_gemini_transcription, VERBATIM_SYSTEM_PROMPT

class TestCloudSecurityAndCommercialApiPolicy(unittest.TestCase):

    def setUp(self):
        self.allowed_videos = [
            "video_01_genel_anamnez",
            "video_02_psikiyatrik_anamnez",
            "video_03_pediatrik_anamnez",
            "video_04_simule_hasta_gorusmesi",
            "video_05_poliklinik_anamnez_fizik_muayene",
            "video_06_gogus_hastaliklari_oyku_alma"
        ]
        self.denied_saha = [
            "saha_01_img_3463",
            "saha_02_img_3492",
            "saha_03_img_3602",
            "saha_04_vid_20220913_142911"
        ]

    def test_utf8_system_prompt_encoding_is_clean(self):
        expected_turkish = "T\u00fcrk\u00e7e"
        expected_speaker = "Konu\u015fmac\u0131n\u0131n"
        broken_turkish = "T\u00c3\u00bcrk\u00c3\u00a7e"

        self.assertIn(expected_turkish, VERBATIM_SYSTEM_PROMPT)
        self.assertIn(expected_speaker, VERBATIM_SYSTEM_PROMPT)
        self.assertNotIn(broken_turkish, VERBATIM_SYSTEM_PROMPT)
    def test_six_allowed_video_basenames_pass_security(self):
        for video in self.allowed_videos:
            meta = {"source_group": "video", "scenario_group": "video"}
            audio_p = os.path.join(_pkg_root, "data", f"{video}.wav")
            res = validate_cloud_upload_permission(basename=video, audio_path=audio_p, metadata=meta)
            self.assertTrue(res)

    def test_four_saha_basenames_are_denied(self):
        for saha in self.denied_saha:
            meta = {"source_group": "saha", "scenario_group": "saha"}
            audio_p = os.path.join(_pkg_root, "data", f"{saha}.wav")
            with self.assertRaises(CloudUploadDeniedError) as ctx:
                validate_cloud_upload_permission(basename=saha, audio_path=audio_p, metadata=meta)
            self.assertIn("DENIED_PREFIX", str(ctx.exception))

    def test_new_saha_99_deneme_is_denied(self):
        meta = {"source_group": "saha"}
        with self.assertRaises(CloudUploadDeniedError) as ctx:
            validate_cloud_upload_permission(basename="saha_99_deneme", metadata=meta)
        self.assertIn("saha_", str(ctx.exception))

    def test_unknown_basename_not_in_allowlist_is_denied(self):
        meta = {"source_group": "video"}
        with self.assertRaises(CloudUploadDeniedError) as ctx:
            validate_cloud_upload_permission(basename="unknown_video_99", metadata=meta)
        self.assertIn("DENIED_NOT_IN_ALLOWLIST", str(ctx.exception))

    def test_missing_metadata_source_or_invalid_source_is_denied(self):
        with self.assertRaises(CloudUploadDeniedError):
            validate_cloud_upload_permission(basename="video_01_genel_anamnez", metadata={"source_group": "saha"})

    def test_saha_metadata_with_video_basename_is_denied(self):
        meta = {"source_group": "saha"}
        with self.assertRaises(CloudUploadDeniedError) as ctx:
            validate_cloud_upload_permission(basename="video_01_genel_anamnez", metadata=meta)
        self.assertIn("DENIED_METADATA_SOURCE", str(ctx.exception))

    def test_path_traversal_outside_allowed_audio_root_is_denied(self):
        outside_path = os.path.abspath(os.path.join(_pkg_root, "..", "secret_audio.wav"))
        meta = {"source_group": "video"}
        with self.assertRaises(CloudUploadDeniedError) as ctx:
            validate_cloud_upload_permission(basename="video_01_genel_anamnez", audio_path=outside_path, metadata=meta)
        self.assertIn("DENIED_PATH_TRAVERSAL", str(ctx.exception))

    def test_saha_denied_prevents_api_client_and_file_read_calls(self):
        mock_api_client = MagicMock()
        meta = {"source_group": "saha", "reference_text": "gizli hasta verisi"}
        saha_path = os.path.join(_pkg_root, "data", "saha_01_img_3463.wav")

        with self.assertRaises(CloudUploadDeniedError):
            run_commercial_gemini_transcription(
                basename="saha_01_img_3463",
                audio_path=saha_path,
                metadata=meta,
                api_client=mock_api_client
            )

        self.assertEqual(mock_api_client.generate_content.call_count, 0)
        self.assertEqual(mock_api_client.models.generate_content.call_count, 0)

    @patch("os.path.exists", return_value=True)
    def test_dry_run_zero_api_client_and_file_read(self, mock_exists):
        mock_api_client = MagicMock()
        meta = {"source_group": "video"}
        fake_audio_path = os.path.join(_pkg_root, "data", "video_01_genel_anamnez.wav")

        res = run_commercial_gemini_transcription(
            basename="video_01_genel_anamnez",
            audio_path=fake_audio_path,
            metadata=meta,
            api_client=mock_api_client,
            dry_run=True
        )

        self.assertEqual(res["status"], "dry_run_success")
        self.assertEqual(mock_api_client.generate_content.call_count, 0)
        self.assertEqual(mock_api_client.models.generate_content.call_count, 0)

    @patch("os.path.getsize", return_value=1024)
    @patch(
        "run_commercial_api.normalize_turkish_asr_v2",
        side_effect=lambda x: x
    )
    @patch(
        "run_commercial_api.calculate_wer_cer",
        return_value=(0.0, 0.0, 0, 0, 0)
    )
    def test_reference_text_not_included_in_api_payload(
        self,
        mock_calc,
        mock_norm,
        mock_size
    ):
        mock_api_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Doktor Fatih Kamış bugün sizlere"
        mock_api_client.models.generate_content.return_value = mock_response

        secret_reference = "GİZLİ REFERANS METNİ API'YE GİTMEMELİ"
        meta = {
            "source_group": "video",
            "duration_seconds": 30.0,
            "reference_text": secret_reference
        }

        data_dir = os.path.join(_pkg_root, "data")
        os.makedirs(data_dir, exist_ok=True)

        temp_audio = tempfile.NamedTemporaryFile(
            dir=data_dir,
            suffix=".wav",
            delete=False
        )

        try:
            temp_audio.write(b"fake_audio")
            temp_audio.close()

            with patch("run_commercial_api.save_commercial_api_result"):
                run_commercial_gemini_transcription(
                    basename="video_01_genel_anamnez",
                    audio_path=temp_audio.name,
                    metadata=meta,
                    api_client=mock_api_client
                )

            mock_api_client.models.generate_content.assert_called_once()
            call_kwargs = (
                mock_api_client.models.generate_content.call_args.kwargs
            )
            contents = call_kwargs["contents"]

            self.assertNotIn(secret_reference, str(contents))
            self.assertEqual(contents[0], VERBATIM_SYSTEM_PROMPT)
        finally:
            if os.path.exists(temp_audio.name):
                os.remove(temp_audio.name)


    @patch("os.path.getsize", return_value=1024)
    def test_empty_response_not_written_to_csv(self, mock_size):
        mock_api_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_api_client.models.generate_content.return_value = mock_response

        meta = {"source_group": "video"}

        data_dir = os.path.join(_pkg_root, "data")
        os.makedirs(data_dir, exist_ok=True)

        temp_audio = tempfile.NamedTemporaryFile(
            dir=data_dir,
            suffix=".wav",
            delete=False
        )

        try:
            temp_audio.write(b"fake_audio")
            temp_audio.close()

            with patch(
                "run_commercial_api.save_commercial_api_result"
            ) as mock_save:
                with self.assertRaises(ValueError) as ctx:
                    run_commercial_gemini_transcription(
                        basename="video_01_genel_anamnez",
                        audio_path=temp_audio.name,
                        metadata=meta,
                        api_client=mock_api_client
                    )

                self.assertIn("EMPTY_API_RESPONSE", str(ctx.exception))
                self.assertEqual(mock_save.call_count, 0)
        finally:
            if os.path.exists(temp_audio.name):
                os.remove(temp_audio.name)


if __name__ == "__main__":
    unittest.main()


