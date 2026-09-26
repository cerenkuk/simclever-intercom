from pathlib import Path

import pytest

from debrief.config import load_indexing_config


def test_load_indexing_config_converts_seconds_and_hashes_source(tmp_path: Path):
    path = tmp_path / "indexing.yaml"
    path.write_text(
        "version: '007'\nlead_in_seconds: 3\nwindow_seconds: 120\n"
        "overlap_seconds: 15\ntarget_event_interval_seconds:\n"
        "  minimum: 30\n  maximum: 90\nllm_temperature: 0\n",
        encoding="utf-8",
    )
    config = load_indexing_config(path)
    assert config.version == "007"
    assert config.lead_in_ms == 3_000
    assert config.window_ms == 120_000
    assert config.overlap_ms == 15_000
    assert len(config.fingerprint) == 64


def test_load_indexing_config_rejects_overlap_equal_to_window(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("window_seconds: 60\noverlap_seconds: 60\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap_seconds"):
        load_indexing_config(path)
