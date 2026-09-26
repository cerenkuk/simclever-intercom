import json

import numpy as np

from shared.intercom_engine import IntercomEngine


def make_engine(input_gain=75, agc=True):
    engine = IntercomEngine.__new__(IntercomEngine)
    engine.config = {"audio": {"input_gain": input_gain, "agc": agc}}
    engine._agc_preference_gain = input_gain / 100.0
    engine._agc_gain = engine._agc_preference_gain
    return engine


def test_agc_runtime_gain_does_not_mutate_config():
    engine = make_engine(input_gain=75)
    raw = np.full(160, 0.01, dtype=np.float32)

    for _ in range(100):
        engine._calculate_input_gain(raw)

    assert engine._agc_gain != 0.75
    assert engine.config["audio"]["input_gain"] == 75


def test_saved_config_keeps_user_gain(tmp_path):
    engine = make_engine(input_gain=75)
    engine.config_path = tmp_path / "teacher.json"
    raw = np.full(160, 0.01, dtype=np.float32)

    for _ in range(100):
        engine._calculate_input_gain(raw)
    engine.save_config()

    saved = json.loads(engine.config_path.read_text(encoding="utf-8"))
    assert saved["audio"]["input_gain"] == 75


def test_disabling_agc_uses_user_preference():
    engine = make_engine(input_gain=80)
    engine._agc_gain = 1.4
    engine.config["audio"]["agc"] = False

    gain = engine._calculate_input_gain(np.full(160, 0.01, dtype=np.float32))

    assert gain == 0.8
    assert engine._agc_gain == 0.8
