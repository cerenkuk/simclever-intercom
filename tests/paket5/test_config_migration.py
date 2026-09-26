import json

from shared.events.event_bus import EventBus
from shared.intercom_engine import IntercomEngine


def load_config_without_audio(config_path):
    engine = IntercomEngine.__new__(IntercomEngine)
    engine.config_path = config_path
    engine.event_bus = EventBus()
    engine._config_migrated = False
    return engine, engine._load_config()


def test_bom_config_loads_without_corrupt_backup(tmp_path):
    config_path = tmp_path / "teacher.json"
    config_path.write_text(
        json.dumps({"schema": 2, "network": {"port": 4999}}),
        encoding="utf-8-sig",
    )

    _engine, config = load_config_without_audio(config_path)

    assert config["network"]["port"] == 4999
    assert not list(tmp_path.glob("teacher.corrupt-*.json"))


def test_legacy_numeric_devices_migrate_to_schema_2(tmp_path, caplog):
    config_path = tmp_path / "teacher.json"
    config_path.write_text(
        json.dumps(
            {
                "audio": {"input_device": 5, "output_device": 7},
                "network": {"port": 4999},
            }
        ),
        encoding="utf-8",
    )

    engine, config = load_config_without_audio(config_path)

    assert config["schema"] == 2
    assert config["audio"]["input_device"] is None
    assert config["audio"]["output_device"] is None
    assert config["network"]["port"] == 4999
    assert engine._config_migrated is True
    assert "select devices again" in caplog.text


def test_schema_2_device_names_are_preserved(tmp_path):
    config_path = tmp_path / "teacher.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "audio": {
                    "input_device": "USB Microphone",
                    "output_device": "USB Headphones",
                },
            }
        ),
        encoding="utf-8",
    )

    engine, config = load_config_without_audio(config_path)

    assert config["audio"]["input_device"] == "USB Microphone"
    assert config["audio"]["output_device"] == "USB Headphones"
    assert engine._config_migrated is False
