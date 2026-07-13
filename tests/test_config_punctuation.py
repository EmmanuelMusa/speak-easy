"""punctuation_source config field: default and load from TOML."""
from app.config import CleanupConfig, load_config


def test_default_is_model():
    assert CleanupConfig().punctuation_source == "model"


def test_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[cleanup]\npunctuation_source = "pauses"\n', encoding="utf-8"
    )
    assert load_config(cfg_file).cleanup.punctuation_source == "pauses"
