"""Tests for config loading and the committed base config."""

from pathlib import Path

from aml_alert_intelligence.config import load_config, raw_file

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def test_paths_resolve_under_root(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "paths:\n"
        "  root: /data/aml\n"
        "  raw_data: data/raw\n"
        "data:\n"
        "  variant: HI-Small\n"
    )
    cfg = load_config(cfg_file)
    assert cfg["paths"]["root"] == Path("/data/aml")
    assert cfg["paths"]["raw_data"] == Path("/data/aml/data/raw")
    assert raw_file(cfg, "transactions") == Path("/data/aml/data/raw/HI-Small_Trans.csv")


def test_base_config_split_dates_are_ordered_strings():
    split = load_config(BASE_CONFIG)["split"]
    # Quoted in YAML, so they stay plain strings instead of Python dates.
    assert all(isinstance(v, str) for v in split.values())
    assert split["train_end"] < split["val_end"] < split["tail_start"]
