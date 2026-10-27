"""Load YAML configs and resolve storage paths."""

from pathlib import Path

import yaml

RAW_FILES = {
    "transactions": "{variant}_Trans.csv",
    "accounts": "{variant}_accounts.csv",
    "patterns": "{variant}_Patterns.txt",
}


def load_config(path: str | Path) -> dict:
    """Read a YAML config and turn storage paths into absolute Path objects."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["paths"]["root"])
    cfg["paths"] = {
        name: root if name == "root" else root / rel
        for name, rel in cfg["paths"].items()
    }
    return cfg


def raw_file(cfg: dict, kind: str) -> Path:
    """Return the raw data path for the configured variant, e.g. HI-Small_Trans.csv."""
    name = RAW_FILES[kind].format(variant=cfg["data"]["variant"])
    return cfg["paths"]["raw_data"] / name
