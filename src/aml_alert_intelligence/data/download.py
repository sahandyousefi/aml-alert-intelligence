"""Download raw AMLworld files from Kaggle into the configured raw data folder."""

import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from aml_alert_intelligence.config import RAW_FILES, raw_file


def download_raw(cfg: dict, force: bool = False) -> list[Path]:
    """Download the configured variant's raw files. Existing files are skipped unless force=True."""
    dataset = cfg["data"]["kaggle_dataset"]
    cfg["paths"]["raw_data"].mkdir(parents=True, exist_ok=True)
    paths = []

    for kind in RAW_FILES:
        target = raw_file(cfg, kind)
        paths.append(target)
        if target.exists() and not force:
            print(f"skip (exists): {target.name}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["kaggle", "datasets", "download", dataset,
                 "-f", target.name, "-p", tmp, "--force"],
                check=True,
            )
            zipped = Path(tmp) / f"{target.name}.zip"
            if zipped.exists():
                with zipfile.ZipFile(zipped) as zf:
                    zf.extractall(tmp)

            partial = target.with_name(target.name + ".part")
            shutil.copy(Path(tmp) / target.name, partial)
            partial.rename(target)
        print(f"downloaded: {target.name}")

    return paths
