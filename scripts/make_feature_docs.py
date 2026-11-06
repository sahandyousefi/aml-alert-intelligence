"""Write docs/features.md from the feature registry.

Run from the repository root after changing the registry:
    python scripts/make_feature_docs.py
A unit test fails while the committed file and the registry disagree.
"""

from pathlib import Path

from aml_alert_intelligence.features.registry import dictionary_markdown

OUT = Path(__file__).resolve().parents[1] / "docs" / "features.md"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dictionary_markdown(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
