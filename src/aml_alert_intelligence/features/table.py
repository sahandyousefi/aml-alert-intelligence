"""The feature table: every registry feature for every transaction, built once and saved.

Building all features takes about 20 minutes on free Colab, so every later step reads the
saved table instead. The split is added at read time, as for the transactions, so the
table does not depend on the split dates. Reading fails if the table no longer matches
the registry.
"""

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from aml_alert_intelligence.data import fx
from aml_alert_intelligence.data.accounts import read_account_attributes
from aml_alert_intelligence.data.load import read_transactions
from aml_alert_intelligence.data.split import add_split
from aml_alert_intelligence.features.build import KEEP, transaction_features
from aml_alert_intelligence.features.registry import FEATURES

SPLIT_COLUMNS = ("split", "is_tail")


def stored_columns() -> list[str]:
    """Columns of the saved table: the KEEP columns except the split, then every feature."""
    return [c for c in KEEP if c not in SPLIT_COLUMNS] + [f.name for f in FEATURES]


def feature_table_path(cfg: dict) -> Path:
    """Processed feature table for the configured dataset variant."""
    variant = cfg["data"]["variant"]
    return cfg["paths"]["processed_data"] / f"{variant}_features.parquet"


def build_feature_table(spark: SparkSession, cfg: dict) -> Path:
    """Build every registry feature for every transaction and save the table as Parquet."""
    out = feature_table_path(cfg)
    table = transaction_features(read_transactions(spark, cfg),
                                 read_account_attributes(spark, cfg), fx.load_rates(cfg))
    table.select(*stored_columns()).write.mode("overwrite").parquet(str(out))
    return out


def check_columns(df: DataFrame) -> DataFrame:
    """Fail when a saved table and the registry disagree, so a stale table is never used."""
    expected = stored_columns()
    if df.columns != expected:
        missing = sorted(set(expected) - set(df.columns))
        extra = sorted(set(df.columns) - set(expected))
        raise ValueError("feature table does not match the registry; rebuild it "
                         f"(missing {missing}, unexpected {extra})")
    return df


def read_feature_table(spark: SparkSession, cfg: dict) -> DataFrame:
    """Read the saved feature table and add the split and tail columns."""
    df = check_columns(spark.read.parquet(str(feature_table_path(cfg))))
    return add_split(df, cfg)
