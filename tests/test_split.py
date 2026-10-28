"""Tests for the chronological split in data/split.py."""

import pytest
from pyspark.sql import functions as F

from aml_alert_intelligence.data.split import add_split

CFG = {"split": {"train_end": "2022-09-06", "val_end": "2022-09-08",
                 "tail_start": "2022-09-11"}}


@pytest.mark.parametrize(
    "ts, split, is_tail",
    [
        ("2022-09-01 00:00", "train", False),   # first transaction in the data
        ("2022-09-06 23:59", "train", False),   # last minute of train
        ("2022-09-07 00:00", "val", False),     # first minute of validation
        ("2022-09-08 23:59", "val", False),     # last minute of validation
        ("2022-09-09 00:00", "test", False),    # first minute of test
        ("2022-09-10 23:59", "test", False),    # last minute before the tail
        ("2022-09-11 00:00", "test", True),     # first minute of the tail
        ("2022-09-18 16:18", "test", True),     # last transaction in the data
    ],
)
def test_split_boundaries(spark, ts, split, is_tail):
    # Parse the text in the session time zone (UTC), as the real loader does.
    df = spark.createDataFrame([(ts,)], ["ts"]).select(
        F.to_timestamp("ts", "yyyy-MM-dd HH:mm").alias("timestamp"))
    row = add_split(df, CFG).first()
    assert (row["split"], row["is_tail"]) == (split, is_tail)


def test_rejects_split_dates_out_of_order():
    bad = {"split": {"train_end": "2022-09-08", "val_end": "2022-09-06",
                     "tail_start": "2022-09-11"}}
    # The date check runs before the data is touched, so no DataFrame is needed.
    with pytest.raises(ValueError):
        add_split(None, bad)
