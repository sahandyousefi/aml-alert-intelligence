"""Tests for the account key rules in data/load.py."""

import pytest

from aml_alert_intelligence.data.load import account_key, canonical_bank


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("010", "10"),        # transactions pad bank IDs with zeros
        ("0010", "10"),       # the same bank, padded differently
        ("001", "1"),
        ("21884", "21884"),   # accounts.csv style, already unpadded
        ("000", "0"),         # all zeros keeps one digit
    ],
)
def test_canonical_bank_strips_padding(spark, raw, expected):
    df = spark.createDataFrame([(raw,)], ["bank"])
    assert df.select(canonical_bank("bank")).first()[0] == expected


def test_account_key_joins_bank_and_account(spark):
    df = spark.createDataFrame([("10", "8000EBD30")], ["bank_id", "account"])
    assert df.select(account_key("bank_id", "account")).first()[0] == "10_8000EBD30"


def test_padding_variants_give_one_key(spark):
    """010 and 0010 are one bank, so the same account must get exactly one key."""
    df = spark.createDataFrame([("010", "8000EBD30"), ("0010", "8000EBD30")],
                               ["bank", "account"])
    keys = df.withColumn("bank_id", canonical_bank("bank")).select(
        account_key("bank_id", "account").alias("key"))
    assert keys.distinct().count() == 1


def test_shared_account_number_at_two_banks_gives_two_keys(spark):
    """One of the 8 real collisions in HI-Small: same number, different banks."""
    df = spark.createDataFrame([("0027755", "80A7FDE00"), ("0028248", "80A7FDE00")],
                               ["bank", "account"])
    keys = df.withColumn("bank_id", canonical_bank("bank")).select(
        account_key("bank_id", "account").alias("key"))
    assert keys.distinct().count() == 2
