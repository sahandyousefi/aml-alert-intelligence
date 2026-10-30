"""Tests for the account key and transaction ID rules in data/load.py."""

import hashlib

import pytest
from pyspark.sql import functions as F

from aml_alert_intelligence.data.load import (
    TIMESTAMP_FORMAT,
    account_key,
    add_transaction_id,
    canonical_bank,
)


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


# Transaction ID

TX_COLUMNS = ["timestamp", "from_bank", "from_key", "to_bank", "to_key", "amount_received",
              "receiving_currency", "amount_paid", "payment_currency", "payment_format",
              "is_laundering"]

# Modeled on two of the duplicate pairs found in HI-Small (Step 2.2d).
BITCOIN_ROW = ("2022/09/01 16:20", "012004", "12004_800C927C1", "0220", "220_813D8C1E1",
               0.000008, "Bitcoin", 0.000008, "Bitcoin", "Bitcoin", 0)
ACH_ROW = ("2022/09/08 21:05", "0113779", "113779_811144AB1", "0113779", "113779_811144AB0",
           0.000002, "Bitcoin", 0.02, "US Dollar", "ACH", 0)


def make_tx(spark, rows):
    return (spark.createDataFrame(rows, TX_COLUMNS)
                 .withColumn("timestamp", F.to_timestamp("timestamp", TIMESTAMP_FORMAT)))


def ids(df):
    return [r["tx_id"] for r in add_transaction_id(df).orderBy("tx_id").collect()]


@pytest.mark.parametrize(
    "row, text",
    [
        (BITCOIN_ROW, "2022/09/01 16:20|12004_800C927C1|220_813D8C1E1|0.000008|Bitcoin"
                      "|0.000008|Bitcoin|Bitcoin|0"),
        (ACH_ROW, "2022/09/08 21:05|113779_811144AB1|113779_811144AB0|0.02|US Dollar"
                  "|0.000002|Bitcoin|ACH|0"),
    ],
)
def test_transaction_id_can_be_rebuilt_by_hand(spark, row, text):
    """Anyone with the transaction and hashlib gets the same ID."""
    expected = hashlib.sha256(text.encode()).hexdigest()[:16]
    assert ids(make_tx(spark, [row])) == [expected]


def test_exact_duplicates_get_distinct_ids(spark):
    out = add_transaction_id(make_tx(spark, [BITCOIN_ROW, BITCOIN_ROW])).collect()
    assert sorted(r["copy_number"] for r in out) == [0, 1]
    assert len({r["tx_id"] for r in out}) == 2


def test_label_does_not_change_id(spark):
    laundering = BITCOIN_ROW[:-1] + (1,)
    assert ids(make_tx(spark, [BITCOIN_ROW])) == ids(make_tx(spark, [laundering]))


def test_id_does_not_depend_on_row_order(spark):
    rows = [BITCOIN_ROW, ACH_ROW, BITCOIN_ROW]
    forward = make_tx(spark, rows)
    reshuffled = make_tx(spark, rows[::-1]).repartition(3)
    assert ids(forward) == ids(reshuffled)
