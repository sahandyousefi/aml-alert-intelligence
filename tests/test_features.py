"""Tests for the feature registry and the family A and B features."""

import math
from pathlib import Path

import pytest
from pyspark.sql import functions as F

from aml_alert_intelligence.features.build import KEEP, transaction_features
from aml_alert_intelligence.features.registry import (
    FAMILIES,
    FEATURES,
    KINDS,
    dictionary_markdown,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "features.md"

RATES = {"US Dollar": 1.0, "Euro": 0.8534, "Bitcoin": 1 / 11882}

TX_COLUMNS = ["tx_id", "timestamp", "from_key", "to_key", "from_bank_id", "to_bank_id",
              "amount_paid", "payment_currency", "amount_received", "receiving_currency",
              "payment_format", "is_laundering", "split", "is_tail"]

ATTR_COLUMNS = ["account_key", "bank_id", "entity_id", "entity_type", "bank_country",
                "owner_accounts", "owner_banks"]


def test_feature_names_are_unique():
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names))


def test_every_feature_has_a_known_family_and_kind():
    assert all(f.family in FAMILIES and f.kind in KINDS for f in FEATURES)


def test_no_feature_is_an_identifier_label_or_date():
    banned = set(KEEP) | {"entity_id", "bank_id", "from_entity_id", "to_entity_id"}
    assert not banned & {f.name for f in FEATURES}


def test_feature_docs_are_current():
    """docs/features.md must match the registry; rerun scripts/make_feature_docs.py if not."""
    committed = DOCS.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    assert committed == dictionary_markdown().strip()


@pytest.fixture
def features(spark):
    tx = spark.createDataFrame([
        # Saturday, cross-border between two accounts of one owner, paid in euros.
        ("t1", "2022-09-03 14:05", "1_A", "2_B", "1", "2", 85.34, "Euro", 85.34, "Euro",
         "ACH", 1, "train", False),
        # Wednesday, self-transfer at one US bank.
        ("t2", "2022-09-07 00:30", "3_C", "3_C", "3", "3", 50.0, "US Dollar", 50.0,
         "US Dollar", "Reinvestment", 0, "val", False),
    ], TX_COLUMNS).withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm"))
    attrs = spark.createDataFrame([
        ("1_A", "1", "E1", "Partnership", "Germany", 10, 2),
        ("2_B", "2", "E1", "Partnership", "France", 10, 2),
        ("3_C", "3", "E2", "Individual", "United States", 1, 1),
    ], ATTR_COLUMNS)
    return {r["tx_id"]: r for r in transaction_features(tx, attrs, RATES).collect()}


def test_transaction_features(features):
    t1, t2 = features["t1"], features["t2"]
    assert t1["amount_usd_log"] == pytest.approx(math.log10(101))
    assert (t1["hour_of_day"], t1["is_weekend"], t2["is_weekend"]) == (14, 1, 0)
    assert (t1["is_self_transfer"], t2["is_self_transfer"]) == (0, 1)
    assert (t1["is_same_bank"], t2["is_same_bank"]) == (0, 1)
    assert (t1["is_cross_currency"], t1["payment_format"]) == (0, "ACH")


def test_party_features(features):
    t1, t2 = features["t1"], features["t2"]
    assert (t1["is_cross_border"], t1["is_same_owner"]) == (1, 1)
    assert (t2["is_cross_border"], t2["is_same_owner"]) == (0, 1)
    assert t1["to_owner_accounts_log"] == pytest.approx(1.0)
    assert t2["from_owner_banks_log"] == pytest.approx(0.0)
    assert (t1["from_bank_country"], t1["to_bank_country"]) == ("Germany", "France")


def test_output_columns_are_keep_then_features(features):
    columns = list(next(iter(features.values())).asDict())
    assert columns == KEEP + [f.name for f in FEATURES]
