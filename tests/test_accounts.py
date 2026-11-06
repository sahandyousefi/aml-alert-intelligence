"""Tests for account attributes in data/accounts.py."""

import pytest

from aml_alert_intelligence.data.accounts import (
    account_attributes,
    add_party_attributes,
    bank_country,
    entity_type,
)

ACCOUNT_COLUMNS = ["bank_name", "bank_id", "account_key", "entity_id", "entity_name"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Germany Bank #4815", "Germany"),
        ("Saudi Arabia Bank #12", "Saudi Arabia"),     # country names can have spaces
        ("UK Bank #33", "UK"),
        ("Crytpo Bank #7", "Crypto"),                  # spelled so in the source
        ("Savings Bank of Omaha", "United States"),    # unnumbered names are US banks
        ("Acme Credit Union", "United States"),
        ("Mystery Trust #5", None),                    # unknown pattern stays visible
    ],
)
def test_bank_country(spark, name, expected):
    df = spark.createDataFrame([(name,)], ["bank_name"])
    assert df.select(bank_country()).first()[0] == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Sole Proprietorship #50438", "Sole Proprietorship"),
        ("Country #1", "Country"),
        ("Weird Name", None),
    ],
)
def test_entity_type(spark, name, expected):
    df = spark.createDataFrame([(name,)], ["entity_name"])
    assert df.select(entity_type()).first()[0] == expected


def test_owner_size_counts_accounts_and_banks(spark):
    accounts = spark.createDataFrame([
        ("Italy Bank #53", "12004", "12004_A1", "E1", "Partnership #1"),
        ("Italy Bank #53", "12004", "12004_A2", "E1", "Partnership #1"),
        ("UK Bank #33", "21884", "21884_A3", "E1", "Partnership #1"),
        ("Acme Bank", "10", "10_B1", "E2", "Corporation #2"),
    ], ACCOUNT_COLUMNS)
    rows = {r["account_key"]: r for r in account_attributes(accounts).collect()}
    assert (rows["12004_A1"]["owner_accounts"], rows["12004_A1"]["owner_banks"]) == (3, 2)
    assert (rows["10_B1"]["owner_accounts"], rows["10_B1"]["owner_banks"]) == (1, 1)
    assert rows["21884_A3"]["bank_country"] == "UK"


def test_party_attributes_flags_and_row_count(spark):
    attrs = spark.createDataFrame([
        ("12004_A1", "12004", "E1", "Partnership", "Italy", 3, 2),
        ("21884_A3", "21884", "E1", "Partnership", "UK", 3, 2),
        ("10_B1", "10", "E2", "Corporation", "United States", 1, 1),
    ], ["account_key", "bank_id", "entity_id", "entity_type", "bank_country",
        "owner_accounts", "owner_banks"])
    tx = spark.createDataFrame([
        ("t1", "12004_A1", "21884_A3"),     # one owner, two countries
        ("t2", "10_B1", "12004_A1"),        # two owners, two countries
        ("t3", "10_B1", "10_B1"),           # self-transfer
        ("t4", "10_B1", "99_UNKNOWN"),      # receiver missing from attrs
    ], ["tx_id", "from_key", "to_key"])
    out = {r["tx_id"]: r for r in add_party_attributes(tx, attrs).collect()}
    assert len(out) == 4
    assert (out["t1"]["cross_border"], out["t1"]["same_owner"]) == (True, True)
    assert (out["t2"]["cross_border"], out["t2"]["same_owner"]) == (True, False)
    assert (out["t3"]["cross_border"], out["t3"]["same_owner"]) == (False, True)
    assert out["t4"]["to_bank_country"] is None


def test_party_attributes_keep_transaction_columns_first(spark):
    attrs = spark.createDataFrame([("10_B1", "10", "E2", "Corporation", "United States", 1, 1)],
                                  ["account_key", "bank_id", "entity_id", "entity_type",
                                   "bank_country", "owner_accounts", "owner_banks"])
    tx = spark.createDataFrame([("t1", "10_B1", "10_B1")], ["tx_id", "from_key", "to_key"])
    assert add_party_attributes(tx, attrs).columns[:3] == ["tx_id", "from_key", "to_key"]
