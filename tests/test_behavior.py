"""Tests for the behavior features (family C) in features/behavior.py."""

import pytest
from pyspark.sql import functions as F

from aml_alert_intelligence.features.behavior import THREE_DAYS, behavior_columns
from aml_alert_intelligence.features.registry import by_name

COLUMNS = ["tx_id", "timestamp", "from_key", "to_key", "amount_paid_usd", "amount_received_usd"]


def make_tx(spark, rows):
    """Rows of (tx_id, "yyyy-MM-dd HH:mm", sender, receiver, US dollars)."""
    return (spark.createDataFrame([(t, ts, a, b, usd, usd) for t, ts, a, b, usd in rows], COLUMNS)
                 .withColumn("timestamp", F.to_timestamp("timestamp", "yyyy-MM-dd HH:mm")))


def history(spark, rows):
    return {r["tx_id"]: r for r in behavior_columns(make_tx(spark, rows)).collect()}


def test_same_minute_is_excluded(spark):
    h = history(spark, [("t1", "2022-09-01 10:00", "A", "B", 10.0),
                        ("t2", "2022-09-01 10:00", "A", "C", 10.0),
                        ("t3", "2022-09-01 10:01", "A", "D", 10.0)])
    assert (h["t1"]["from_out_24h"], h["t2"]["from_out_24h"], h["t3"]["from_out_24h"]) == (0, 0, 2)


def test_window_edges(spark):
    h = history(spark, [("t1", "2022-09-01 10:00", "A", "B", 10.0),
                        ("t2", "2022-09-02 10:00", "A", "C", 10.0),     # exactly 24 hours later
                        ("t3", "2022-09-02 10:01", "A", "D", 10.0)])    # 24 hours and 1 minute
    assert h["t2"]["from_out_24h"] == 1
    assert (h["t3"]["from_out_24h"], h["t3"]["from_out_72h"]) == (1, 2)
    assert (h["t2"]["from_entries_1h"], h["t3"]["from_entries_1h"]) == (0, 1)


def test_self_transfers_get_features_but_are_not_history(spark):
    h = history(spark, [("t1", "2022-09-01 09:00", "A", "A", 10.0),
                        ("t2", "2022-09-01 10:00", "A", "B", 10.0)])
    assert "t1" in h
    assert (h["t2"]["from_out_24h"], h["t2"]["from_in_24h"]) == (0, 0)
    assert h["t2"]["from_minutes_since_last"] == THREE_DAYS


def test_new_payees_count_each_counterparty_once(spark):
    h = history(spark, [("t1", "2022-09-01 10:00", "A", "B", 10.0),
                        ("t2", "2022-09-01 12:00", "A", "B", 10.0),     # B again: not new
                        ("t3", "2022-09-01 13:00", "A", "C", 10.0),
                        ("t4", "2022-09-01 14:00", "A", "E", 10.0),
                        ("t5", "2022-09-05 12:00", "A", "B", 10.0),     # over 72 hours later: new
                        ("t6", "2022-09-05 13:00", "A", "F", 10.0)])
    assert (h["t4"]["from_new_payees_24h"], h["t4"]["from_out_24h"]) == (2, 3)
    assert h["t6"]["from_new_payees_24h"] == 1
    assert h["t2"]["to_new_payers_24h"] == 1


def test_pair_and_reverse_history(spark):
    h = history(spark, [("t1", "2022-09-01 10:00", "A", "B", 10.0),
                        ("t2", "2022-09-01 11:00", "B", "A", 10.0),
                        ("t3", "2022-09-01 12:00", "A", "B", 10.0)])
    assert (h["t3"]["pair_prior_72h"], h["t3"]["reverse_prior_72h"]) == (1, 1)
    assert (h["t2"]["pair_prior_72h"], h["t2"]["reverse_prior_72h"]) == (0, 1)


def test_minutes_since_last_and_inflow(spark):
    h = history(spark, [("t1", "2022-09-01 10:00", "X", "A", 100.0),
                        ("t2", "2022-09-01 10:30", "A", "B", 50.0)])
    assert (h["t1"]["from_minutes_since_last"], h["t2"]["from_minutes_since_last"]) == (THREE_DAYS, 30)
    assert (h["t2"]["from_in_24h"], h["t2"]["from_in_usd_24h"]) == (1, pytest.approx(100.0))


def test_sender_passthrough(spark):
    tx = make_tx(spark, [("t1", "2022-09-01 08:00", "X", "A", 100.0),
                         ("t2", "2022-09-01 10:00", "A", "B", 50.0),
                         ("t3", "2022-09-01 11:00", "A", "C", 300.0)])
    passthrough = by_name("sender_passthrough_24h").column()
    out = {r["tx_id"]: r["sender_passthrough_24h"] for r in
           tx.join(behavior_columns(tx), "tx_id").select("tx_id", passthrough).collect()}
    assert out["t1"] == 0.0
    assert out["t2"] == pytest.approx(1.0)
    assert out["t3"] == pytest.approx(100.0 / 300.0)


def test_future_transfers_do_not_change_past_features(spark):
    """Leakage guard: removing everything after a cutoff leaves earlier features unchanged."""
    rows = [("t1", "2022-09-01 10:00", "A", "B", 10.0),
            ("t2", "2022-09-01 18:00", "B", "C", 20.0),
            ("t3", "2022-09-02 09:00", "C", "A", 30.0),
            ("t4", "2022-09-02 09:00", "A", "B", 40.0),
            ("t5", "2022-09-03 07:00", "B", "A", 50.0),
            ("t6", "2022-09-03 08:00", "A", "D", 60.0)]
    full = history(spark, rows)
    early = history(spark, [r for r in rows if r[1] < "2022-09-02 12:00"])
    assert early and all(early[t].asDict() == full[t].asDict() for t in early)
