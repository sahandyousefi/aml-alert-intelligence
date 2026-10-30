"""Tests for exchange rates in data/fx.py."""

import pytest

from aml_alert_intelligence.data import fx

# Rates in the form the simulator uses: units per US dollar.
RATES = {"US Dollar": 1.0, "Euro": 0.8534, "Yen": 105.4, "Bitcoin": 1 / 11882}

COLUMNS = ["amount_paid", "payment_currency", "amount_received", "receiving_currency"]


def make_transfers(spark, rows):
    return spark.createDataFrame(rows, COLUMNS)


def test_fit_rates_pools_both_directions(spark):
    # 1,000 dollars buy 853.40 euros and 426.70 euros buy 500 dollars: both give 0.8534.
    df = make_transfers(spark, [(1000.0, "US Dollar", 853.40, "Euro"),
                                (426.70, "Euro", 500.0, "US Dollar"),
                                (50.0, "US Dollar", 50.0, "US Dollar")])
    table = fx.fit_rates(df).set_index("currency")
    assert table.loc["Euro", "units_per_usd"] == pytest.approx(0.8534, rel=1e-9)
    assert table.loc["Euro", "transfers"] == 2
    assert table.loc["US Dollar", "units_per_usd"] == 1.0


def test_fit_rates_needs_a_dollar_link(spark):
    # Yen trades only against the euro here, so it has no rate to the dollar.
    df = make_transfers(spark, [(1000.0, "US Dollar", 853.40, "Euro"),
                                (100.0, "Euro", 12350.80, "Yen")])
    with pytest.raises(ValueError, match="Yen"):
        fx.fit_rates(df)


@pytest.mark.parametrize(
    "amount, currency, expected",
    [
        (12.34, "US Dollar", True),
        (12.345, "US Dollar", False),
        (318388.29, "Yen", True),
        (0.000231, "Bitcoin", True),       # Bitcoin is recorded to six decimals
        (0.0000005, "Bitcoin", False),
    ],
)
def test_fits_precision(spark, amount, currency, expected):
    df = spark.createDataFrame([(amount, currency)], ["amount", "currency"])
    assert df.select(fx.fits_precision("amount", "currency")).first()[0] is expected


@pytest.mark.parametrize(
    "paid, pay_cur, received, recv_cur, expected",
    [
        (100.0, "US Dollar", 85.34, "Euro", True),
        (100.0, "US Dollar", 86.00, "Euro", False),
        (0.254231, "Bitcoin", 318388.29, "Yen", True),
        (0.000001, "Bitcoin", 0.65, "Yen", True),     # inside six-decimal rounding
        (0.000010, "Bitcoin", 0.65, "Yen", False),
        (25.00, "Euro", 25.05, "Euro", False),        # same currency, no rate allowance
    ],
)
def test_check_transfers(spark, paid, pay_cur, received, recv_cur, expected):
    df = make_transfers(spark, [(paid, pay_cur, received, recv_cur)])
    assert fx.check_transfers(df, RATES).first()["fx_consistent"] is expected


def test_add_usd_amounts(spark):
    df = make_transfers(spark, [(85.34, "Euro", 10540.0, "Yen")])
    row = fx.add_usd_amounts(df, RATES).first()
    assert row["amount_paid_usd"] == pytest.approx(100.0)
    assert row["amount_received_usd"] == pytest.approx(100.0)


def test_unknown_currency_fails_loudly(spark):
    df = make_transfers(spark, [(10.0, "US Dollar", 10.0, "Peso")])
    with pytest.raises(Exception, match="no exchange rate for currency: Peso"):
        fx.add_usd_amounts(df, RATES).collect()


def test_saved_rates_load_exactly(spark, tmp_path):
    cfg = {"data": {"variant": "HI-Small"}, "paths": {"processed_data": tmp_path}}
    df = make_transfers(spark, [(1000.0, "US Dollar", 853.40, "Euro"),
                                (3.0, "US Dollar", 316.20, "Yen")])
    table = fx.fit_rates(df)
    fx.save_rates(table, cfg)
    assert fx.load_rates(cfg) == fx.as_rates(table)
