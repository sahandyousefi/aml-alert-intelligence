"""Exchange rates recovered from the simulator's own cross-currency transfers.

The simulator converts every currency at one fixed rate, so the rates come from the data
itself, not from market history. Rates are quoted as units of each currency per US dollar
and are fitted on training days only.
"""

from pathlib import Path

import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

BASE = "US Dollar"

# Decimal places each currency is recorded with. Bitcoin is the only one finer than cents.
DECIMALS = {"Bitcoin": 6}
DEFAULT_DECIMALS = 2

# Allowance for error in the fitted rates, on top of the rounding of each amount.
RATE_TOLERANCE = 5e-4


def _lookup(mapping: dict, currency_col: str) -> Column:
    """Value for each row's currency from a small dict, as a map literal (null if absent)."""
    pairs = [F.lit(x) for item in mapping.items() for x in item]
    return F.create_map(*pairs)[F.col(currency_col)]


def half_unit(currency_col: str) -> Column:
    """Largest rounding error of an amount: half of its currency's last decimal place."""
    special = {c: 0.5 * 10.0 ** -d for c, d in DECIMALS.items()}
    return F.coalesce(_lookup(special, currency_col), F.lit(0.5 * 10.0 ** -DEFAULT_DECIMALS))


def fits_precision(amount_col: str, currency_col: str) -> Column:
    """True when an amount has no more decimals than its currency is recorded with."""
    amount, currency = F.col(amount_col), F.col(currency_col)
    rounded = F.round(amount, DEFAULT_DECIMALS)
    for name, places in DECIMALS.items():
        rounded = F.when(currency == name, F.round(amount, places)).otherwise(rounded)
    return F.abs(amount - rounded) < 1e-9


def rate(rates: dict, currency_col: str) -> Column:
    """Units per US dollar for each row's currency. Fails on a currency with no rate."""
    missing = F.raise_error(F.concat(F.lit("no exchange rate for currency: "), F.col(currency_col)))
    return F.coalesce(_lookup(rates, currency_col), missing)


def fit_rates(train: DataFrame) -> pd.DataFrame:
    """Fit units per US dollar for every currency from its transfers with the dollar.

    Pass training-day rows only. Both directions are pooled and the rate is the ratio of
    summed amounts, so cent rounding on small transfers barely moves it. The transfers
    column counts the transfers behind each rate (none for the dollar itself).
    """
    pay, recv = F.col("payment_currency"), F.col("receiving_currency")
    from_usd = train.filter((pay == BASE) & (recv != BASE)).select(
        recv.alias("currency"), F.col("amount_received").alias("units"),
        F.col("amount_paid").alias("usd"))
    to_usd = train.filter((recv == BASE) & (pay != BASE)).select(
        pay.alias("currency"), F.col("amount_paid").alias("units"),
        F.col("amount_received").alias("usd"))
    fitted = (
        from_usd.unionByName(to_usd)
        .groupBy("currency")
        .agg((F.sum("units") / F.sum("usd")).alias("units_per_usd"),
             F.count("*").alias("transfers"))
        .toPandas()
    )
    base = pd.DataFrame({"currency": [BASE], "units_per_usd": [1.0], "transfers": [0]})
    table = pd.concat([base, fitted], ignore_index=True)

    # Every currency in the data needs a rate, or later conversions would fail.
    currencies = train.select(pay.alias("c")).union(train.select(recv.alias("c"))).distinct()
    missing = sorted({r["c"] for r in currencies.collect()} - set(table["currency"]))
    if missing:
        raise ValueError(f"no US Dollar transfers to fit a rate for: {missing}")

    table["usd_per_unit"] = 1.0 / table["units_per_usd"]
    columns = ["currency", "units_per_usd", "usd_per_unit", "transfers"]
    return table[columns].sort_values("currency", ignore_index=True)


def as_rates(table: pd.DataFrame) -> dict:
    """Rate table as {currency: units per US dollar} with plain Python floats."""
    return {str(c): float(r) for c, r in zip(table["currency"], table["units_per_usd"])}


def add_usd_amounts(df: DataFrame, rates: dict) -> DataFrame:
    """Add amount_paid_usd and amount_received_usd."""
    return (
        df.withColumn("amount_paid_usd", F.col("amount_paid") / rate(rates, "payment_currency"))
          .withColumn("amount_received_usd",
                      F.col("amount_received") / rate(rates, "receiving_currency"))
    )


def check_transfers(df: DataFrame, rates: dict,
                    rate_tolerance: float = RATE_TOLERANCE) -> DataFrame:
    """Flag transfers whose paid and received sides agree in US dollars.

    Each amount is exact only to half a unit of its last decimal, so the two sides may
    differ by both rounding errors in dollars. Cross-currency transfers also get a small
    allowance for the fitted rates. Adds usd_gap, usd_tolerance and fx_consistent.
    """
    paid_rate = rate(rates, "payment_currency")
    received_rate = rate(rates, "receiving_currency")
    paid_usd = F.col("amount_paid") / paid_rate
    received_usd = F.col("amount_received") / received_rate
    cross = F.col("payment_currency") != F.col("receiving_currency")
    tolerance = (half_unit("payment_currency") / paid_rate
                 + half_unit("receiving_currency") / received_rate
                 + F.when(cross, rate_tolerance * F.greatest(paid_usd, received_usd))
                    .otherwise(0.0))
    return (
        df.withColumn("usd_gap", F.abs(paid_usd - received_usd))
          .withColumn("usd_tolerance", tolerance)
          .withColumn("fx_consistent", F.col("usd_gap") <= F.col("usd_tolerance"))
    )


def rates_path(cfg: dict) -> Path:
    """Fitted rate table location for the configured dataset variant."""
    variant = cfg["data"]["variant"]
    return cfg["paths"]["processed_data"] / f"{variant}_fx_rates.csv"


def save_rates(table: pd.DataFrame, cfg: dict) -> Path:
    """Write the rate table through a .part file, so a broken write never looks complete."""
    out = rates_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    table.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def load_rates(cfg: dict) -> dict:
    """Read the saved rate table as {currency: units per US dollar}."""
    return as_rates(pd.read_csv(rates_path(cfg), float_precision="round_trip"))
