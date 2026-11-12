"""Behavior features (family C): what each party did before a transfer.

Every transfer becomes two ledger entries, an outgoing one for the sender and an incoming
one for the receiver. A transfer's features summarize each party's entries in fixed windows
that end one minute before it: timestamps are recorded to the minute, so entries in the
same minute cannot be ordered and are left out. Self-transfers keep their entries, so they
get features too, but they never count as history for anyone.
"""

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

HOUR, DAY, THREE_DAYS = 60, 1440, 4320     # window lengths in minutes

# Per-party columns produced for each side ("from_" sender, "to_" receiver).
ACCOUNT_COLUMNS = [
    "entries_1h",
    "in_24h", "out_24h", "in_usd_24h", "out_usd_24h",
    "in_72h", "out_72h", "in_usd_72h", "out_usd_72h",
    "new_payers_24h", "new_payees_24h",
    "minutes_since_last",
]


def _earlier(partition: list[str], minutes: int) -> Window:
    """Entries of the same partition from `minutes` before up to one minute before."""
    return Window.partitionBy(*partition).orderBy("minute").rangeBetween(-minutes, -1)


def _total(col: Column, window: Window) -> Column:
    return F.coalesce(F.sum(col).over(window), F.lit(0))


def ledger(tx: DataFrame) -> DataFrame:
    """Two entries per transfer; w_out and w_in mark what counts as history (0 for self-transfers)."""
    minute = (F.unix_timestamp("timestamp") / 60).cast("long")
    counts = (F.col("from_key") != F.col("to_key")).cast("int")
    outgoing = tx.select("tx_id", F.lit("from").alias("side"),
                         F.col("from_key").alias("account"), F.col("to_key").alias("counterparty"),
                         minute.alias("minute"), counts.alias("w_out"), F.lit(0).alias("w_in"),
                         F.col("amount_paid_usd").alias("usd"))
    incoming = tx.select("tx_id", F.lit("to").alias("side"),
                         F.col("to_key").alias("account"), F.col("from_key").alias("counterparty"),
                         minute.alias("minute"), F.lit(0).alias("w_out"), counts.alias("w_in"),
                         F.col("amount_received_usd").alias("usd"))
    return outgoing.unionByName(incoming)


def entry_history(entries: DataFrame) -> DataFrame:
    """Per ledger entry: the pair's and the account's history in strictly earlier minutes."""
    pair = _earlier(["account", "counterparty"], THREE_DAYS)
    with_pair = (
        entries.withColumn("pair_out_72h", _total(F.col("w_out"), pair))
               .withColumn("pair_in_72h", _total(F.col("w_in"), pair))
               # A counterparty is new when there was no transfer with it, in that
               # direction, in the past 72 hours.
               .withColumn("new_payee", F.col("w_out") * (F.col("pair_out_72h") == 0).cast("int"))
               .withColumn("new_payer", F.col("w_in") * (F.col("pair_in_72h") == 0).cast("int"))
    )
    hour, day, three = (_earlier(["account"], m) for m in (HOUR, DAY, THREE_DAYS))
    w_any = F.col("w_out") + F.col("w_in")
    usd_in, usd_out = F.col("usd") * F.col("w_in"), F.col("usd") * F.col("w_out")
    last = F.max(F.when(w_any > 0, F.col("minute"))).over(three)
    return with_pair.select(
        "tx_id", "side", "pair_out_72h", "pair_in_72h",
        _total(w_any, hour).alias("entries_1h"),
        _total(F.col("w_in"), day).alias("in_24h"),
        _total(F.col("w_out"), day).alias("out_24h"),
        _total(usd_in, day).alias("in_usd_24h"),
        _total(usd_out, day).alias("out_usd_24h"),
        _total(F.col("w_in"), three).alias("in_72h"),
        _total(F.col("w_out"), three).alias("out_72h"),
        _total(usd_in, three).alias("in_usd_72h"),
        _total(usd_out, three).alias("out_usd_72h"),
        _total(F.col("new_payer"), day).alias("new_payers_24h"),
        _total(F.col("new_payee"), day).alias("new_payees_24h"),
        # Capped at the 72-hour horizon, which also covers "no history at all".
        F.coalesce(F.col("minute") - last, F.lit(THREE_DAYS)).alias("minutes_since_last"),
    )


def behavior_columns(tx: DataFrame) -> DataFrame:
    """One row per transaction: tx_id, sender and receiver history, and pair history.

    pair_prior_72h counts earlier transfers from this sender to this receiver in the
    past 72 hours, reverse_prior_72h those in the opposite direction.
    Needs tx_id, timestamp, from_key, to_key, amount_paid_usd and amount_received_usd.
    """
    hist = entry_history(ledger(tx))
    sender = hist.filter(F.col("side") == "from").select(
        "tx_id",
        F.col("pair_out_72h").alias("pair_prior_72h"),
        F.col("pair_in_72h").alias("reverse_prior_72h"),
        *[F.col(c).alias(f"from_{c}") for c in ACCOUNT_COLUMNS])
    receiver = hist.filter(F.col("side") == "to").select(
        "tx_id", *[F.col(c).alias(f"to_{c}") for c in ACCOUNT_COLUMNS])
    return sender.join(receiver, "tx_id")
