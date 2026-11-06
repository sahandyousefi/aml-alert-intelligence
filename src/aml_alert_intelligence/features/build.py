"""Build registry features from transactions, exchange rates, and account attributes."""

from pyspark.sql import DataFrame

from aml_alert_intelligence.data.accounts import add_party_attributes
from aml_alert_intelligence.data.fx import add_usd_amounts
from aml_alert_intelligence.features.registry import FEATURES, Feature

# Carried next to the features: keys for joins and the graph, the time, the split, the
# label, and the dollar amount for dollar-weighted metrics. None of these is a model input.
KEEP = ["tx_id", "timestamp", "from_key", "to_key", "split", "is_tail",
        "is_laundering", "amount_paid_usd"]


def transaction_features(tx: DataFrame, attrs: DataFrame, rates: dict,
                         features: list[Feature] = FEATURES) -> DataFrame:
    """One row per transaction: the KEEP columns that exist, then every registry feature."""
    prepared = add_party_attributes(add_usd_amounts(tx, rates), attrs)
    keep = [c for c in KEEP if c in prepared.columns]
    return prepared.select(*keep, *[f.column() for f in features])
