"""Feature registry: every model feature is declared once, here.

Each entry holds the Spark expression that computes the feature, its family and type,
a definition for the feature dictionary, and a short label used wherever the feature is
shown to a person: explanation charts, alert evidence, and SAR drafts.
docs/features.md is generated from this module by scripts/make_feature_docs.py.
"""

from dataclasses import dataclass
from typing import Callable

from pyspark.sql import Column
from pyspark.sql import functions as F

FAMILIES = {
    "A": "Transaction",
    "B": "Account attributes",
}

KINDS = ("numeric", "categorical", "flag")


@dataclass(frozen=True)
class Feature:
    name: str
    family: str                     # a key of FAMILIES
    kind: str                       # one of KINDS; flags are stored as 0 or 1
    compute: Callable[[], Column]
    label: str
    definition: str

    def column(self) -> Column:
        """The feature as a named Spark column."""
        col = self.compute()
        if self.kind == "flag":
            col = col.cast("int")
        return col.alias(self.name)


def _party(side: str) -> list[Feature]:
    """Family B features for one side of a transfer: "from" (sender) or "to" (receiver)."""
    who = {"from": "sender", "to": "receiver"}[side]
    title = who.capitalize()
    return [
        Feature(f"{side}_entity_type", "B", "categorical",
                lambda: F.col(f"{side}_entity_type"),
                f"{title} owner type",
                f"Owner type of the {who}'s account: Partnership, Corporation, Sole "
                "Proprietorship, Individual, Country or Direct"),
        Feature(f"{side}_bank_country", "B", "categorical",
                lambda: F.col(f"{side}_bank_country"),
                f"{title} bank country",
                f"Country of the {who}'s bank, read from the bank name; Crypto for crypto "
                "banks, United States for unnumbered US bank names"),
        Feature(f"{side}_owner_accounts_log", "B", "numeric",
                lambda: F.log10(F.col(f"{side}_owner_accounts")),
                f"{title} owner's accounts (log10)",
                f"log10 of the number of accounts held by the {who}'s owner"),
        Feature(f"{side}_owner_banks_log", "B", "numeric",
                lambda: F.log10(F.col(f"{side}_owner_banks")),
                f"{title} owner's banks (log10)",
                f"log10 of the number of banks at which the {who}'s owner holds accounts"),
    ]


FEATURES: list[Feature] = [
    # Family A: the transaction itself.
    Feature("amount_usd_log", "A", "numeric",
            lambda: F.log10(F.col("amount_paid_usd") + 1),
            "Amount (USD, log10)",
            "log10(1 + amount paid in US dollars), converted with the simulator's fixed "
            "exchange rates fitted on training days"),
    Feature("payment_format", "A", "categorical",
            lambda: F.col("payment_format"),
            "Payment format",
            "Payment method: ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment or Wire"),
    Feature("payment_currency", "A", "categorical",
            lambda: F.col("payment_currency"),
            "Payment currency",
            "Currency the sender pays in"),
    Feature("receiving_currency", "A", "categorical",
            lambda: F.col("receiving_currency"),
            "Receiving currency",
            "Currency the receiver is paid in"),
    Feature("is_cross_currency", "A", "flag",
            lambda: F.col("payment_currency") != F.col("receiving_currency"),
            "Cross-currency transfer",
            "1 if the two sides use different currencies"),
    Feature("is_self_transfer", "A", "flag",
            lambda: F.col("from_key") == F.col("to_key"),
            "Transfer to the same account",
            "1 if sender and receiver are the same account"),
    Feature("is_same_bank", "A", "flag",
            lambda: F.col("from_bank_id") == F.col("to_bank_id"),
            "Same bank",
            "1 if both accounts are at the same bank"),
    Feature("hour_of_day", "A", "numeric",
            lambda: F.hour("timestamp"),
            "Hour of day",
            "Hour of the transfer, 0 to 23, as recorded"),
    Feature("is_weekend", "A", "flag",
            lambda: F.dayofweek("timestamp").isin(1, 7),
            "Weekend",
            "1 on Saturday or Sunday; replaces day of week, which would identify single dates"),
    # Family B: the two parties, from accounts.csv.
    *_party("from"),
    *_party("to"),
    Feature("is_cross_border", "B", "flag",
            lambda: F.col("cross_border"),
            "Cross-border transfer",
            "1 if the two banks are in different countries"),
    Feature("is_same_owner", "B", "flag",
            lambda: F.col("same_owner"),
            "Same owner",
            "1 if both accounts belong to the same owner, self-transfers included"),
]

# Left out on purpose, with the reason. Shown in the feature dictionary.
EXCLUDED = [
    ("Day of week", "Six training days, so each weekday appears once and would identify a "
                    "single date; replaced by the weekend flag"),
    ("Calendar date, day of month, time since data start", "A hidden clock that points at "
                                                           "the simulation tail"),
    ("Account, bank and owner IDs", "Memorization across the 147 laundering attempts that "
                                    "span splits; IDs are kept for joins only"),
    ("Lifetime totals", "They grow with time"),
    ("Past laundering of an account", "Label leakage"),
    ("Target encoding", "Not needed; XGBoost handles categories natively"),
    ("Round-amount flags", "No signal on the training days"),
]


def by_name(name: str) -> Feature:
    """Look up one feature by its name."""
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)


def dictionary_markdown() -> str:
    """The feature dictionary as Markdown, one table per family."""
    lines = [
        "# Feature dictionary",
        "",
        "Generated from `src/aml_alert_intelligence/features/registry.py` by "
        "`scripts/make_feature_docs.py`. Do not edit by hand.",
        "",
        f"{len(FEATURES)} features. Flags are 0 or 1. No feature uses the label, a calendar "
        "date, or an account identifier, and none is fitted on anything but training days.",
    ]
    for key, title in FAMILIES.items():
        members = [f for f in FEATURES if f.family == key]
        lines += ["", f"## {key}. {title}", "",
                  "| Feature | Type | Label | Definition |", "| --- | --- | --- | --- |"]
        lines += [f"| `{f.name}` | {f.kind} | {f.label} | {f.definition} |" for f in members]
    lines += ["", "## Excluded on purpose", "", "| Left out | Reason |", "| --- | --- |"]
    lines += [f"| {what} | {why} |" for what, why in EXCLUDED]
    return "\n".join(lines) + "\n"
