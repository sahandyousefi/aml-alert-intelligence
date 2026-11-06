"""Account attributes from accounts.csv: owner type, bank country, and owner size.

Every attribute is fixed per account, as a bank would know it from onboarding records,
so none of them carries information from later transactions.
"""

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from aml_alert_intelligence.data.load import load_accounts

NUMBERED = r"^(.*\S) #(\d+)$"
US = "United States"
CRYPTO = "Crypto"
CRYPTO_STEMS = ["Crytpo Bank", "Crypto Bank"]    # the source spells it "Crytpo"

# Attributes carried onto each side of a transaction. entity_id is kept for linking
# (same owner, graph construction) and is never a model feature.
PARTY_COLUMNS = ["entity_id", "entity_type", "bank_country", "owner_accounts", "owner_banks"]


def _stem(col: str) -> Column:
    """Text before a trailing " #<number>", or "" when the name has no number."""
    return F.regexp_extract(F.col(col), NUMBERED, 1)


def entity_type(col: str = "entity_name") -> Column:
    """Owner type from the entity name, e.g. "Corporation #33520" -> "Corporation".

    Null when the name does not end in a number, so an unexpected name shows up in checks.
    """
    stem = _stem(col)
    return F.when(stem != "", stem)


def bank_country(col: str = "bank_name") -> Column:
    """Country from the bank name, e.g. "Germany Bank #4815" -> "Germany".

    "Crytpo Bank #n" is the crypto category, and unnumbered names such as "Savings Bank
    of Omaha" are US banks; both rules were checked against the currencies accounts use.
    Null for any other pattern.
    """
    stem = _stem(col)
    return (F.when(stem == "", F.lit(US))
             .when(stem.isin(*CRYPTO_STEMS), F.lit(CRYPTO))
             .when(stem.endswith(" Bank"), F.regexp_replace(stem, " Bank$", "")))


def account_attributes(accounts: DataFrame) -> DataFrame:
    """One row per account: owner, owner type, bank country, and the owner's size.

    owner_accounts and owner_banks count the accounts and distinct banks of the owner.
    Pass the output of load_accounts.
    """
    owners = accounts.groupBy("entity_id").agg(
        F.count("*").alias("owner_accounts"),
        F.countDistinct("bank_id").alias("owner_banks"),
    )
    return accounts.join(owners, "entity_id").select(
        "account_key",
        "bank_id",
        "entity_id",
        entity_type().alias("entity_type"),
        bank_country().alias("bank_country"),
        "owner_accounts",
        "owner_banks",
    )


def add_party_attributes(tx: DataFrame, attrs: DataFrame) -> DataFrame:
    """Add sender (from_) and receiver (to_) attributes, cross_border and same_owner.

    Left joins, so no transaction is dropped; an account missing from attrs gives nulls.
    """
    def side(prefix: str, key: str) -> DataFrame:
        return attrs.select(F.col("account_key").alias(key),
                            *[F.col(c).alias(f"{prefix}_{c}") for c in PARTY_COLUMNS])

    added = [f"{p}_{c}" for p in ("from", "to") for c in PARTY_COLUMNS]
    joined = (tx.join(side("from", "from_key"), "from_key", "left")
                .join(side("to", "to_key"), "to_key", "left"))
    return joined.select(
        *tx.columns,
        *added,
        (F.col("from_bank_country") != F.col("to_bank_country")).alias("cross_border"),
        (F.col("from_entity_id") == F.col("to_entity_id")).alias("same_owner"),
    )


def accounts_parquet_path(cfg: dict) -> Path:
    """Processed account attribute table for the configured dataset variant."""
    variant = cfg["data"]["variant"]
    return cfg["paths"]["processed_data"] / f"{variant}_accounts.parquet"


def build_account_attributes(spark: SparkSession, cfg: dict) -> Path:
    """Parse accounts.csv once and save the attribute table as Parquet."""
    out = accounts_parquet_path(cfg)
    account_attributes(load_accounts(spark, cfg)).write.mode("overwrite").parquet(str(out))
    return out


def read_account_attributes(spark: SparkSession, cfg: dict) -> DataFrame:
    """Read the processed account attribute table."""
    return spark.read.parquet(str(accounts_parquet_path(cfg)))
