"""Chronological train, validation, and test split on whole calendar days."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_split(df: DataFrame, cfg: dict) -> DataFrame:
    """Label each transaction train, val, or test by its day, and flag the simulation tail.

    End dates are inclusive: day <= train_end is train, day <= val_end is val,
    later days are test. Days on or after tail_start get is_tail = True.
    """
    s = cfg["split"]
    if not s["train_end"] < s["val_end"] < s["tail_start"]:
        raise ValueError("split dates must satisfy train_end < val_end < tail_start")

    day = F.to_date("timestamp")
    return (
        df.withColumn(
            "split",
            F.when(day <= F.lit(s["train_end"]).cast("date"), "train")
             .when(day <= F.lit(s["val_end"]).cast("date"), "val")
             .otherwise("test"),
        )
        .withColumn("is_tail", day >= F.lit(s["tail_start"]).cast("date"))
    )
