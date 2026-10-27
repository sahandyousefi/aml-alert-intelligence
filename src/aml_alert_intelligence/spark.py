"""Create the Spark session from config settings."""

from pyspark.sql import SparkSession


def get_spark(cfg: dict) -> SparkSession:
    """Build (or reuse) a local Spark session configured from cfg["spark"]."""
    s = cfg["spark"]
    return (
        SparkSession.builder
        .appName(s["app_name"])
        .master(s["master"])
        .config("spark.driver.memory", s["driver_memory"])
        .config("spark.sql.shuffle.partitions", s["shuffle_partitions"])
        .config("spark.sql.session.timeZone", s["timezone"])
        .getOrCreate()
    )
