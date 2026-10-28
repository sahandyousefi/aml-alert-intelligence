"""Shared test fixtures."""

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    """A small local Spark session, started once and shared by every test in the run."""
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("aml-alert-intelligence-tests")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()
