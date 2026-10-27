"""Build the Phase 1 exploration figures from the processed transactions.

In Colab, run it inside the notebook so it reuses the open Spark session:
    %run scripts/make_eda_figures.py --only daily_volume
From a terminal:
    python scripts/make_eda_figures.py --config configs/base.yaml

Each figure has two parts: a data function that aggregates with Spark and returns
a small pandas table, and a plot function that draws that table. Keeping them apart
means the drawing can be checked without Spark.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F

from aml_alert_intelligence import viz
from aml_alert_intelligence.config import load_config
from aml_alert_intelligence.data.load import read_transactions
from aml_alert_intelligence.data.split import add_split
from aml_alert_intelligence.spark import get_spark

SOURCE = "Source: IBM AMLworld, HI-Small, all 5,078,345 transactions"


# Figure 1: daily volume and laundering rate ------------------------------------

def daily_volume_data(tx) -> pd.DataFrame:
    """One row per calendar day: transactions, laundering count, laundering rate."""
    daily = (
        tx.groupBy(F.to_date("timestamp").alias("date"))
          .agg(F.count("*").alias("transactions"),
               F.sum("is_laundering").alias("laundering"))
          .orderBy("date")
          .toPandas()
    )
    daily["rate_pct"] = daily["laundering"] / daily["transactions"] * 100
    return daily


def daily_volume_plot(daily: pd.DataFrame, cfg: dict):
    """Two panels on one time axis: volume on top, laundering rate (log scale) below."""
    tail = pd.to_datetime(daily["date"]) >= pd.Timestamp(cfg["split"]["tail_start"])
    x = range(len(daily))
    first_tail = int(tail.idxmax())
    last_normal = first_tail - 1

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})

    # Lay out the figure first, so bar sizes are measured on the final layout.
    viz.frame(
        fig,
        "After Sept 10, normal traffic stops while laundering keeps going",
        "Transactions and laundering rate per day. The shaded days are kept "
        "in the data and reported separately.",
        SOURCE,
    )
    # Extra room: each panel carries its own title under the figure header.
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - (viz.HEADER_IN + 0.35) / height,
                        bottom=(viz.FOOTER_IN + 0.35) / height, hspace=0.38)

    # The simulation tail, shaded in both panels.
    for ax in (top, bottom):
        ax.axvspan(first_tail - 0.5, len(daily) - 0.5, color=viz.BAND, zorder=0)
    top.set_xlim(-0.6, len(daily) - 0.4)

    # Top panel: transactions per day.
    y_max = daily["transactions"].max() * 1.12
    top.set_ylim(0, y_max)
    colors = [viz.ACCENT if t else viz.DATA_GREY for t in tail]
    viz.bars(top, x, daily["transactions"], colors)
    viz.value_axis(top, "y", nbins=4)
    top.set_title("Transactions per day", loc="left", fontsize=12,
                  fontweight="semibold", color=viz.INK, pad=10)
    tail_tx = int(daily.loc[tail, "transactions"].sum())
    top.text(first_tail - 0.2, y_max * 0.86,
             f"Simulation tail\n{tail_tx:,} transactions in {tail.sum()} days",
             fontsize=10, color=viz.INK_SECONDARY, va="top", linespacing=1.5)

    # Bottom panel: laundering rate on a log scale, so 0.1% and 60% fit one axis.
    normal = ~tail
    bottom.plot([i for i in x if normal[i]], daily.loc[normal, "rate_pct"],
                color=viz.INK_MUTED, linewidth=2, zorder=2)
    bottom.plot([last_normal, first_tail], daily.loc[[last_normal, first_tail], "rate_pct"],
                color=viz.ACCENT, linewidth=2, zorder=2)
    bottom.plot([i for i in x if tail[i]], daily.loc[tail, "rate_pct"],
                color=viz.ACCENT, linewidth=2, zorder=2)
    bottom.scatter(x, daily["rate_pct"], s=42, zorder=3, linewidths=2,
                   edgecolors=viz.SURFACE,
                   c=[viz.ACCENT if t else viz.INK_MUTED for t in tail])
    bottom.set_yscale("log")
    bottom.set_ylim(0.01, 200)
    ticks = [0.01, 0.1, 1, 10, 100]
    bottom.set_yticks(ticks, [f"{t:g}%" for t in ticks])
    bottom.minorticks_off()
    bottom.set_title("Laundering rate per day (log scale)", loc="left", fontsize=12,
                     fontweight="semibold", color=viz.INK, pad=10)

    # Label only the two days the story turns on.
    r_before = daily.loc[last_normal, "rate_pct"]
    r_after = daily.loc[first_tail, "rate_pct"]
    bottom.annotate(f"{r_before:.2f}%", (last_normal, r_before), xytext=(-10, 12),
                    textcoords="offset points", ha="right", fontsize=10.5, color=viz.INK_SECONDARY)
    bottom.annotate(f"{r_after:.1f}%", (first_tail, r_after), xytext=(-10, 4),
                    textcoords="offset points", ha="right", va="bottom", fontsize=10.5,
                    fontweight="semibold", color=viz.INK)

    # Day numbers along the shared time axis.
    days = pd.to_datetime(daily["date"])
    bottom.set_xticks(list(x), [str(d.day) for d in days])
    bottom.set_xlabel(f"Day of {days.iloc[0]:%B %Y}", labelpad=8)
    top.tick_params(axis="x", labelbottom=False)

    path = viz.save(fig, cfg, "fig1_daily_volume")
    plt.close(fig)
    return path


def daily_volume(tx, cfg):
    return daily_volume_plot(daily_volume_data(tx), cfg)


# Registry and entry point -------------------------------------------------------

FIGURES = {
    "daily_volume": daily_volume,
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 exploration figures.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES),
                        help="build only these figures")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    spark = get_spark(cfg)
    viz.apply_style()

    # Every figure reads the same table, so keep it in memory while they are built.
    tx = add_split(read_transactions(spark, cfg), cfg).cache()
    for name in args.only or FIGURES:
        print(f"saved {FIGURES[name](tx, cfg)}")
    tx.unpersist()


if __name__ == "__main__":
    main()
