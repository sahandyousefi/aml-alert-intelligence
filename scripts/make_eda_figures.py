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


def day_label(d) -> str:
    """A date as it reads in a title, for example "Sept 10"."""
    d = pd.Timestamp(d)
    return f"{d:%b} {d.day}".replace("Sep ", "Sept ")


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
    # Titles are built from the data, so they can never disagree with the chart.
    viz.frame(
        fig,
        f"After {day_label(daily.loc[last_normal, 'date'])}, "
        "normal traffic stops while laundering keeps going",
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


# Figure 2: the chronological split ----------------------------------------------

SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}


def split_timeline_data(tx) -> pd.DataFrame:
    """One row per calendar day: its split, tail flag, transactions, and laundering."""
    return (
        tx.groupBy(F.to_date("timestamp").alias("date"), "split", "is_tail")
          .agg(F.count("*").alias("transactions"),
               F.sum("is_laundering").alias("laundering"))
          .orderBy("date")
          .toPandas()
    )


def split_timeline_plot(days: pd.DataFrame, cfg: dict):
    """Laundering per day, colored by split, with a key of each split's totals above."""
    x = list(range(len(days)))
    span = {s: f"{day_label(g['date'].min())} to {pd.Timestamp(g['date'].max()).day}"
            for s, g in days.groupby("split")}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    viz.frame(
        fig,
        f"Train on {span['train']}, tune on {span['val']}, test on {span['test']}",
        "Laundering transactions per day, colored by split. Whole days only, "
        "so no transaction sits on a boundary.",
        "Source: IBM AMLworld, HI-Small. Split dates from configs/base.yaml.",
    )
    # Extra room: a key row above the plot and an axis label below it.
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - (viz.HEADER_IN + 0.9) / height,
                        bottom=(viz.FOOTER_IN + 0.35) / height)

    # The simulation tail, shaded as in Figure 1.
    tail = days["is_tail"].to_numpy()
    first_tail = int(tail.argmax())
    ax.axvspan(first_tail - 0.5, len(days) - 0.5, color=viz.BAND, zorder=0)

    y_max = days["laundering"].max() * 1.15
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.set_ylim(0, y_max)
    viz.bars(ax, x, days["laundering"], [viz.SPLIT_COLORS[s] for s in days["split"]])
    viz.value_axis(ax, "y", nbins=4)
    ax.text(first_tail - 0.2, y_max * 0.95, "Simulation tail",
            fontsize=10, color=viz.INK_SECONDARY, va="top")

    # A colored rule over the days of each split, and a key with its totals.
    above = ax.get_xaxis_transform()        # x in data units, y in axes fraction
    key_x = {"train": 0.0, "val": 0.34, "test": 0.64}
    for split, group in days.groupby("split", sort=False):
        lo, hi = group.index.min(), group.index.max()
        color = viz.SPLIT_COLORS[split]
        ax.plot([lo - 0.4, hi + 0.4], [1.02, 1.02], transform=above, color=color,
                linewidth=3, solid_capstyle="butt", clip_on=False)

        n_tx = int(group["transactions"].sum())
        n_laund = int(group["laundering"].sum())
        detail = f"{n_tx:,} transactions\n{n_laund:,} laundering"
        if split == "test":
            before_tail = int(group.loc[~group["is_tail"], "laundering"].sum())
            detail += f", {before_tail:,} before the tail"
        kx = key_x[split]
        ax.scatter([kx + 0.008], [1.30], transform=ax.transAxes, marker="s", s=60,
                   color=color, clip_on=False)
        ax.text(kx + 0.025, 1.30, SPLIT_NAMES[split], transform=ax.transAxes,
                fontsize=11.5, fontweight="semibold", color=viz.INK, va="center")
        ax.text(kx, 1.255, detail, transform=ax.transAxes, fontsize=10,
                color=viz.INK_SECONDARY, va="top", linespacing=1.45)

    dates = pd.to_datetime(days["date"])
    ax.set_xticks(x, [str(d.day) for d in dates])
    ax.set_xlabel(f"Day of {dates.iloc[0]:%B %Y}", labelpad=8)

    path = viz.save(fig, cfg, "fig2_split_timeline")
    plt.close(fig)
    return path


def split_timeline(tx, cfg):
    return split_timeline_plot(split_timeline_data(tx), cfg)


# Figure 3: laundering risk by payment format -------------------------------------

def payment_format_data(tx) -> pd.DataFrame:
    """Training days only: transactions, laundering, and lift for each payment format."""
    table = (
        tx.filter(F.col("split") == "train")
          .groupBy("payment_format")
          .agg(F.count("*").alias("transactions"),
               F.sum("is_laundering").alias("laundering"))
          .toPandas()
    )
    base_rate = table["laundering"].sum() / table["transactions"].sum()
    table["lift"] = table["laundering"] / table["transactions"] / base_rate
    table.attrs["base_rate"] = base_rate
    return table.sort_values(["lift", "transactions"], ascending=False).reset_index(drop=True)


def payment_format_plot(table: pd.DataFrame, cfg: dict):
    """Lift per format as slim horizontal bars, with the counts behind each bar beside it."""
    top = table.iloc[0]
    share = top["laundering"] / table["laundering"].sum()
    rows = list(range(len(table)))[::-1]            # highest lift at the top

    fig, ax = plt.subplots(figsize=(9, 4.9))
    viz.frame(
        fig,
        f"{top['payment_format']} carries {share:.0%} of training laundering "
        f"at {top['lift']:.1f} times the average rate",
        "Lift: the laundering rate within a payment format divided by the overall "
        f"training rate ({table.attrs['base_rate']:.3%}).",
        f"Source: IBM AMLworld, HI-Small, training days only (through "
        f"{day_label(cfg['split']['train_end'])}), so the test set does not shape feature choices.",
    )
    # Room on the right for the counts column, and above the plot for its header.
    height = fig.get_figheight()
    fig.subplots_adjust(right=0.74, top=1 - (viz.HEADER_IN + 0.3) / height)

    lift_max = table["lift"].max()
    ax.set_xlim(0, lift_max * 1.18)
    ax.set_ylim(-0.6, len(table) - 0.4)
    colors = [viz.ACCENT if i == 0 else viz.DATA_GREY for i in range(len(table))]
    viz.bars(ax, rows, table["lift"], colors, horizontal=True)
    ax.set_yticks(rows, table["payment_format"])
    viz.value_axis(ax, "x", nbins=4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))

    # The average (lift 1) as a reference line behind the bars.
    ax.axvline(1, color=viz.INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=0.8)
    ax.text(1, 1.01, "average", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=10, color=viz.INK_MUTED)

    # Lift at the end of each bar; short bars put it past the average line.
    for y, (_, r) in zip(rows, table.iterrows()):
        strong = r["payment_format"] == top["payment_format"]
        label = f"{r['lift']:.2f}x" if r["laundering"] > 0 else "none"
        ax.text(max(r["lift"], 1) + lift_max * 0.02, y, label, va="center",
                fontsize=10.5, color=viz.INK if strong else viz.INK_SECONDARY,
                fontweight="semibold" if strong else "normal")

    # A column of the counts behind each bar, so no lift is read without its size.
    side = ax.get_yaxis_transform()                 # x in axes fraction, y in data units
    ax.text(1.04, 1.01, "Laundering / transactions", transform=ax.transAxes,
            fontsize=10, fontweight="semibold", color=viz.INK_SECONDARY, va="bottom")
    for y, (_, r) in zip(rows, table.iterrows()):
        ax.text(1.04, y, f"{int(r['laundering']):,} / {int(r['transactions']):,}",
                transform=side, va="center", fontsize=10.5, color=viz.INK_SECONDARY)

    path = viz.save(fig, cfg, "fig3_payment_format")
    plt.close(fig)
    return path


def payment_format(tx, cfg):
    return payment_format_plot(payment_format_data(tx), cfg)



# Figure 4: laundering by amount band ----------------------------------------------

MIN_CASES = 20          # a band with fewer laundering cases is drawn as unreliable
FAINT = "#e6eaef"       # lighter than the context grey, for unreliable bands


def money(v: float) -> str:
    """An amount in short form: $10, $1K, $10M.

    The dollar sign is escaped, because matplotlib reads text between two
    dollar signs as a math formula.
    """
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= size:
            return rf"\${v / size:g}{unit}"
    return rf"\${v:g}" if v >= 1 else rf"\${v:.2f}"


def amount_band_data(tx) -> pd.DataFrame:
    """Training days, US Dollar to US Dollar: transactions and laundering per order of magnitude."""
    usd = tx.filter(
        (F.col("split") == "train")
        & (F.col("payment_currency") == "US Dollar")
        & (F.col("receiving_currency") == "US Dollar")
    )
    # Each band is named by its lower edge: 1, 10, 100, 1000, ...
    band = F.pow(F.lit(10.0), F.floor(F.log10(F.greatest("amount_paid", F.lit(0.01)))))
    table = (
        usd.groupBy(band.alias("band"))
           .agg(F.count("*").alias("transactions"),
                F.sum("is_laundering").alias("laundering"))
           .orderBy("band")
           .toPandas()
    )
    base_rate = table["laundering"].sum() / table["transactions"].sum()
    table["lift"] = table["laundering"] / table["transactions"] / base_rate
    return table


def amount_band_plot(table: pd.DataFrame, cfg: dict):
    """Lift per amount band; the reliable above-average bands carry the accent."""
    reliable = table["laundering"] >= MIN_CASES
    focus = reliable & (table["lift"] > 1)
    lo, hi = table.loc[focus, "band"].min(), table.loc[focus, "band"].max() * 10
    in_range = int(table.loc[focus, "laundering"].sum())
    total = int(table["laundering"].sum())
    x = list(range(len(table)))

    fig, ax = plt.subplots(figsize=(9, 4.9))
    viz.frame(
        fig,
        f"Laundering concentrates between {money(lo)} and {money(hi)}",
        f"{in_range:,} of {total:,} US Dollar laundering cases ({in_range / total:.0%}) "
        "fall in this range. Lift by order of magnitude of the amount.",
        f"Source: IBM AMLworld, HI-Small, training days only (through "
        f"{day_label(cfg['split']['train_end'])}), US Dollar transfers. "
        f"Faint bars rest on fewer than {MIN_CASES} laundering cases.",
    )
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - (viz.HEADER_IN + 0.05) / height,
                        bottom=(viz.FOOTER_IN + 0.35) / height)

    lift_max = table["lift"].max()
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylim(0, lift_max * 1.12)
    colors = [viz.ACCENT if f else (viz.DATA_GREY if r else FAINT)
              for f, r in zip(focus, reliable)]
    viz.bars(ax, x, table["lift"], colors)
    viz.value_axis(ax, "y", nbins=4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}x"))

    # The average (lift 1) as a reference line behind the bars.
    ax.axhline(1, color=viz.INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=0.8)
    ax.text(len(table) - 0.45, 1, "average", ha="right", va="bottom",
            fontsize=10, color=viz.INK_MUTED)

    # Labels: the lift on the accent bars, the case count on tall unreliable ones.
    pad = lift_max * 0.02
    for i, r in table.iterrows():
        if focus[i]:
            ax.text(i, r["lift"] + pad, f"{r['lift']:.2f}x", ha="center", va="bottom",
                    fontsize=10.5, fontweight="semibold", color=viz.INK)
        elif not reliable[i] and r["lift"] > 1:
            ax.text(i, r["lift"] + pad, f"{int(r['laundering'])} cases", ha="center",
                    va="bottom", fontsize=10, color=viz.INK_MUTED)

    ax.set_xticks(x, [money(b) for b in table["band"]])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Amount paid (each band runs from its label to ten times it)", labelpad=8)

    path = viz.save(fig, cfg, "fig4_amount_bands")
    plt.close(fig)
    return path


def amount_bands(tx, cfg):
    return amount_band_plot(amount_band_data(tx), cfg)


# Registry and entry point -------------------------------------------------------

FIGURES = {
    "daily_volume": daily_volume,
    "split_timeline": split_timeline,
    "payment_format": payment_format,
    "amount_bands": amount_bands,
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
