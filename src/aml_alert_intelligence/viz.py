"""Shared figure style: one palette, one type scale, one way to title and save."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

# Surface and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Data is grey unless it carries the finding; the finding gets the accent.
DATA_GREY = "#c3c2b7"
ACCENT = "#2a78d6"
BAND = "#f0efec"

# Fixed categorical order, checked for colorblind separation. Colors follow the
# entity: train is always blue, validation orange, test aqua.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPLIT_COLORS = {"train": CATEGORICAL[0], "val": CATEGORICAL[1], "test": CATEGORICAL[2]}

HEADER_INCHES = 0.85  # space reserved above the plot for title and subtitle


def apply_style() -> None:
    """Set matplotlib defaults for every figure in the project."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def header(fig, title: str, subtitle: str = "") -> None:
    """Title that states the finding, and a quieter subtitle, pinned to the top-left corner."""
    def corner(down_pts):
        return offset_copy(fig.transFigure, fig=fig, x=8, y=-down_pts, units="points")

    fig.text(0, 1, title, transform=corner(12), fontsize=14, fontweight="bold",
             color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0, 1, subtitle, transform=corner(34), fontsize=10,
                 color=INK_SECONDARY, va="top", ha="left")
    fig.subplots_adjust(top=1 - HEADER_INCHES / fig.get_figheight())


def compact(value, _pos=None) -> str:
    """Axis label in short form: 400K, 1.2M."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


COMPACT = FuncFormatter(compact)


def save(fig, cfg: dict, name: str) -> Path:
    """Save a figure as PNG under outputs/figures on Drive and return its path."""
    out_dir = cfg["paths"]["outputs"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    return path
