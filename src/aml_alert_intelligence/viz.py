"""Shared figure style: one font, one palette, one layout, one way to save."""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import offset_copy

# Colors chosen to sit cleanly on a GitHub README (white page, GitHub's own greys).
SURFACE = "#ffffff"
INK = "#1f2328"
INK_SECONDARY = "#59636e"
INK_MUTED = "#818b98"
GRID = "#eef1f4"
AXIS = "#d1d9e0"
DATA_GREY = "#d1d9e0"
BAND = "#f6f8fa"
ACCENT = "#2a78d6"

# Fixed categorical order, checked for colorblind separation. Colors follow the
# entity: train is always blue, validation orange, test aqua.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPLIT_COLORS = {"train": CATEGORICAL[0], "val": CATEGORICAL[1], "test": CATEGORICAL[2]}

HEADER_IN = 0.95       # space above the plot for title and subtitle
FOOTER_IN = 0.75       # space below the plot for tick labels and the source note
BAR_MAX_PX = 26        # bars stay slim however much room they have
CORNER_PX = 4          # rounded end on the data side of each bar


def _font_family() -> str:
    """Use Source Sans Pro when the font package is installed, else matplotlib's default."""
    try:
        import font_source_sans_pro
    except ImportError:
        return "DejaVu Sans"
    files = Path(font_source_sans_pro.__file__).parent / "files"
    for f in files.glob("SourceSansPro-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    return "Source Sans Pro"


def apply_style() -> None:
    """Set matplotlib defaults for every figure in the project."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": _font_family(),
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 11,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,
        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def frame(fig, title: str, subtitle: str = "", note: str = "") -> None:
    """Finding-first title, quieter subtitle, and a small source note, at fixed distances."""
    def shifted(dy_pts):
        return offset_copy(fig.transFigure, fig=fig, x=10, y=dy_pts, units="points")

    fig.text(0, 1, title, transform=shifted(-14), fontsize=16, fontweight="semibold",
             color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0, 1, subtitle, transform=shifted(-38), fontsize=11.5,
                 color=INK_SECONDARY, va="top", ha="left")
    if note:
        fig.text(0, 0, note, transform=shifted(10), fontsize=9.5,
                 color=INK_MUTED, va="bottom", ha="left")
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - HEADER_IN / height,
                        bottom=(FOOTER_IN if note else 0.45) / height)


def compact(value, _pos=None) -> str:
    """Axis label in short form: 400K, 1.2M."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


COMPACT = FuncFormatter(compact)


def value_axis(ax, axis: str = "y", nbins: int = 5) -> None:
    """Round ticks and gridlines on the value axis, a
