"""Shared figure style: one neutral series color, one accent for fails, dashed limits."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2933"
SERIES = "#52606d"
MUTED = "#9aa5b1"
ACCENT = "#d64545"  # fails / violations
LIMIT = "#3e7cb1"


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": SERIES,
            "ytick.color": SERIES,
            "axes.grid": True,
            "grid.color": "#e4e7eb",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )


def limit_line(ax, y, label=None, style="--"):
    ax.axhline(y, color=LIMIT, lw=1, ls=style, label=label)
