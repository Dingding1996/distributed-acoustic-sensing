"""utils/plot_style.py

Portfolio-wide figure styling.  Three palettes are defined:
  CMAP_BLUE     — blue-green palette
  CMAP_RED      — red-salmon palette
  CMAP_BLUE_RED — blue to orange diverging palette

Usage (Section 0 of any notebook)::

    plot_style = _load_module("plot_style", "utils/plot_style.py")
    from plot_style import (apply_style, FigSize,
                            CMAP_BLUE, C1, C2, C3, FAULT_COLORS,
                            CMAP_RED, D1, D2, D3, FAULT_COLORS_DMG,
                            CMAP_BLUE_RED, I1, I2, I3, FAULT_COLORS_IR)
    apply_style()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Palettes — single source of truth for all colours
# ---------------------------------------------------------------------------

CMAP_BLUE     = sns.color_palette("ch:start=.2,rot=-.3",  as_cmap=True)
CMAP_RED      = sns.color_palette("light:#d6604d",          as_cmap=True)
CMAP_BLUE_RED = LinearSegmentedColormap.from_list(
    "blue_orange", ["#2166ac", "#f7f7f7", "#d6604d"]
)


def blues(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP_BLUE.

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap (0 = lightest, 1 = darkest).
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples, light-to-dark.
    """
    return [CMAP_BLUE(v) for v in np.linspace(lo, hi, n)]


def salmons(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP_RED.

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap.
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples.
    """
    return [CMAP_RED(v) for v in np.linspace(lo, hi, n)]


def blue_reds(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n evenly-spaced RGBA colours from CMAP_BLUE_RED.

    Args:
        n:   Number of colours to return.
        lo:  Lower bound on the colormap.
        hi:  Upper bound on the colormap.

    Returns:
        List of n RGBA tuples.
    """
    return [CMAP_BLUE_RED(v) for v in np.linspace(lo, hi, n)]


# Three standard line colours — blue and red series with readable aliases
C1, C2, C3 = blues(3)
D1, D2, D3 = salmons(3)
I1, I2, I3 = blue_reds(3)

# Readable aliases
blue1, blue2, blue3 = C1, C2, C3
red1,  red2,  red3  = D1, D2, D3

# Fault-frequency marker colours
FAULT_COLORS: dict = dict(zip(
    ["BPFO", "BPFI"],
    blues(2, lo=0.40, hi=0.95),
))

FAULT_COLORS_DMG: dict = dict(zip(
    ["BPFO", "BPFI"],
    salmons(2, lo=0.40, hi=0.95),
))

FAULT_COLORS_IR: dict = dict(zip(
    ["BPFO", "BPFI"],
    blue_reds(2, lo=0.40, hi=0.95),
))

# ---------------------------------------------------------------------------
# Figure sizes  (CLAUDE.md §11.3) — scaled down ~20 % from original
# ---------------------------------------------------------------------------


class FigSize:
    """Standard figure dimensions used across all notebooks."""

    DEFAULT            = (6,   4)    # bar charts, general
    HEATMAP            = (5,   3.5)  # correlation / confusion matrix (small)
    HEATMAP_LARGE      = (6,   5)    # confusion matrix (large)
    FEATURE_IMPORTANCE = (6,   4.5)  # wide horizontal bar
    COUNT              = (4,   3)    # small count / distribution plot
    MULTI_PANEL        = (10,  6)    # large grid of subplots
    WIDE_TALL          = (10,  5)    # DSP multi-channel subplots


# ---------------------------------------------------------------------------
# Global activator
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply colour cycle, whitegrid theme, and default figure size.

    Call once at the bottom of Section 0 imports, replacing the bare
    sns.set_theme / plt.rcParams lines.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = FigSize.DEFAULT
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=blues(6))
