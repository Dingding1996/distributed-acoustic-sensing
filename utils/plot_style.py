"""utils/plot_style.py

Shared figure styling for all plots in this project.

Colourmaps
----------
CMAP_BLUE      : blue sequential  (vik left half)  — spatial heatmaps (input, prediction, ground-truth)
CMAP_RED       : red sequential   (vik right half) — secondary data series
CMAP_DIVERGING : blue-red diverging (full vik)     — signed error maps

Standard discrete colours  (3 shades each, light → dark)
---------------------------------------------------------
BLUE_1, BLUE_2, BLUE_3   — train series
RED_1,  RED_2,  RED_3    — validation series
DIV_1,  DIV_2,  DIV_3    — test / diverging series

Helpers
-------
blues(n)  — n shades sampled from CMAP_BLUE
reds(n)   — n shades sampled from CMAP_RED
divs(n)   — n shades sampled from CMAP_DIVERGING

Figure sizes (FigSize)
----------------------
DEFAULT     (6,  4)   — single line/bar chart
WIDE        (10, 5)   — loss curve, histogram
SQUARE      (6,  6)   — scatter plot
MULTI_PANEL (10, 6)   — 2×3 sample grid

Usage::
    from utils.plot_style import (apply_style, FigSize,
                                   CMAP_BLUE, CMAP_DIVERGING,
                                   BLUE_1, BLUE_2, BLUE_3,
                                   RED_1,  RED_2,  RED_3,
                                   DIV_1,  DIV_2,  DIV_3)
    apply_style()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Colourmaps
# ---------------------------------------------------------------------------

CMAP_BLUE = LinearSegmentedColormap.from_list(
    "vik_blue", ["#ece5e0", "#307da6", "#001261"]
)
CMAP_RED = LinearSegmentedColormap.from_list(
    "vik_red", ["#ece5e0", "#c27041", "#590008"]
)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "vik", ["#001261", "#307da6", "#ece5e0", "#c27041", "#590008"]
)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def blues(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n RGBA colours from CMAP_BLUE, light → dark."""
    return [CMAP_BLUE(v) for v in np.linspace(lo, hi, n)]


def reds(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n RGBA colours from CMAP_RED, light → dark."""
    return [CMAP_RED(v) for v in np.linspace(lo, hi, n)]


def divs(n: int, lo: float = 0.35, hi: float = 0.95) -> list:
    """Return n RGBA colours from CMAP_DIVERGING."""
    return [CMAP_DIVERGING(v) for v in np.linspace(lo, hi, n)]


# ---------------------------------------------------------------------------
# Standard discrete colours
# ---------------------------------------------------------------------------

BLUE_1, BLUE_2, BLUE_3 = blues(3)  # train series        (light → dark)
RED_1,  RED_2,  RED_3  = reds(3)   # validation series
DIV_1,  DIV_2,  DIV_3  = divs(3)   # diverging series

GREY_TEST = "#6b6b6b"              # test series — neutral grey for contrast


# ---------------------------------------------------------------------------
# Figure sizes
# ---------------------------------------------------------------------------

class FigSize:
    """Standard figure dimensions used across all plots."""

    DEFAULT     = (6,  4)   # single line/bar chart
    WIDE        = (10, 5)   # loss curve, histogram
    SQUARE      = (6,  6)   # scatter plot
    MULTI_PANEL = (10, 6)   # 2×3 sample grid


# ---------------------------------------------------------------------------
# Global style activator
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply whitegrid theme, blue colour cycle, and default figure size.

    Call once at the start of the script before any plotting.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = FigSize.DEFAULT
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=blues(6))
