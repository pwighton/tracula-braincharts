"""QA figures (static PNG)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Light chart tokens and the first two categorical slots of the dataviz reference palette (validated all-pairs:
# CVD dE 24.7, normal-vision dE 33.6). Sex is also encoded by marker shape, so identity isn't colour alone.
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEXES = [(1, "Male", "#2a78d6", "o"), (0, "Female", "#eb6834", "^")]  # (code, label, colour, marker), fixed order
FLAG_Z = 2.0


def _short(tract: str) -> str:
    return tract.removeprefix("FA_mean_of_").replace("_", " ")


def centile_panels(curves: pd.DataFrame, points: pd.DataFrame, tracts: list[str], title: str, path: Path) -> None:
    """One panel per tract: 50th and corrected 5th/95th centile curves per sex, and the controls' FA with a dark
    ring where their (cross-validated) |z_cal| > 2.

    ``curves``: ``tract, age, sex, fa_p50, fa_p05_cal, fa_p95_cal``; ``points``: ``tract, age, sex, fa, z_cal``.
    """
    ncol = 6
    nrow = math.ceil(len(tracts) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.2 * nrow + 0.9), sharex=True, squeeze=False,
                             facecolor=SURFACE, layout="constrained")
    for ax, tract in zip(axes.flat, tracts):
        ax.set_facecolor(SURFACE)
        c_tract, p_tract = curves[curves["tract"] == tract], points[points["tract"] == tract]
        for code, _label, colour, marker in SEXES:
            c = c_tract[c_tract["sex"] == code].sort_values("age")
            ax.plot(c["age"], c["fa_p50"], color=colour, lw=1.5, zorder=2)
            for col in ("fa_p05_cal", "fa_p95_cal"):
                ax.plot(c["age"], c[col], color=colour, lw=0.9, ls=(0, (4, 2)), zorder=2)
            p = p_tract[p_tract["sex"] == code]
            flagged = p["z_cal"].abs() > FLAG_Z
            for sel, edge, width, z in ((~flagged, SURFACE, 1.0, 3), (flagged, TEXT, 1.4, 4)):
                ax.scatter(p.loc[sel, "age"], p.loc[sel, "fa"], s=30, color=colour, marker=marker,
                           edgecolors=edge, linewidths=width, zorder=z)
        ax.set_title(_short(tract), fontsize=8.5, color=TEXT, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=7, length=2)
    for ax in axes.flat[len(tracts):]:
        ax.set_visible(False)
    # x labels on each column's lowest visible panel (a figure-level label would collide with the legend below)
    for col in range(ncol):
        visible = [ax for ax in axes[:, col] if ax.get_visible()]
        if visible:
            visible[-1].set_xlabel("Age (years)", color=TEXT_2, fontsize=8)
            visible[-1].tick_params(labelbottom=True)

    handles = [Line2D([], [], color=colour, marker=marker, lw=1.5, markersize=6, markeredgecolor=SURFACE,
                      label=label) for _code, label, colour, marker in SEXES]
    handles += [Line2D([], [], color=TEXT_2, lw=1.5, label="50th centile"),
                Line2D([], [], color=TEXT_2, lw=0.9, ls=(0, (4, 2)), label="5th / 95th centile"),
                Line2D([], [], color=MUTED, marker="o", lw=0, markersize=6, markeredgecolor=TEXT,
                       markeredgewidth=1.4, label=f"|z_cal| > {FLAG_Z:g}")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=TEXT_2)
    fig.suptitle(title, color=TEXT, fontsize=11)
    fig.supylabel("FA (skeleton mean)", color=TEXT_2, fontsize=9)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
