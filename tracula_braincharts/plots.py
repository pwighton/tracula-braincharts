"""QA figures (static PNG)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Light chart tokens and the first two categorical slots of the dataviz reference palette (validated all-pairs:
# CVD dE 24.7, normal-vision dE 33.6). Sex is also encoded by marker shape, so identity isn't colour alone.
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEXES = [(1, "Male", "#2a78d6", "o"), (0, "Female", "#eb6834", "^")]  # (code, label, colour, marker), fixed order
FLAG_Z = 2.0

# Slice images: dark background, as image viewers show them. The template outline and "no FA" use the palette's
# orange and blue; projected FA one sequential ramp, red (0.2) to yellow (0.8), as TBSS skeletons are usually shown.
IMAGE_BG, IMAGE_TEXT, IMAGE_TEXT_2, BOUNDARY = "#111111", "#f2f1ec", "#c3c2b7", "#d9d8d0"
OUTLINE, NO_FA = "#eb6834", "#2a78d6"
SKELETON_CMAP = "YlOrRd_r"
FA_RANGE = (0.0, 0.8)


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


def _slice(vol: np.ndarray, affine: np.ndarray, axis: int, mm: float) -> np.ndarray:
    """The slice of ``vol`` nearest to world coordinate ``mm`` along ``axis`` (0 x, 1 y, 2 z), transposed for
    ``imshow(origin="lower")``."""
    i = int(round((mm - affine[axis, 3]) / affine[axis, axis]))
    return np.take(vol, min(max(i, 0), vol.shape[axis] - 1), axis=axis).T


def _image_axes(nrow: int, ncol: int, width: float, height: float, title: str):
    fig, axes = plt.subplots(nrow, ncol, figsize=(width * ncol, height * nrow + 0.7), squeeze=False,
                             facecolor=IMAGE_BG, layout="constrained")
    for ax in axes.flat:
        ax.set_facecolor(IMAGE_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle(title, color=IMAGE_TEXT, fontsize=10)
    return fig, axes


def _caption(ax, text: str, affine: np.ndarray | None = None) -> None:
    """Panel title; on axial panels (``affine`` given) also patient R/L (columns run along voxel x, which FSL's
    MNI152 stores right to left)."""
    ax.set_title(text, color=IMAGE_TEXT_2, fontsize=8, loc="left")
    if affine is not None:
        left, right = ("R", "L") if affine[0, 0] < 0 else ("L", "R")
        for x, side, ha in ((0.02, left, "left"), (0.98, right, "right")):
            ax.text(x, 0.02, side, transform=ax.transAxes, color=IMAGE_TEXT_2, fontsize=8, ha=ha, va="bottom")


def _save_image(fig, handles: list, path: Path) -> None:
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=8,
               labelcolor=IMAGE_TEXT_2)
    fig.savefig(path, dpi=130, facecolor=IMAGE_BG)
    plt.close(fig)


AXIAL_MM = [-30, -15, 0, 15, 30, 45]
OTHER_MM = [(1, -60.0), (1, -20.0), (1, 20.0), (0, -30.0), (0, 0.0), (0, 30.0)]  # (axis, mm): 3 coronal, 3 sagittal
SKELETON_AXIAL_MM = [-35, -20, -5, 10, 20, 30, 40, 50]


def registration_panels(warped: np.ndarray, template: np.ndarray, affine: np.ndarray, title: str,
                        path: Path) -> None:
    """The subject's warped FA with the template's white-matter outline (FMRIB58_FA = 0.2): 6 axial slices, then
    3 coronal and 3 sagittal."""
    fig, axes = _image_axes(2, len(AXIAL_MM), 2.0, 2.3, title)
    for ax, (axis, mm) in zip(axes.flat, [(2, float(z)) for z in AXIAL_MM] + OTHER_MM):
        ax.imshow(_slice(warped, affine, axis, mm), cmap="gray", vmin=FA_RANGE[0], vmax=FA_RANGE[1],
                  origin="lower", interpolation="nearest")
        ax.contour(_slice(template, affine, axis, mm), levels=[0.2], colors=OUTLINE, linewidths=0.6)
        _caption(ax, f"{'xyz'[axis]} = {mm:g} mm", affine if axis == 2 else None)
    _save_image(fig, [Line2D([], [], color=OUTLINE, lw=1.2, label="FMRIB58_FA = 0.2 (template white matter)")],
                path)


def _boundaries(labels: np.ndarray) -> np.ndarray:
    """In-plane label boundaries: labelled pixels whose right or upper neighbour has another label."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return edge & (labels > 0)


def skeleton_panels(warped: np.ndarray, skeleton: np.ndarray, projected: np.ndarray, labels: np.ndarray,
                    affine: np.ndarray, title: str, path: Path) -> None:
    """8 axial slices: the subject's warped FA (dimmed), JHU label boundaries, and the skeleton coloured by its
    projected FA, blue where nothing was projected."""
    fig, axes = _image_axes(2, 4, 2.6, 2.9, title)
    image = None
    for ax, z in zip(axes.flat, SKELETON_AXIAL_MM):
        ax.imshow(_slice(warped, affine, 2, z), cmap="gray", vmin=FA_RANGE[0], vmax=FA_RANGE[1] * 1.6,
                  origin="lower", interpolation="nearest")
        edges = _boundaries(_slice(labels, affine, 2, z))
        ax.imshow(np.ma.masked_where(~edges, edges), cmap=ListedColormap([BOUNDARY]), alpha=0.45, origin="lower",
                  interpolation="nearest")
        skel, proj = _slice(skeleton, affine, 2, z) > 0, _slice(projected, affine, 2, z)
        image = ax.imshow(np.ma.masked_where(~skel | (proj == 0), proj), cmap=SKELETON_CMAP, vmin=0.2,
                          vmax=FA_RANGE[1], origin="lower", interpolation="nearest")
        missing = skel & (proj == 0)
        ax.imshow(np.ma.masked_where(~missing, missing), cmap=ListedColormap([NO_FA]), origin="lower",
                  interpolation="nearest")
        _caption(ax, f"z = {z:g} mm", affine)
    bar = fig.colorbar(image, ax=axes, location="right", shrink=0.6, pad=0.01)
    bar.set_label("projected FA", color=IMAGE_TEXT_2, fontsize=8)
    bar.ax.tick_params(colors=IMAGE_TEXT_2, labelsize=7)
    bar.outline.set_visible(False)
    _save_image(fig, [Line2D([], [], color=BOUNDARY, lw=1.2, label="JHU label boundaries"),
                      Line2D([], [], color=NO_FA, marker="s", lw=0, markersize=5, label="skeleton, no FA projected")],
                path)
