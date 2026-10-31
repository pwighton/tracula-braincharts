"""QA metrics for ``extract``, written to ``provenance.json`` (``qa``). Values are reported, not pass/fail; ``qa.flags``
holds only the unambiguous problems (name -> detail), for ``collate`` and visual review.

Inputs beyond the FA map are optional: a metric whose input is missing is null, with a ``reason``.
"""

import re
import shlex
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import inputs

# eddy's EddyUtils::b_range (FSL 6.0.x source, src/eddy/EddyUtils.h): b <= 100 is b0, and b-values closer than this
# share a shell
EDDY_B_RANGE = 100
EDDY_SUFFIXES = {"movement_rms": ".eddy_movement_rms", "restricted_movement_rms": ".eddy_restricted_movement_rms",
                 "outlier_map": ".eddy_outlier_map", "cnr_maps": ".eddy_cnr_maps.nii.gz",
                 "command": ".eddy_command_txt"}


def _data(path: Path, dtype=np.float64) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=dtype).squeeze()


def _r(x) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _rel(subject_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(subject_dir))
    except ValueError:
        return str(path)


# --- native FA and the DWI it was fitted from ---

def native_qa(fa: Path, mask: Path | None, dwi: Path | None, bvals: np.ndarray) -> dict:
    """Brain-mask volume, FA outside [0, 1] or non-finite, median FA in white matter (FA > 0.2), and the b0 tSNR
    (mean / SD over the b=0 volumes of the dtifit input) in the brain mask."""
    img = nib.load(fa)
    data = np.asarray(img.dataobj, dtype=np.float64).squeeze()
    in_mask = _data(mask) > 0 if mask is not None else data != 0
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000
    vals = data[in_mask]
    finite = vals[np.isfinite(vals)]
    out = {"mask": "dtifit -m" if mask is not None else "FA != 0", "mask_volume_ml": round(in_mask.sum() * voxel_ml, 1),
           "nonfinite_voxels": int((~np.isfinite(data)).sum()),
           "fa_above_1_voxels": int((finite > 1).sum()),
           "fa_above_1_pct": _r(100 * (finite > 1).mean()) if len(finite) else None,
           "fa_below_0_voxels": int((finite < 0).sum()),
           "wm_median_fa": _r(np.median(finite[finite > 0.2])) if (finite > 0.2).any() else None}
    if dwi is None:
        out["b0_tsnr_median"], out["b0_tsnr_reason"] = None, "dtifit's input DWI not found"
        return out
    b0 = np.flatnonzero(bvals <= inputs.B0_MAX)
    if len(b0) < 3:
        out["b0_tsnr_median"], out["b0_tsnr_reason"] = None, f"{len(b0)} b=0 volumes (need 3)"
        return out
    vols = nib.load(dwi).get_fdata(dtype=np.float32)[..., b0][in_mask]
    sd = vols.std(axis=1, ddof=1)
    tsnr = np.divide(vols.mean(axis=1), sd, out=np.full(len(sd), np.nan, dtype=np.float32), where=sd > 0)
    out["b0_tsnr_median"], out["b0_volumes"] = _r(np.nanmedian(tsnr)), len(b0)
    return out


# --- eddy ---

def eddy_shells(bvals: np.ndarray) -> tuple[list[float], np.ndarray]:
    """eddy's shells (FSL 6.0.x ``EddyUtils::get_groups``): templates in acquisition order, one per b-value not
    within ``EDDY_B_RANGE`` of an earlier template; each shell's b is the mean over the volumes within range of its
    template; shells sorted by b. The first shell is b0 when its b <= ``EDDY_B_RANGE``. Returns (shell b-values,
    shell index per volume). eddy puts a volume in every shell within range, which matches two shells when they are
    exactly ``EDDY_B_RANGE`` apart (Connectome b=2300 and 2400); here each volume goes to the nearest shell."""
    templates = [0]
    for i in range(1, len(bvals)):
        if not any(abs(bvals[t] - bvals[i]) < EDDY_B_RANGE for t in templates):
            templates.append(i)
    means = sorted(float(bvals[np.abs(bvals - bvals[t]) < EDDY_B_RANGE].mean()) for t in templates)
    shell = np.abs(bvals[:, None] - np.array(means)[None, :]).argmin(axis=1)
    return means, shell


def find_eddy(subject_dir: Path) -> tuple[dict[str, Path], str | None]:
    """eddy's outputs (``eddy/results/<prefix>.eddy_*``) and the b-values it was given: the ``--bvals`` of
    ``eddy_command_txt``, found by its last two path components, else ``diff/*_diff.bval``."""
    found = sorted((subject_dir / "eddy" / "results").glob("*" + EDDY_SUFFIXES["movement_rms"]))
    if len(found) != 1:
        return {}, f"{len(found)} eddy/results/*{EDDY_SUFFIXES['movement_rms']} files (need 1)"
    prefix = str(found[0])[:-len(EDDY_SUFFIXES["movement_rms"])]
    files = {k: Path(prefix + s) for k, s in EDDY_SUFFIXES.items() if Path(prefix + s).is_file()}
    bval = None
    if "command" in files:
        for word in shlex.split(files["command"].read_text()):
            if word.startswith("--bvals="):
                candidate = subject_dir.joinpath(*Path(word.partition("=")[2]).parts[-2:])
                bval = candidate if candidate.is_file() else None
    if bval is None:
        diff = sorted((subject_dir / "diff").glob("*_diff.bval"))
        bval = diff[0] if len(diff) == 1 else None
    if bval is None:
        return files, "eddy's b-values not found (eddy_command_txt --bvals, diff/*_diff.bval)"
    files["bvals"] = bval
    return files, None


def _motion(rms: np.ndarray, idx: np.ndarray) -> dict:
    sub = rms[idx]
    return {"abs_mean_mm": _r(sub[:, 0].mean()), "abs_max_mm": _r(sub[:, 0].max()),
            "rel_mean_mm": _r(sub[:, 1].mean()), "rel_max_mm": _r(sub[:, 1].max())}


def eddy_qa(subject_dir: Path, fa_bvals: np.ndarray, shell_b: float | None) -> dict:
    """Motion (eddy's RMS displacement, absolute = vs the first volume, relative = vs the previous one), outlier
    slices, and eddy's b0 SNR and CNR maps, for all of eddy's volumes and for the FA's (b=0 + its shell). The FA's
    volumes are matched to eddy's by b-value, in order (the wrapper's ``dwiextract -shells`` keeps eddy's order)."""
    files, reason = find_eddy(subject_dir)
    if reason:
        return {"reason": reason}
    out = {"source": {k: _rel(subject_dir, p) for k, p in files.items()}}
    b = inputs.read_bvals(files["bvals"])
    n = len(b)
    fa_idx = None
    if shell_b is not None:
        sel = np.flatnonzero((b <= inputs.B0_MAX) | (np.abs(b - shell_b) <= inputs.SHELL_TOLERANCE))
        if np.array_equal(b[sel].round(), np.asarray(fa_bvals).round()):
            fa_idx = sel
        else:
            out["fa_volumes_reason"] = (f"the FA's {len(fa_bvals)} b-values don't match eddy's volumes at b=0 and "
                                        f"b={shell_b:.0f} ({len(sel)})")
    out["volumes"], out["fa_volumes"] = n, None if fa_idx is None else len(fa_idx)

    for key in ("movement_rms", "restricted_movement_rms"):
        if key in files:
            rms = np.loadtxt(files[key], ndmin=2)
            if rms.shape != (n, 2):
                out[key] = {"reason": f"{rms.shape[0]} rows, {n} b-values"}
                continue
            out[key] = {"all": _motion(rms, np.arange(n))}
            if fa_idx is not None:
                out[key]["fa_volumes"] = _motion(rms, fa_idx)

    if "outlier_map" in files:
        lines = files["outlier_map"].read_text().splitlines()
        omap = np.loadtxt(lines[1:], ndmin=2)  # first line is a header
        if omap.shape[0] != n:
            out["outliers"] = {"reason": f"{omap.shape[0]} rows, {n} b-values"}
        else:
            dwi = b > EDDY_B_RANGE  # eddy only tests diffusion-weighted volumes
            out["outliers"] = {"slices_per_volume": omap.shape[1],
                               "all_dwi_slices_pct": _r(100 * omap[dwi].mean()) if dwi.any() else None,
                               "all_dwi_volumes_with_outliers": int(omap[dwi].any(axis=1).sum())}
            if fa_idx is not None:
                fa_dwi = fa_idx[b[fa_idx] > EDDY_B_RANGE]
                out["outliers"]["fa_dwi_slices_pct"] = _r(100 * omap[fa_dwi].mean()) if len(fa_dwi) else None
                out["outliers"]["fa_dwi_volumes_with_outliers"] = int(omap[fa_dwi].any(axis=1).sum())

    if "cnr_maps" in files:
        out["cnr"] = _cnr(files["cnr_maps"], b, fa_idx, shell_b)
    return out


def _cnr(path: Path, b: np.ndarray, fa_idx: np.ndarray | None, shell_b: float | None) -> dict:
    """``eddy --cnr_maps``: volume 0 is the b0 SNR, then one CNR map per diffusion shell in ascending b (FSL 6.0.x
    ``WriteCNRMaps``). Medians over the voxels eddy computed (non-zero b0 SNR, i.e. its mask)."""
    means, shell = eddy_shells(b)
    dwi_shells = means[1:] if means[0] <= EDDY_B_RANGE else means
    img = nib.load(path)
    if img.ndim != 4 or img.shape[3] != len(dwi_shells) + 1:
        return {"reason": f"{img.shape[3] if img.ndim == 4 else 1} volumes, expected {len(dwi_shells) + 1} "
                          f"(b0 + {len(dwi_shells)} eddy shells)"}
    snr = np.asarray(img.dataobj[..., 0], dtype=np.float64)
    mask = snr > 0
    # eddy's b0 group is every volume with b <= 100 (here also b=50), so its "b0 SNR" isn't a b=0 tSNR
    out = {"eddy_shells": [round(x, 1) for x in dwi_shells], "b0_group_mean_b": round(means[0], 1),
           "b0_group_snr_median": _r(np.median(snr[mask]))}
    if fa_idx is None or shell_b is None:
        return out | {"shell_cnr_reason": "the FA's volumes weren't matched to eddy's"}
    fa_dwi = fa_idx[b[fa_idx] > EDDY_B_RANGE]
    groups = set(shell[fa_dwi])
    if len(groups) != 1:
        return out | {"shell_cnr_reason": f"the FA's shell spans {len(groups)} eddy shells"}
    j = groups.pop() - (len(means) - len(dwi_shells))
    cnr = np.asarray(img.dataobj[..., j + 1], dtype=np.float64)
    return out | {"shell_b": round(dwi_shells[j], 1), "shell_cnr_median": _r(np.median(cnr[mask]))}


# --- registration and skeleton (TBSS outputs, MNI152 1 mm) ---

def _one(pattern: str, work: Path) -> Path:
    found = sorted(work.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one {pattern} in {work}, found {len(found)}")
    return found[0]


def parse_avscale(text: str) -> dict:
    """``avscale --allparams``: rotations (rad), translations (mm), scales, skews, average scaling, determinant."""
    def nums(label: str) -> list[float] | None:
        m = re.search(re.escape(label) + r"[^=\n]*=\s*([-0-9.eE+ \t]+)", text)
        return [float(x) for x in m.group(1).split()] if m else None

    return {"rotations_rad": nums("Rotation Angles"), "translations_mm": nums("Translations"),
            "scales": nums("Scales"), "skews": nums("Skews"),
            "average_scaling": (nums("Average scaling") or [None])[0],
            "determinant": (nums("Determinant") or [None])[0]}


def registration_qa(work: Path, fsldir: Path, run: Callable[[list[str], Path], str]) -> dict:
    """Warped FA vs FMRIB58_FA (Pearson r in the template brain and in its white matter, FA > 0.2), the template
    brain's coverage by the subject's warped FA and their Dice, FLIRT's affine (``avscale``), and the Jacobian
    determinant of the whole warp, FLIRT + FNIRT (``fnirtfileutils -j -a``), in the template brain. FSL's warps map
    template to subject, so J > 1 where the subject is larger than the template; J <= 0 is folding."""
    template = _data(fsldir / "data/standard/FMRIB58_FA_1mm.nii.gz") / 10000
    warped = _data(work / "stats" / "all_FA.nii.gz")
    subj = _data(work / "stats" / "mean_FA_mask.nii.gz") > 0
    brain, wm = template > 0, template > 0.2
    both = subj & brain
    out = {"r_brain": _r(np.corrcoef(warped[brain], template[brain])[0, 1]),
           "r_wm": _r(np.corrcoef(warped[wm], template[wm])[0, 1]),
           "template_brain_coverage_pct": _r(100 * both.sum() / brain.sum()),
           "dice": _r(2 * both.sum() / (subj.sum() + brain.sum()))}

    mat = _one("FA/*_FA_to_target.mat", work)
    out["flirt"] = parse_avscale(run(["avscale", "--allparams", str(mat), str(work / "FA" / "target")], work))

    coef = _one("FA/*_FA_to_target_warp.nii.gz", work)
    jac = coef.with_name(coef.name.replace("_warp.nii.gz", "_jac.nii.gz"))
    run(["fnirtfileutils", "-i", str(coef), "-r", str(work / "FA" / "target"), "-j", str(jac), "-a"], work)
    j = _data(jac)[brain]
    out["jacobian"] = {"min": _r(j.min()), "p01": _r(np.percentile(j, 1)), "median": _r(np.median(j)),
                       "p99": _r(np.percentile(j, 99)), "max": _r(j.max()), "nonpositive_voxels": int((j <= 0).sum()),
                       "file": _rel(work, jac)}
    return out


def skeleton_qa(work: Path, threshold: float) -> dict:
    """Skeleton voxels; projected FA that is 0 (nothing found) or below the threshold; the projection gain (projected
    minus warped FA at the same voxel, a proxy for how far the search moved)."""
    skel = _data(work / "stats" / "mean_FA_skeleton_mask.nii.gz") > 0
    proj = _data(work / "stats" / "all_FA_skeletonised.nii.gz")[skel]
    warped = _data(work / "stats" / "all_FA.nii.gz")[skel]
    found = proj != 0
    return {"voxels": int(skel.sum()), "zero_pct": _r(100 * (~found).mean()),
            "below_threshold_pct": _r(100 * (proj[found] < threshold).mean()) if found.any() else None,
            "projection_gain_mean": _r((proj - warped)[found].mean()) if found.any() else None,
            "projection_gain_p95": _r(np.percentile((proj - warped)[found], 95)) if found.any() else None}


def flags(native: dict, registration: dict, labels: pd.DataFrame) -> dict[str, str]:
    """The unambiguous problems, name -> detail: non-finite FA, a folding warp (J <= 0), labels without a value,
    labels whose skeleton has voxels without projected FA, and labels whose template skeleton lies partly outside
    the subject's warped FA (cut-off field of view)."""
    out = {}
    if native["nonfinite_voxels"]:
        out["fa_nonfinite"] = f"{native['nonfinite_voxels']} NaN/inf voxels in the FA map"
    if registration["jacobian"]["nonpositive_voxels"]:
        out["jacobian_nonpositive"] = (f"{registration['jacobian']['nonpositive_voxels']} voxels with J <= 0 "
                                       "(folding) in the warp")

    def listed(rows, column):
        return ", ".join(f"{t.removeprefix('FA_mean_of_')} ({v:.1f}%)" for t, v in zip(rows["tract"], rows[column]))

    none = labels[labels["nonzero_voxels"] == 0]
    if len(none):
        out["label_no_value"] = f"{len(none)} label(s): " + ", ".join(t.removeprefix("FA_mean_of_")
                                                                    for t in none["tract"])
    partial = labels[(labels["coverage_pct"] < 100) & (labels["nonzero_voxels"] > 0)]
    if len(partial):
        out["label_coverage"] = f"{len(partial)} label(s) with skeleton voxels without FA: " + listed(
            partial, "coverage_pct")
    cut = labels[labels["template_in_mask_pct"] < 100]
    if len(cut):
        out["skeleton_outside_fa"] = f"{len(cut)} label(s) with template skeleton outside the warped FA: " + listed(
            cut, "template_in_mask_pct")
    return out
