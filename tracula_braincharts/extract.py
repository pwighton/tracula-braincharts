"""The ``extract`` subcommand: TBSS on one subject's FA map, then mean skeleton FA per JHU label.

As in the training data (``fa_normative_modeling/01.diffusion_processing.ipynb``, FSL 6.0.5): ``tbss_1_preproc`` ->
``tbss_2_reg -T`` (FLIRT + FNIRT to FMRIB58_FA_1mm) -> ``tbss_3_postreg -T`` (FMRIB58 skeleton; ``-S`` with
``--skeleton subject``) -> ``tbss_4_prestats 0.2``, then ``fslstats -K JHU-ICBM-labels-1mm all_FA_skeletonised -M``.
The skeletonised FA is on the JHU atlas's 1 mm MNI152 grid, so ``fslstats -K`` needs no resampling (checked).
The atlas's labels are matched to the model's 48 tracts by name: FSL 6.0.5's atlas has 50 labels, and the training
data's means fit its labels 1-44 and 47-50 (the inferior fronto-occipital fasciculus, 45/46, isn't in the model).
QA metrics (``qa.py``) go into ``provenance.json``, with two images in ``qc/``.
"""

import logging
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from . import inputs, models, plots, provenance, qa
from .cli_errors import CliError
from .config import SKELETONS

log = logging.getLogger(__name__)

SKELETON_THRESHOLD = 0.2

# FSL files, relative to $FSLDIR
JHU_LABELS = "data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz"
JHU_XML = "data/atlases/JHU-labels.xml"
TEMPLATES = ["data/standard/FMRIB58_FA_1mm.nii.gz", "data/standard/FMRIB58_FA-skeleton_1mm.nii.gz"]

LABEL_QC_COLUMNS = ["tract", "label", "atlas_voxels", "skeleton_voxels", "nonzero_voxels", "coverage_pct",
                    "template_skeleton_voxels", "template_in_mask_pct", "fa_mean", "fa_sd", "fa_min", "fa_max"]


def fsl_dir() -> Path:
    fsldir = os.environ.get("FSLDIR")
    if not fsldir or shutil.which("tbss_1_preproc") is None:
        raise CliError("FSL not found (FSLDIR unset or tbss_1_preproc not on PATH): run extract inside the container")
    return Path(fsldir)


def fsl_version(fsldir: Path) -> str | None:
    path = fsldir / "etc" / "fslversion"
    return path.read_text().strip() if path.is_file() else None


def run(cmd: list[str], cwd: Path) -> str:
    """Run an FSL command, logging it and its output; raise on failure."""
    log.info("extract: $ %s", " ".join(cmd))
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    for stream in (proc.stdout, proc.stderr):
        for line in stream.splitlines():
            if line.strip():
                log.info("extract:   %s", line)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {proc.returncode}) in {cwd}; see the log")
    log.info("extract:   (%s took %.0f s)", cmd[0], time.monotonic() - t0)
    return proc.stdout


def jhu_label_names(xml_path: Path) -> dict[int, str]:
    """Label value -> name from FSL's atlas XML (``<label index="i">name</label>``; index = value in the image)."""
    root = ET.parse(xml_path).getroot()
    return {int(lab.get("index")): (lab.text or "").strip() for lab in root.iter("label")}


def _tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def match_labels(names: dict[int, str], tracts: list[str]) -> dict[str, int]:
    """Template tract -> JHU label value, by name: the one atlas label whose name contains every word of the tract's
    name, on the same side (R/L). FSL's numbering changes between versions: FSL 6.0.5's atlas has 50 labels, with the
    inferior fronto-occipital fasciculus (which the model doesn't have) at 45/46, before the uncinate and tapetum."""
    def side(tokens: list[str]) -> str | None:
        return tokens[-1] if tokens and tokens[-1] in ("r", "l") else None

    labels = {value: _tokens(name) for value, name in names.items() if value > 0}
    mapping = {}
    for tract in tracts:
        want = _tokens(tract.removeprefix("FA_mean_of_"))
        found = [v for v, have in labels.items() if set(want) <= set(have) and side(want) == side(have)]
        if len(found) != 1:
            raise CliError(f"{tract} matches {len(found)} JHU labels ({', '.join(repr(names[v]) for v in found)}), "
                           "expected 1 (atlas names changed?)")
        mapping[tract] = found[0]
    return mapping


def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def label_qc(skeletonised: Path, skeleton_mask: Path, labels: Path, mapping: dict[str, int],
             fa_mask: Path | None = None, template_skeleton: Path | None = None) -> pd.DataFrame:
    """Per label: atlas voxels, skeleton voxels, those with projected FA > 0 (coverage), and FA statistics. With
    ``fa_mask`` (TBSS's ``mean_FA_mask``) and ``template_skeleton`` (FMRIB58's): the % of the label's template skeleton
    inside the subject's warped FA, which shows a cut-off field of view in both skeleton modes (with ``-T`` the
    skeleton mask is the template's whatever the subject covers)."""
    skel_img, mask_img, lab_img = nib.load(skeletonised), nib.load(skeleton_mask), nib.load(labels)
    others = [("skeletonised FA", skel_img), ("skeleton mask", mask_img)]
    others += [(name, nib.load(p)) for name, p in (("FA mask", fa_mask), ("template skeleton", template_skeleton))
               if p is not None]
    for name, img in others:
        if not _same_grid(img, lab_img):
            raise CliError(f"{name} is not on the JHU atlas grid ({img.shape}, {lab_img.shape}): cannot use -K")
    fa = np.asarray(skel_img.dataobj, dtype=float).squeeze()
    on_skeleton = np.asarray(mask_img.dataobj).squeeze() > 0
    lab = np.asarray(lab_img.dataobj).astype(int)
    in_fa = np.asarray(nib.load(fa_mask).dataobj).squeeze() > 0 if fa_mask is not None else None
    # FMRIB58 values are FA x 10000; tbss_4_prestats -T thresholds the raw values, as here
    tskel = (np.asarray(nib.load(template_skeleton).dataobj).squeeze() > SKELETON_THRESHOLD
             if template_skeleton is not None else None)
    rows = []
    for tract, value in mapping.items():
        in_label = lab == value
        skel = in_label & on_skeleton
        vals = fa[in_label & (fa != 0)]
        n_skel = int(skel.sum())
        row = {"tract": tract, "label": value, "atlas_voxels": int(in_label.sum()), "skeleton_voxels": n_skel,
               "nonzero_voxels": len(vals),
               "coverage_pct": 100 * int((skel & (fa != 0)).sum()) / n_skel if n_skel else np.nan,
               "template_skeleton_voxels": np.nan, "template_in_mask_pct": np.nan,
               "fa_mean": vals.mean() if len(vals) else np.nan, "fa_sd": vals.std() if len(vals) else np.nan,
               "fa_min": vals.min() if len(vals) else np.nan, "fa_max": vals.max() if len(vals) else np.nan}
        if tskel is not None and in_fa is not None:
            t = tskel & in_label
            row["template_skeleton_voxels"] = int(t.sum())
            row["template_in_mask_pct"] = 100 * int((t & in_fa).sum()) / int(t.sum()) if t.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=LABEL_QC_COLUMNS)


def fslstats_means(skeletonised: Path, labels: Path, values: list[int], cwd: Path) -> np.ndarray:
    """``fslstats -K labels skeletonised -M``, as the training pipeline ran it: mean of non-zero voxels for each label
    value 1..max, of which ``values`` are returned in order. A label with no non-zero voxels gives 0 (or nan),
    returned as NaN."""
    out = run(["fslstats", "-K", str(labels), str(skeletonised), "-M"], cwd)
    means = np.array([float(v) for v in out.split()])
    if len(means) < max(values):
        raise RuntimeError(f"fslstats -K returned {len(means)} values, expected at least {max(values)}")
    means = means[np.asarray(values) - 1]
    means[means == 0] = np.nan
    return means


def output_dir(inp: inputs.Inputs, out: Path | None) -> Path:
    """``<subject_dir>/fa-normative`` by default, or ``<out>/<subject>``."""
    return inp.subject_dir / "fa-normative" if out is None else Path(out) / inp.subject


def run_extract(inp: inputs.Inputs, out_dir: Path, skeleton: str = "fmrib58", allow_shells: bool = False) -> None:
    """Run TBSS and write ``features.json``, ``label_qc.csv``, ``provenance.json`` (with the QA metrics, ``qa``),
    ``qc/*.png`` and ``tbss/`` to ``out_dir``."""
    fsldir = fsl_dir()
    log.info("extract: subject %s (%s); FA %s; b-values %s (found via %s%s)", inp.subject, inp.subject_dir, inp.fa,
             inp.bval, inp.method, f", scripts/log.txt line {inp.log_line}" if inp.log_line else "")
    warnings = []

    # Acquisition checks
    bvals = inputs.read_bvals(inp.bval)
    sh = inputs.shells(bvals)
    warnings += [f"acquisition: {p}" for p in inputs.check_shells(sh, allow=allow_shells)]
    fa_img = nib.load(inp.fa)
    if inp.dwi is not None:
        n_vol = nib.load(inp.dwi).shape[3] if len(nib.load(inp.dwi).shape) == 4 else 1
        if n_vol != len(bvals):
            raise CliError(f"{inp.bval.name} has {len(bvals)} b-values but {inp.dwi.name} has {n_vol} volumes")
    if inp.mask is not None and not _same_grid(fa_img, nib.load(inp.mask)):
        raise CliError(f"FA map and brain mask {inp.mask.name} are on different grids")
    delta = inputs.diffusion_time(inp.subject_dir, sh.shells[0]["b"]) if len(sh.shells) == 1 else \
        {"delta_ms": None, "reason": "not exactly one shell"}
    log.info("extract: b=0 volumes %d; shells %s; Delta %s ms", sh.n_b0,
             ", ".join(f"b={s['b']:.0f} ({s['n']} vols)" for s in sh.shells), delta["delta_ms"])

    # QA of the inputs: native FA, its DWI, and eddy's outputs when present
    shell_b = sh.shells[0]["b"] if len(sh.shells) == 1 else None
    qa_native = qa.native_qa(inp.fa, inp.mask, inp.dwi, bvals)
    log.info("extract: QA FA: brain mask %s ml, FA > 1 in %d voxels, white-matter median FA %s, b0 tSNR %s",
             qa_native["mask_volume_ml"], qa_native["fa_above_1_voxels"], qa_native["wm_median_fa"],
             qa_native["b0_tsnr_median"])
    qa_eddy = qa.eddy_qa(inp.subject_dir, bvals, shell_b)
    if "reason" in qa_eddy:
        log.info("extract: QA eddy: %s", qa_eddy["reason"])
    else:
        motion = qa_eddy.get("movement_rms", {})
        log.info("extract: QA eddy: %d volumes (FA %s); motion abs/rel mean %s / %s mm (FA volumes %s / %s); "
                 "outlier slices %s%%; b=%s CNR %s", qa_eddy["volumes"], qa_eddy["fa_volumes"],
                 motion.get("all", {}).get("abs_mean_mm"), motion.get("all", {}).get("rel_mean_mm"),
                 motion.get("fa_volumes", {}).get("abs_mean_mm"), motion.get("fa_volumes", {}).get("rel_mean_mm"),
                 qa_eddy.get("outliers", {}).get("all_dwi_slices_pct"), qa_eddy.get("cnr", {}).get("shell_b"),
                 qa_eddy.get("cnr", {}).get("shell_cnr_median"))

    # JHU labels vs the template's tract columns
    tracts = models.template_columns()[4:]
    names = jhu_label_names(fsldir / JHU_XML)
    mapping = match_labels(names, tracts)
    unused = {v: n for v, n in names.items() if v > 0 and v not in mapping.values()}
    log.info("extract: JHU labels not in the model: %s",
             "; ".join(f"{v} {n}" for v, n in unused.items()) or "none")

    # TBSS, in a fresh working directory
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "tbss"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    fa_name = f"{inp.subject}_FA.nii.gz"
    shutil.copyfile(inp.fa, work / fa_name)
    run(["tbss_1_preproc", fa_name], work)
    run(["tbss_2_reg", "-T"], work)
    run(["tbss_3_postreg", SKELETONS[skeleton]], work)
    run(["tbss_4_prestats", str(SKELETON_THRESHOLD)], work)
    skeletonised = work / "stats" / "all_FA_skeletonised.nii.gz"
    skeleton_mask = work / "stats" / "mean_FA_skeleton_mask.nii.gz"

    # Features: fslstats -K as in training, cross-checked against our own per-label statistics
    labels = fsldir / JHU_LABELS
    qc = label_qc(skeletonised, skeleton_mask, labels, mapping, work / "stats" / "mean_FA_mask.nii.gz",
                  fsldir / TEMPLATES[1])
    means = fslstats_means(skeletonised, labels, [mapping[t] for t in tracts], work)
    if not np.allclose(means, qc["fa_mean"].to_numpy(), atol=1e-5, equal_nan=True):
        raise RuntimeError("fslstats -K means differ from the per-label means computed from the image")
    missing = [t for t, v in zip(tracts, means) if np.isnan(v)]

    # QA of registration and projection, flags, images
    qa_reg = qa.registration_qa(work, fsldir, run)
    qa_skel = qa.skeleton_qa(work, SKELETON_THRESHOLD)
    jac = qa_reg["jacobian"]
    log.info("extract: QA registration: r %s (white matter %s); template brain covered %s%%; Jacobian %s-%s "
             "(%d <= 0); FLIRT average scaling %s", qa_reg["r_brain"], qa_reg["r_wm"],
             qa_reg["template_brain_coverage_pct"], jac["min"], jac["max"], jac["nonpositive_voxels"],
             qa_reg["flirt"]["average_scaling"])
    log.info("extract: QA skeleton: %d voxels; %s%% without FA, %s%% of the rest below %g; mean projection gain %s",
             qa_skel["voxels"], qa_skel["zero_pct"], qa_skel["below_threshold_pct"], SKELETON_THRESHOLD,
             qa_skel["projection_gain_mean"])
    qa_flags = qa.flags(qa_native, qa_reg, qc)
    warnings += [f"QA {name}: {detail}" for name, detail in qa_flags.items()]
    for w in warnings:
        log.warning("extract: %s", w)
    images = qc_images(inp.subject, work, fsldir, qa_reg, qa_skel, skeleton, out_dir / "qc")

    features = {t: (None if np.isnan(v) else round(float(v), 6)) for t, v in zip(tracts, means)}
    provenance.write({"subject": inp.subject, "skeleton": skeleton, "features": features}, out_dir / "features.json")
    qc.to_csv(out_dir / "label_qc.csv", index=False)

    record = provenance.base_record("extract", None)
    record["subject"] = {"id": inp.subject, "dir": str(inp.subject_dir)}
    record["inputs"] = {"method": inp.method, "log_line": inp.log_line,
                        **{k: provenance.file_info(p) if p is not None else None
                           for k, p in (("fa", inp.fa), ("bval", inp.bval), ("dwi", inp.dwi), ("mask", inp.mask))}}
    record["acquisition"] = {"n_b0": sh.n_b0, "shells": sh.shells, "diffusion_time": delta,
                             "fa_shape": list(fa_img.shape), "fa_voxel_mm": [round(float(z), 4) for z in
                                                                             fa_img.header.get_zooms()[:3]]}
    record["tbss"] = {"fsl_version": fsl_version(fsldir), "fsldir": str(fsldir), "skeleton": skeleton,
                      "tbss_3_postreg": SKELETONS[skeleton], "threshold": SKELETON_THRESHOLD,
                      "jhu_labels_not_used": {str(v): n for v, n in unused.items()},
                      "files": {rel: provenance.file_info(fsldir / rel) for rel in [JHU_LABELS, JHU_XML, *TEMPLATES]}}
    record["qa"] = {"fa": qa_native, "eddy": qa_eddy, "registration": qa_reg, "skeleton": qa_skel, "flags": qa_flags,
                    "images": [str(p.relative_to(out_dir)) for p in images]}
    record["warnings"] = warnings
    provenance.write(record, out_dir / "provenance.json")
    log.info("extract: %d of %d tracts have a value; %d QA flag(s); wrote features.json, label_qc.csv, "
             "provenance.json, qc/, tbss/ to %s", len(tracts) - len(missing), len(tracts), len(qa_flags), out_dir)


def qc_images(subject: str, work: Path, fsldir: Path, qa_reg: dict, qa_skel: dict, skeleton: str,
              qc_dir: Path) -> list[Path]:
    """``qc/registration.png`` (warped FA with the template's white-matter outline) and ``qc/skeleton.png``
    (projected skeleton FA over the JHU label boundaries)."""
    qc_dir.mkdir(exist_ok=True)
    template = nib.load(fsldir / TEMPLATES[0])
    affine = template.affine

    def data(path):
        return np.asarray(nib.load(path).dataobj, dtype=float).squeeze()

    def fmt(x, spec):
        return "n/a" if x is None else format(x, spec)

    warped = data(work / "stats" / "all_FA.nii.gz")
    paths = [qc_dir / "registration.png", qc_dir / "skeleton.png"]
    plots.registration_panels(
        warped, data(fsldir / TEMPLATES[0]) / 10000, affine,
        f"{subject}: warped FA and the FMRIB58 white-matter outline (r {fmt(qa_reg['r_wm'], '.3f')} in white "
        f"matter, template brain {fmt(qa_reg['template_brain_coverage_pct'], '.1f')}% covered)", paths[0])
    plots.skeleton_panels(
        warped, data(work / "stats" / "mean_FA_skeleton_mask.nii.gz"),
        data(work / "stats" / "all_FA_skeletonised.nii.gz"), data(fsldir / JHU_LABELS), affine,
        f"{subject}: skeleton FA ({skeleton} skeleton; {fmt(qa_skel['zero_pct'], '.2f')}% without FA, "
        f"{fmt(qa_skel['below_threshold_pct'], '.1f')}% below {SKELETON_THRESHOLD:g})", paths[1])
    return paths
