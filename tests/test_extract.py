"""extract: JHU label matching, per-label statistics, and (with FSL, in the container) the real atlas files and
the whole TBSS chain."""

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tracula_braincharts import extract, models
from tracula_braincharts.cli_errors import CliError

TRACTS = models.template_columns()[4:]

requires_fsl = pytest.mark.skipif(shutil.which("fslstats") is None or not os.environ.get("FSLDIR"),
                                  reason="needs FSL (run inside the container)")


IFOF = {45: "Inferior fronto-occipital fasciculus R", 46: "Inferior fronto-occipital fasciculus L"}


def _atlas_names(ifof: bool = True) -> dict[int, str]:
    """JHU-labels.xml names as FSL has them (used when FSL isn't available): 50 labels in FSL 6.0.5, with the
    inferior fronto-occipital fasciculus at 45/46; without it (``ifof=False``), the 48 template tracts in order."""
    names = [t.removeprefix("FA_mean_of_").replace("_", " ").replace("-", " (") for t in TRACTS]
    names[TRACTS.index("FA_mean_of_Fornix_cres-Stria_terminalis_R")] = \
        "Fornix (cres) / Stria terminalis (can not be resolved with current resolution) R"
    if ifof:
        names[44:44] = IFOF.values()
    return {0: "Unclassified", **{i + 1: n for i, n in enumerate(names)}}


def _expected(ifof: bool = True) -> dict[str, int]:
    return {t: i + 1 + (2 if ifof and i >= 44 else 0) for i, t in enumerate(TRACTS)}


@pytest.mark.parametrize("ifof", [True, False])
def test_labels_match_template_by_name(ifof):
    names = _atlas_names(ifof)
    assert len(names) == (51 if ifof else 49)
    mapping = extract.match_labels(names, TRACTS)
    assert mapping == _expected(ifof)
    assert mapping["FA_mean_of_Uncinate_fasciculus_R"] == (47 if ifof else 45)
    assert set(names) - {0} - set(mapping.values()) == (set(IFOF) if ifof else set())


@pytest.mark.parametrize("change, message", [
    (lambda n: n.pop(50), "Tapetum_L matches 0 JHU labels"),
    (lambda n: n.update({3: "Body of corpus callosum"}), "Genu_of_corpus_callosum matches 0"),
    (lambda n: n.update({7: "Corticospinal tract L"}), "Corticospinal_tract_R matches 0"),
    (lambda n: n.update({51: "Uncinate fasciculus (copy) R"}), "Uncinate_fasciculus_R matches 2"),
])
def test_shifted_or_wrong_labels_are_rejected(change, message):
    names = _atlas_names()
    change(names)
    with pytest.raises(CliError, match=message):
        extract.match_labels(names, TRACTS)


def _img(data, path):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)
    return path


def test_label_qc(tmp_path):
    lab = np.zeros((4, 4, 4))
    lab[0, :, :] = 1  # 16 voxels
    lab[1, :, :] = 2
    mask = np.zeros_like(lab)
    mask[0, 0, :] = 1  # 4 skeleton voxels in label 1
    mask[1, 0, :2] = 1  # 2 in label 2
    fa = np.zeros_like(lab)
    fa[0, 0, :] = [0.4, 0.5, 0.6, 0]  # one skeleton voxel without FA
    qc = extract.label_qc(_img(fa, tmp_path / "fa.nii.gz"), _img(mask, tmp_path / "m.nii.gz"),
                          _img(lab, tmp_path / "lab.nii.gz"), {"A": 1, "B": 2})
    a, b = qc.set_index("tract").loc["A"], qc.set_index("tract").loc["B"]
    assert (a["atlas_voxels"], a["skeleton_voxels"], a["nonzero_voxels"], a["coverage_pct"]) == (16, 4, 3, 75.0)
    assert np.isclose(a["fa_mean"], 0.5) and a["fa_min"] == pytest.approx(0.4) and a["fa_max"] == pytest.approx(0.6)
    assert b["nonzero_voxels"] == 0 and b["coverage_pct"] == 0 and np.isnan(b["fa_mean"])


def test_label_qc_rejects_other_grids(tmp_path):
    lab = _img(np.ones((4, 4, 4)), tmp_path / "lab.nii.gz")
    fa = _img(np.ones((4, 4, 5)), tmp_path / "fa.nii.gz")
    with pytest.raises(CliError, match="not on the JHU atlas grid"):
        extract.label_qc(fa, lab, lab, {"A": 1})


@requires_fsl
def test_fsl_jhu_atlas_matches_template():
    fsldir = Path(os.environ["FSLDIR"])
    names = extract.jhu_label_names(fsldir / extract.JHU_XML)
    assert len(names) == 51 and {v: names[v] for v in IFOF} == IFOF
    assert extract.match_labels(names, TRACTS) == _expected()
    labels = nib.load(fsldir / extract.JHU_LABELS)
    assert labels.shape == (182, 218, 182) and int(np.asarray(labels.dataobj).max()) == 50
    for rel in extract.TEMPLATES:
        assert (fsldir / rel).is_file()


@requires_fsl
def test_fslstats_agrees_with_label_qc(tmp_path):
    """fslstats -K (the training pipeline's feature) equals our per-label mean of non-zero voxels."""
    fsldir = Path(os.environ["FSLDIR"])
    labels = fsldir / extract.JHU_LABELS
    fa_img = nib.load(fsldir / "data/standard/FMRIB58_FA_1mm.nii.gz")
    skel = np.asarray(nib.load(fsldir / "data/standard/FMRIB58_FA-skeleton_1mm.nii.gz").dataobj) > 2000
    fa = np.where(skel, np.asarray(fa_img.dataobj) / 10000.0, 0).astype(np.float32)
    fa[np.asarray(nib.load(labels).dataobj) == 6] = 0  # one label without FA -> NaN
    skeletonised = tmp_path / "all_FA_skeletonised.nii.gz"
    nib.save(nib.Nifti1Image(fa, fa_img.affine, fa_img.header), skeletonised)
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(skel.astype(np.uint8), fa_img.affine), mask)
    mapping = _expected()
    qc = extract.label_qc(skeletonised, mask, labels, mapping)
    means = extract.fslstats_means(skeletonised, labels, list(mapping.values()), tmp_path)
    np.testing.assert_allclose(means, qc["fa_mean"].to_numpy(), atol=1e-5)
    assert np.isnan(means[5]) and np.isfinite(np.delete(means, 5)).all()


@requires_fsl
def test_extract_end_to_end(tmp_path):
    """The whole TBSS chain on a synthetic subject: FMRIB58_FA at 2 mm, so the features should be close to the
    template's own skeleton means (takes a few minutes: FNIRT)."""
    from tracula_braincharts.cli import main

    fsldir = Path(os.environ["FSLDIR"])
    template = nib.load(fsldir / "data/standard/FMRIB58_FA_1mm.nii.gz")
    fa = np.asarray(template.dataobj, dtype=np.float32)[::2, ::2, ::2] / 10000
    affine = template.affine.copy()
    affine[:3, :3] *= 2
    subject = tmp_path / "SUBJ01"
    (subject / "dti").mkdir(parents=True)
    nib.save(nib.Nifti1Image(fa, affine), subject / "dti" / "SUBJ01_dti_FA.nii.gz")
    (subject / "dti" / "SUBJ01.bval").write_text(" ".join(["0"] * 5 + ["1000"] * 30) + "\n")

    assert main(["extract", str(subject), "-o", str(tmp_path / "out")]) == 0
    out = tmp_path / "out" / "SUBJ01"
    features = json.loads((out / "features.json").read_text())["features"]
    assert list(features) == TRACTS and all(v is not None and 0 < v < 1 for v in features.values())
    qc = pd.read_csv(out / "label_qc.csv")
    assert qc["label"].tolist() == list(_expected().values())
    np.testing.assert_allclose(qc["fa_mean"], list(features.values()), atol=1e-6)

    # the template's own skeleton means, on the same skeleton
    skel = np.asarray(nib.load(out / "tbss" / "stats" / "mean_FA_skeleton_mask.nii.gz").dataobj) > 0
    labels = np.asarray(nib.load(fsldir / extract.JHU_LABELS).dataobj)
    tpl = np.asarray(template.dataobj) / 10000
    expected = [tpl[skel & (labels == v)].mean() for v in _expected().values()]
    assert np.corrcoef(expected, list(features.values()))[0, 1] > 0.95

    prov = json.loads((out / "provenance.json").read_text())
    assert prov["tbss"]["fsl_version"].startswith("6.0.5.2") and prov["inputs"]["method"] == "glob"
    assert set(prov["tbss"]["jhu_labels_not_used"]) == {"45", "46"}
    assert "$ tbss_4_prestats 0.2" in (out / "extract.log").read_text()
