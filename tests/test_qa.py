"""extract's QA metrics: eddy's shells and outputs, the native FA, the skeleton, and the flags."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from tracula_braincharts import extract, qa

# Connectome-like protocol: b=0 and b=50 (one eddy shell, b <= 100), 2300 and 2400 (exactly 100 apart: two shells)
BVALS = np.array([0, 50, 950, 200, 950, 2300, 2400, 0, 950, 50, 200, 2300, 2400, 0], dtype=float)


def _img(data, path, affine=None):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine), path)
    return path


def test_eddy_shells_follow_eddy():
    means, shell = qa.eddy_shells(BVALS)
    assert means == pytest.approx([20, 200, 950, 2300, 2400])  # b0 group = mean of 0,50,0,50,0
    np.testing.assert_array_equal(shell, [0, 0, 2, 1, 2, 3, 4, 0, 2, 0, 1, 3, 4, 0])
    # within range of a template: one shell at the mean
    means, shell = qa.eddy_shells(np.array([0, 1000, 1050, 0, 1040], dtype=float))
    assert means == pytest.approx([0, 1030]) and shell.tolist() == [0, 1, 1, 0, 1]


@pytest.fixture
def eddy_subject(tmp_path):
    """A subject with eddy outputs for BVALS; the FA comes from b=0 and b=950 (dwiextract keeps eddy's order)."""
    subject = tmp_path / "S1"
    (subject / "eddy" / "results").mkdir(parents=True)
    (subject / "diff-comb").mkdir()
    np.savetxt(subject / "diff-comb" / "S1_diff.bval", BVALS[None], fmt="%g")
    prefix = subject / "eddy" / "results" / "S1_diff_eddy"
    (prefix.parent / (prefix.name + ".eddy_command_txt")).write_text(
        "/opt/fsl/bin/eddy_cuda --imain=/elsewhere/S1/diff-comb/S1_diff.nii.gz "
        "--bvals=/elsewhere/S1/diff-comb/S1_diff.bval --out=/elsewhere/S1/eddy/results/S1_diff_eddy --cnr_maps\n")
    n = len(BVALS)
    rms = np.column_stack([np.arange(n, dtype=float), np.full(n, 0.5)])
    rms[2, 1] = 3.0  # a jump at a b=950 volume
    np.savetxt(prefix.parent / (prefix.name + ".eddy_movement_rms"), rms)
    np.savetxt(prefix.parent / (prefix.name + ".eddy_restricted_movement_rms"), rms / 2)
    omap = np.zeros((n, 4), dtype=int)
    omap[5, :2] = 1  # 2 outlier slices in a b=2300 volume, none in the FA's volumes
    with open(prefix.parent / (prefix.name + ".eddy_outlier_map"), "w") as f:
        f.write("One row per scan, one column per slice. Outlier: 1, Non-outlier: 0\n")
        np.savetxt(f, omap, fmt="%d")
    cnr = np.zeros((3, 3, 3, 5))
    cnr[1:, :, :, :] = np.arange(5) + 1.0  # volume k holds k + 1 inside eddy's mask
    _img(cnr, prefix.parent / (prefix.name + ".eddy_cnr_maps.nii.gz"))
    fa_bvals = BVALS[(BVALS == 0) | (BVALS == 950)]
    return subject, fa_bvals


def test_eddy_qa(eddy_subject):
    subject, fa_bvals = eddy_subject
    out = qa.eddy_qa(subject, fa_bvals, 950.0)
    assert out["source"]["bvals"] == "diff-comb/S1_diff.bval"  # from eddy_command_txt, found locally
    assert out["volumes"] == 14 and out["fa_volumes"] == 6
    fa_idx = [0, 2, 4, 7, 8, 13]
    m = out["movement_rms"]
    assert m["all"]["abs_max_mm"] == 13 and m["fa_volumes"]["abs_mean_mm"] == pytest.approx(np.mean(fa_idx), abs=1e-4)
    assert m["fa_volumes"]["rel_max_mm"] == 3.0
    assert out["restricted_movement_rms"]["fa_volumes"]["rel_max_mm"] == 1.5
    o = out["outliers"]
    assert o["all_dwi_volumes_with_outliers"] == 1
    assert o["all_dwi_slices_pct"] == pytest.approx(100 * 2 / (9 * 4), abs=1e-4)
    assert o["fa_dwi_slices_pct"] == 0 and o["fa_dwi_volumes_with_outliers"] == 0
    c = out["cnr"]
    # volume 0 = b0 group SNR; then 200, 950, 2300, 2400: b=950 is volume 2
    assert c["eddy_shells"] == [200, 950, 2300, 2400] and c["b0_group_mean_b"] == 20
    assert c["b0_group_snr_median"] == 1 and c["shell_b"] == 950 and c["shell_cnr_median"] == 3


def test_eddy_qa_mismatches(eddy_subject):
    subject, fa_bvals = eddy_subject
    out = qa.eddy_qa(subject, fa_bvals[:-1], 950.0)
    assert out["fa_volumes"] is None and "don't match" in out["fa_volumes_reason"]
    assert "fa_volumes" not in out["movement_rms"] and "shell_cnr_reason" in out["cnr"]
    for f in (subject / "eddy" / "results").glob("*cnr_maps*"):
        _img(np.ones((3, 3, 3, 4)), f)
    assert "expected 5" in qa.eddy_qa(subject, fa_bvals, 950.0)["cnr"]["reason"]
    assert "0 eddy/results" in qa.eddy_qa(subject / "nowhere", fa_bvals, 950.0)["reason"]


def test_native_qa(tmp_path):
    fa = np.full((4, 4, 4), 0.5)
    fa[0, 0, 0], fa[0, 0, 1], fa[0, 0, 2] = 1.2, 0.1, np.nan
    mask = np.ones((4, 4, 4))
    mask[3] = 0
    rng = np.random.default_rng(0)
    b0 = 100 + rng.normal(0, 10, (4, 4, 4, 20))
    dwi = np.concatenate([b0, np.full((4, 4, 4, 5), 30.0)], axis=3)
    bvals = np.array([0.0] * 20 + [1000.0] * 5)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    out = qa.native_qa(_img(fa, tmp_path / "fa.nii.gz", affine), _img(mask, tmp_path / "m.nii.gz", affine),
                       _img(dwi, tmp_path / "dwi.nii.gz", affine), bvals)
    assert out["mask_volume_ml"] == pytest.approx(48 * 8 / 1000, abs=0.05)
    assert out["nonfinite_voxels"] == 1 and out["fa_above_1_voxels"] == 1 and out["wm_median_fa"] == 0.5
    assert out["b0_volumes"] == 20 and 7 < out["b0_tsnr_median"] < 14  # mean 100 / SD 10
    out = qa.native_qa(tmp_path / "fa.nii.gz", None, None, bvals)
    assert out["mask"] == "FA != 0" and out["b0_tsnr_median"] is None and "not found" in out["b0_tsnr_reason"]


def test_parse_avscale():
    text = """Rotation Angles (x,y,z) [rads] = 0.049309 0.031816 -0.006877 \n
Translations (x,y,z) [mm] = 20.127772 3.456435 -49.110320 \n
Scales (x,y,z) = 0.881536 0.956375 0.864983 \n
Skews (xy,xz,yz) = 0.071748 -0.037891 0.006310 \n
Average scaling = 0.900965\n
Determinant = 0.729249
Left-Right orientation: preserved
"""
    out = qa.parse_avscale(text)
    assert out["scales"] == [0.881536, 0.956375, 0.864983] and out["rotations_rad"][2] == -0.006877
    assert out["average_scaling"] == 0.900965 and out["determinant"] == 0.729249


def test_skeleton_qa(tmp_path):
    stats = tmp_path / "stats"
    stats.mkdir()
    skel = np.zeros((4, 4, 4))
    skel[1, 1, :] = 1
    warped = np.full((4, 4, 4), 0.3)
    proj = np.zeros((4, 4, 4))
    proj[1, 1, :] = [0.5, 0.4, 0.1, 0]  # one below 0.2, one without FA
    _img(skel, stats / "mean_FA_skeleton_mask.nii.gz")
    _img(warped, stats / "all_FA.nii.gz")
    _img(proj, stats / "all_FA_skeletonised.nii.gz")
    out = qa.skeleton_qa(tmp_path, 0.2)
    assert out["voxels"] == 4 and out["zero_pct"] == 25
    assert out["below_threshold_pct"] == pytest.approx(100 / 3, abs=1e-4)
    assert out["projection_gain_mean"] == pytest.approx(np.mean([0.2, 0.1, -0.2]), abs=1e-4)


def test_label_qc_template_skeleton(tmp_path):
    lab = np.zeros((4, 4, 4))
    lab[0] = 1
    skel = np.zeros_like(lab)
    skel[0, 0, :] = 1
    fa = np.zeros_like(lab)
    fa[0, 0, :] = 0.5
    template = np.zeros_like(lab)
    template[0, :2, :] = 5000  # 8 template skeleton voxels (FA x 10000)
    fa_mask = np.ones_like(lab)
    fa_mask[0, 1, :2] = 0  # 2 of them outside the subject's warped FA
    qc = extract.label_qc(_img(fa, tmp_path / "fa.nii.gz"), _img(skel, tmp_path / "s.nii.gz"),
                          _img(lab, tmp_path / "l.nii.gz"), {"A": 1}, _img(fa_mask, tmp_path / "m.nii.gz"),
                          _img(template, tmp_path / "t.nii.gz"))
    row = qc.iloc[0]
    assert row["template_skeleton_voxels"] == 8 and row["template_in_mask_pct"] == 75 and row["coverage_pct"] == 100


def test_flags():
    labels = pd.DataFrame({"tract": ["FA_mean_of_A", "FA_mean_of_B", "FA_mean_of_C"],
                           "nonzero_voxels": [10, 0, 5], "coverage_pct": [100, 0, 80],
                           "template_in_mask_pct": [100, 100, 90.0]})
    native = {"nonfinite_voxels": 0}
    reg = {"jacobian": {"nonpositive_voxels": 3}}
    out = qa.flags(native, reg, labels)
    assert set(out) == {"jacobian_nonpositive", "label_no_value", "label_coverage", "skeleton_outside_fa"}
    assert out["label_no_value"] == "1 label(s): B" and "C (80.0%)" in out["label_coverage"]
    assert "C (90.0%)" in out["skeleton_outside_fa"]
    ok = labels.iloc[[0]]
    assert qa.flags({"nonfinite_voxels": 2}, {"jacobian": {"nonpositive_voxels": 0}}, ok) == {
        "fa_nonfinite": "2 NaN/inf voxels in the FA map"}
