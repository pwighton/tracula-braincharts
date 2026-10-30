"""Finding a subject's FA map and b-values, and the acquisition checks (no FSL or models needed)."""

import nibabel as nib
import numpy as np
import pytest

from tracula_braincharts import inputs
from tracula_braincharts.cli_errors import CliError

SUBJ = "Tract_XX_001"
DTIFIT = ("dtifit -k /data/site/{s}/dti/{s}_diff_b950 -o /data/site/{s}/dti/{s}_dti "
          "-m /data/site/{s}/bmask-eddy/{s}_mask.nii.gz -r /data/site/{s}/dti/{s}_diff_b950.bvec "
          "-b /data/site/{s}/dti/{s}_diff_b950.bval --save_tensor")


def _nifti(path, shape):
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.diag([2.0, 2.0, 2.0, 1.0])), path)


@pytest.fixture
def subject(tmp_path):
    """A minimal wrapper-style subject: dti/ (FA, b950 DWI, bval), bmask-eddy/ mask, scripts/log.txt, diff/."""
    d = tmp_path / SUBJ
    for sub in ("dti", "bmask-eddy", "scripts", "diff", "kurt"):
        (d / sub).mkdir(parents=True)
    _nifti(d / "dti" / f"{SUBJ}_dti_FA.nii.gz", (4, 4, 4))
    _nifti(d / "dti" / f"{SUBJ}_diff_b950.nii.gz", (4, 4, 4, 5))
    _nifti(d / "bmask-eddy" / f"{SUBJ}_mask.nii.gz", (4, 4, 4))
    (d / "dti" / f"{SUBJ}_diff_b950.bval").write_text("0 950 955 0 945\n")
    kurt = DTIFIT.format(s=SUBJ).replace("/dti/", "/kurt/").replace("_dti ", "_kurt ")
    (d / "scripts" / "log.txt").write_text(f"echo start\n{DTIFIT.format(s=SUBJ)}\n{kurt}\n")
    (d / "diff" / f"{SUBJ}_diff.bval").write_text("0 950 950 1500 0\n")
    (d / "diff" / f"{SUBJ}_diff.delta").write_text("49 49 49 19 19\n")
    return d


def test_from_log_uses_the_dti_dtifit_with_local_paths(subject):
    i = inputs.resolve(subject)
    assert i.method == "log" and i.log_line == 2 and i.subject == SUBJ
    assert i.fa == subject / "dti" / f"{SUBJ}_dti_FA.nii.gz"
    assert i.bval == subject / "dti" / f"{SUBJ}_diff_b950.bval"
    assert i.dwi == subject / "dti" / f"{SUBJ}_diff_b950.nii.gz"
    assert i.mask == subject / "bmask-eddy" / f"{SUBJ}_mask.nii.gz"


def test_long_options_are_understood():
    args = inputs.parse_dtifit("/usr/bin/dtifit --data=/a/dti/x --out=/a/dti/s_dti --bvals=/a/dti/x.bval")
    assert args == {"data": "/a/dti/x", "out": "/a/dti/s_dti", "bvals": "/a/dti/x.bval"}
    assert inputs.parse_dtifit("echo dtifit -o x") is None


def test_falls_back_to_glob(subject):
    (subject / "scripts" / "log.txt").unlink()
    i = inputs.resolve(subject)
    assert i.method == "glob" and i.dwi is None and i.fa.name == f"{SUBJ}_dti_FA.nii.gz"


def test_glob_needs_exactly_one_candidate(subject):
    (subject / "scripts" / "log.txt").unlink()
    (subject / "dti" / "other.bval").write_text("0 1000\n")
    with pytest.raises(CliError, match="pass --fa and --bval"):
        inputs.resolve(subject)


def test_explicit_files(subject, tmp_path):
    fa, bval = tmp_path / "fa.nii.gz", tmp_path / "b.bval"
    _nifti(fa, (4, 4, 4))
    bval.write_text("0 1000\n")
    i = inputs.resolve(subject, fa=fa, bval=bval)
    assert i.method == "explicit" and i.fa == fa and i.mask is None
    with pytest.raises(CliError, match="both --fa and --bval"):
        inputs.resolve(subject, fa=fa)


def test_logged_fa_missing_is_an_error(subject):
    (subject / "dti" / f"{SUBJ}_dti_FA.nii.gz").unlink()
    with pytest.raises(CliError, match="line 2.*FA map not found"):
        inputs.resolve(subject)


def test_shells_and_checks():
    s = inputs.shells(np.array([0, 5, 950, 955, 945, 0]))
    assert s.n_b0 == 3 and s.shells == [{"b": 950.0, "n": 3}]
    assert inputs.check_shells(s) == []
    for bvals, message in (([0, 1000, 2000], "2 non-zero shells"), ([950, 955], "no b=0"),
                           ([0, 3000, 3000], "outside 800-1600")):
        with pytest.raises(CliError, match=message):
            inputs.check_shells(inputs.shells(np.array(bvals, dtype=float)))
    assert inputs.check_shells(inputs.shells(np.array([0.0, 3000.0])), allow=True)[0].startswith("shell b = 3000")


def test_diffusion_time(subject):
    assert inputs.diffusion_time(subject, 950)["delta_ms"] == 49.0
    assert inputs.diffusion_time(subject, 0)["delta_ms"] is None  # b=0 at both Delta
    assert "no volumes" in inputs.diffusion_time(subject, 2300)["reason"]
    (subject / "diff" / f"{SUBJ}_diff.delta").unlink()
    assert inputs.diffusion_time(subject, 950)["delta_ms"] is None
