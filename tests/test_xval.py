"""Cross-validation of site adaptation (``x-val``) on the ``NEW_SITE`` fixture controls (see test_adaptation.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, requires_models
from tracula_braincharts.cli import main
from tracula_braincharts.cli_errors import CliError
from tracula_braincharts.cohort import ID_COLUMNS, TRACTS, read_cohort
from tracula_braincharts.xval import QA_COLUMNS, assign_folds, check

ADAPT = FIXTURES / "adapt-new-site.csv"
MISSING_TRACT = "FA_mean_of_Middle_cerebellar_peduncle"
TRAINING = SimpleNamespace(sites=["HCPA_3002", "UKB_11025"])  # check() only needs the training site list


@pytest.fixture(scope="module")
def adapt():
    return read_cohort(ADAPT)


def _two_sites(adapt):
    return pd.concat([adapt, adapt.iloc[:5].assign(subject=[f"b{i}" for i in range(5)], site="SITE_B")],
                     ignore_index=True)


def test_leave_one_out_folds(adapt):
    folds = assign_folds(_two_sites(adapt))
    assert list(folds[:12]) == list(range(12)) and list(folds[12:]) == list(range(5))


def test_k_fold_folds_are_balanced_within_site_and_seeded(adapt):
    two = _two_sites(adapt)
    folds = assign_folds(two, n_folds=3, seed=1)
    assert sorted(np.bincount(folds[:12])) == [4, 4, 4] and sorted(np.bincount(folds[12:])) == [1, 2, 2]
    np.testing.assert_array_equal(folds, assign_folds(two, n_folds=3, seed=1))
    assert not np.array_equal(folds, assign_folds(two, n_folds=3, seed=2))


@pytest.mark.parametrize("make, n_folds, message", [
    (lambda a: a.assign(site="HCPA_3002"), None, "training sites.*HCPA_3002"),
    (lambda a: a.iloc[:2], None, r"2 control\(s\); leave-one-out cross-validation needs at least 3"),
    (lambda a: a.iloc[:3], 2, r"3 control\(s\); 2-fold cross-validation needs at least 4"),
    (lambda a: a.iloc[:4], 5, r"4 control\(s\); 5-fold cross-validation needs at least 5"),
    (lambda a: a, 1, "at least 2"),
])
def test_rejects_what_cannot_be_cross_validated(adapt, make, n_folds, message):
    with pytest.raises(CliError, match=message):
        check(TRAINING, make(adapt), n_folds)


def test_accepts_smallest_cohorts(adapt):
    check(TRAINING, adapt.iloc[:3], None)
    check(TRAINING, adapt.iloc[:4], 2)


@requires_models
def test_each_z_is_scored_without_its_own_data(model_set, adapt):
    from tracula_braincharts.scoring import score
    from tracula_braincharts.xval import cross_validate

    folds = assign_folds(adapt)
    scores = cross_validate(model_set, adapt, folds)
    cv = scores.results
    assert len(cv) == 12 * 48 and set(cv["fold"]) == set(range(12))
    # each fold adapts with the other 11 controls (10 for the tract ctl05 is missing, except in ctl05's own fold)
    n = scores.adaptation.set_index(["fold", "tract"])["n"]
    assert n.xs("FA_mean_of_Tapetum_L", level="tract").eq(11).all()
    assert sorted(n.xs(MISSING_TRACT, level="tract")) == [10] * 11 + [11]
    assert cv.loc[cv["tract"] == MISSING_TRACT, "z"].isna().sum() == 1 and cv["z"].notna().sum() == 12 * 48 - 1
    for i in (0, 7):
        held_out = adapt.iloc[[i]]
        direct = score(model_set, held_out, adapt.drop(index=adapt.index[i])).results
        mine = cv[cv["subject"] == held_out["subject"].iloc[0]].drop(columns="fold").reset_index(drop=True)
        pd.testing.assert_frame_equal(mine, direct)
    # out-of-sample z are not the standardized in-sample z
    in_sample = score(model_set, adapt, adapt).results
    for tract in model_set.tracts:
        z = cv.loc[cv["tract"] == tract, "z"].dropna()
        assert not np.isclose(z.std(ddof=0), 1.0) or not np.isclose(z.mean(), 0.0)
        assert len(in_sample.loc[in_sample["tract"] == tract, "z"].dropna()) == len(z)


@requires_models
def test_qa_table_matches_the_z_scores(model_set, adapt):
    from scipy import stats

    from tracula_braincharts.xval import cross_validate, qa_adaptation

    cv = cross_validate(model_set, adapt, assign_folds(adapt)).results
    qa = qa_adaptation(cv, adapt, model_set.tracts)
    assert list(qa.columns) == QA_COLUMNS and len(qa) == 48
    row = qa.set_index("tract").loc["FA_mean_of_Splenium_of_corpus_callosum"]
    g = cv[cv["tract"] == "FA_mean_of_Splenium_of_corpus_callosum"].merge(adapt[["subject", "age", "sex"]])
    assert row["n"] == 12
    # QA statistics are of the corrected z (each control's fold adapts with m = 11)
    assert g["n_adapt"].eq(11).all()
    assert np.isclose(row["z_mean"], g["z_cal"].mean()) and np.isclose(row["z_sd"], g["z_cal"].std(ddof=1))
    assert np.isclose(row["r_age"], stats.pearsonr(g["age"], g["z_cal"]).statistic)
    assert np.isclose(row["frac_abs_z_gt2"], (g["z_cal"].abs() > 2).mean())
    assert qa.set_index("tract").loc[MISSING_TRACT, "n"] == 11
    assert qa[["z_skew", "z_kurtosis", "shapiro_p", "rho", "smse", "expv"]].notna().all().all()


def test_cli_requires_adapt_and_out():
    from tracula_braincharts.cli import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["x-val", "--adapt", "a.csv"])
    args = build_parser().parse_args(["x-val", "--adapt", "a.csv", "-o", "out"])
    assert args.folds is None and args.seed == 0


@requires_models
def test_cli_writes_outputs(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["x-val", "--adapt", str(ADAPT), "-o", str(out)]) == 0
    for name in ("z_adapt_cv.csv", "z_cal_adapt_cv.csv"):
        z = pd.read_csv(out / name)
        assert list(z.columns) == ID_COLUMNS + ["fold"] + TRACTS and len(z) == 12
    assert len(pd.read_csv(out / "centiles_adapt_cv.csv")) == 12 * 48
    assert list(pd.read_csv(out / "qa_adaptation.csv").columns) == QA_COLUMNS
    png = out / "plots" / "centiles_NEW_SITE.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and png.stat().st_size > 50_000
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["command"] == "x-val" and prov["cross_validation"]["scheme"] == "leave-one-out"
    assert prov["sites"] == {"NEW_SITE": 12}
    for text in (capsys.readouterr().err, (out / "x-val.log").read_text()):
        assert "INFO x-val: leave-one-out cross-validation: 12 fold(s)" in text
        assert "INFO x-val: NEW_SITE: z_cal over 48 tracts: median mean" in text
        assert "plots/centiles_NEW_SITE.png" in text
