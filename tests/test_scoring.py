import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, link_models, requires_models
from tracula_braincharts import models
from tracula_braincharts.cli import main
from tracula_braincharts.cli_errors import CliError
from tracula_braincharts.cohort import ID_COLUMNS, TRACTS, read_cohort

FIXTURE = FIXTURES / "test-training-sites.csv"

pytestmark = requires_models


@pytest.fixture(scope="module")
def test_cohort():
    return read_cohort(FIXTURE)


@pytest.fixture(scope="module")
def results(model_set, test_cohort):
    from tracula_braincharts.scoring import score
    scores = score(model_set, test_cohort)
    assert scores.warnings == [] and scores.adaptation is None
    return scores.results


def test_one_row_per_subject_and_tract(results, test_cohort):
    assert len(results) == len(test_cohort) * 48
    assert set(results["tract"]) == set(TRACTS)


def test_missing_fa_gives_missing_z_only_there(results):
    missing = results["fa"].isna()
    assert missing.sum() == 1
    assert results.loc[missing, "z"].isna().all() and results.loc[missing, "centile"].isna().all()
    assert np.isfinite(results.loc[~missing, ["z", "centile", "yhat_warped", "s2_warped"]].to_numpy()).all()


def test_centiles_are_consistent(results):
    from scipy.stats import norm
    ok = results["z"].notna()
    np.testing.assert_allclose(results.loc[ok, "centile"], 100 * norm.cdf(results.loc[ok, "z"]))
    assert (results["fa_p05"] < results["fa_p50"]).all() and (results["fa_p50"] < results["fa_p95"]).all()
    assert results["fa_p05"].gt(0).all() and results["fa_p95"].lt(1).all()
    assert results["s2_warped"].gt(0).all()


def test_training_mean_fa_is_typical_at_ukb(results):
    # UKB is ~90% of the training data (mean age 60.8), so the pooled training-mean FA is typical for a
    # 60-year-old at a UKB site. With mislabelled site columns the splenium came out at z = 3.4.
    z = results.loc[results["subject"] == "sub-ukb60f", "z"]
    assert z.abs().max() < 1.5


def test_young_adult_has_higher_median_fa(results):
    # FA falls with age from early adulthood: the 28-year-old (HCP-YA) should be above the 75-year-old (HCP-A)
    # in nearly every tract. With mislabelled site columns this held in only 8% of tracts.
    p50 = results.pivot(index="tract", columns="subject", values="fa_p50")
    assert (p50["sub-hcpya28m"] > p50["sub-hcpa75f"]).mean() >= 0.9


def test_median_fa_scores_zero(model_set, test_cohort, results):
    from tracula_braincharts.scoring import score
    at_median = test_cohort.copy()
    p50 = results.pivot(index="subject", columns="tract", values="fa_p50").loc[test_cohort["subject"], TRACTS]
    at_median[TRACTS] = p50.to_numpy()
    res = score(model_set, at_median).results
    np.testing.assert_allclose(res["z"], 0, atol=1e-8)


def test_matches_pcntoolkit_predict(model_set, test_cohort, results, tmp_path, monkeypatch):
    """Same z-scores as pcntoolkit.normative.predict, which is what the portal runs for training sites."""
    from pcntoolkit import normative

    mdir = link_models(model_set, tmp_path / "Models")

    filled = test_cohort.fillna(0.5)  # pcntoolkit's evaluation step doesn't take missing values
    X = models.design_matrix(filled["age"], filled["sex"], filled["site"], model_set.sites)
    np.savetxt(tmp_path / "cov.txt", X)
    np.savetxt(tmp_path / "resp.txt", filled[model_set.tracts].to_numpy())
    monkeypatch.chdir(tmp_path)  # predict writes its outputs to the working directory
    _yhat, _s2, Z = normative.predict(str(tmp_path / "cov.txt"), str(tmp_path / "resp.txt"), alg="blr",
                                      model_path=str(mdir), inputsuffix="estimate", outputsuffix="ref")

    from tracula_braincharts.scoring import score
    ours = score(model_set, filled).results
    z = ours.pivot(index="subject", columns="tract", values="z").loc[filled["subject"], model_set.tracts]
    np.testing.assert_allclose(z.to_numpy(), Z, rtol=0, atol=1e-10)


def test_unseen_site_needs_adaptation(model_set, test_cohort):
    from tracula_braincharts.scoring import score
    with pytest.raises(CliError, match="not in the training set.*pass --adapt"):
        score(model_set, test_cohort.assign(site="MGH_Connectome"))


def test_adaptation_data_ignored_for_training_sites(model_set, test_cohort, results):
    from tracula_braincharts.scoring import score
    scores = score(model_set, test_cohort, adapt=test_cohort)
    assert any("not used" in w for w in scores.warnings) and scores.adaptation is None
    pd.testing.assert_frame_equal(scores.results, results)


def test_cli_writes_outputs(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["score", "--test", str(FIXTURE), "-o", str(out)]) == 0
    err, log_text = capsys.readouterr().err, (out / "score.log").read_text()
    for text in (err, log_text):
        assert " INFO score: loaded 48 tract models" in text and " INFO score: wrote z_test.csv" in text
        assert "1 of 144 FA values missing" in text
    z = pd.read_csv(out / "z_test.csv")
    assert list(z.columns) == ID_COLUMNS + TRACTS and len(z) == 3
    cent = pd.read_csv(out / "centiles_test.csv")
    assert len(cent) == 3 * 48
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["command"] == "score" and prov["mode"] == "training-sites"
    assert prov["inputs"]["test"]["n_rows"] == 3 and len(prov["inputs"]["test"]["sha256"]) == 64
    assert prov["model"]["upstream_commit"].startswith("c017a3e")
