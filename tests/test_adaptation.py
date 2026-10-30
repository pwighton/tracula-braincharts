"""Site adaptation (``score --adapt``), as PCNtoolkit 0.35's BLR transfer and the portal do it.

Fixtures: ``adapt-new-site.csv``, 12 synthetic controls at ``NEW_SITE`` aged 24-72, with one missing value
(ctl05, middle cerebellar peduncle), and ``test-new-site.csv``, 3 subjects at ``NEW_SITE``. FA values are the
training means shifted by about -0.5 training SD plus noise: arbitrary but in range.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, link_models, requires_models
from tracula_braincharts import models
from tracula_braincharts.cli import main
from tracula_braincharts.cli_errors import CliError
from tracula_braincharts.cohort import read_cohort
from tracula_braincharts.scoring import ADAPTATION_COLUMNS, predict, score

ADAPT = FIXTURES / "adapt-new-site.csv"
TEST = FIXTURES / "test-new-site.csv"
MISSING_TRACT = "FA_mean_of_Middle_cerebellar_peduncle"

pytestmark = requires_models


@pytest.fixture(scope="module")
def adapt():
    return read_cohort(ADAPT)


@pytest.fixture(scope="module")
def test_new():
    return read_cohort(TEST)


@pytest.fixture(scope="module")
def scores(model_set, test_new, adapt):
    s = score(model_set, test_new, adapt)
    assert s.warnings == []
    return s


def test_outputs(scores, model_set):
    assert len(scores.results) == 3 * 48 and scores.results["z"].notna().all()
    params = scores.adaptation
    assert list(params.columns) == ADAPTATION_COLUMNS and len(params) == 48 and set(params["site"]) == {"NEW_SITE"}
    assert params.set_index("tract")["n"].to_dict() == {t: 11 if t == MISSING_TRACT else 12 for t in model_set.tracts}


def test_adaptation_shifts_mean_and_sets_sd(model_set, test_new, scores):
    """Adapted prediction = unadapted prediction + mean residual; adapted SD = SD of the residuals."""
    X = models.design_matrix(test_new["age"], test_new["sex"], test_new["site"], model_set.sites)
    params = scores.adaptation.set_index("tract")
    for tract in model_set.tracts:
        yhat0, _ = predict(model_set.nms[tract], X)
        rows = scores.results[scores.results["tract"] == tract]
        np.testing.assert_allclose(rows["yhat_warped"], yhat0 + params.loc[tract, "offset_warped"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(rows["s2_warped"], params.loc[tract, "resid_sd_warped"] ** 2, rtol=1e-12)
    np.testing.assert_allclose(params["sd_ratio"], params["resid_sd_warped"] / params["model_sd_warped"])
    np.testing.assert_allclose(params["offset_model_sd"], params["offset_warped"] / params["model_sd_warped"])
    np.testing.assert_allclose(params["offset_se_warped"], params["resid_sd_warped"] / np.sqrt(params["n"]))


def test_in_sample_z_is_standardized(model_set, adapt):
    """Scoring the adaptation subjects themselves gives z with mean 0 and SD 1 per tract, by construction."""
    s = score(model_set, adapt, adapt)
    assert any("also in the adaptation data" in w for w in s.warnings)
    for tract, rows in s.results.groupby("tract"):
        z = rows["z"].dropna()
        assert len(z) == (11 if tract == MISSING_TRACT else 12)
        assert abs(z.mean()) < 1e-10 and abs(z.std(ddof=0) - 1) < 1e-10


def test_matches_pcntoolkit_transfer(model_set, test_new, adapt, tmp_path, monkeypatch):
    """Same z-scores as pcntoolkit.normative.transfer, which the portal runs when a test site is new.

    Two test sites, one new and one a training site, so the per-site grouping is covered too.
    """
    from pcntoolkit import normative

    te = pd.concat([test_new, test_new.iloc[:2].assign(subject=["hcpa1", "hcpa2"], site="HCPA_3002")],
                   ignore_index=True)
    ad = adapt.fillna(0.5)  # pcntoolkit would make the whole site x tract missing
    ad.loc[:3, "site"] = "HCPA_3002"
    ours = score(model_set, te, ad).results

    group = {s: i for i, s in enumerate(sorted(set(te["site"])))}  # as the portal numbers the test sites
    for name, df in [("ad", ad), ("te", te)]:
        np.savetxt(tmp_path / f"cov_{name}.txt", models.design_matrix(df["age"], df["sex"], df["site"], model_set.sites))
        np.savetxt(tmp_path / f"resp_{name}.txt", df[model_set.tracts].to_numpy())
        np.savetxt(tmp_path / f"be_{name}.txt", df["site"].map(group).to_numpy(dtype=float))
    mdir = link_models(model_set, tmp_path / "Models")
    monkeypatch.chdir(tmp_path)  # transfer writes its outputs to the working directory
    _yhat, _s2, Z = normative.transfer(str(tmp_path / "cov_ad.txt"), str(tmp_path / "resp_ad.txt"),
                                       testcov=str(tmp_path / "cov_te.txt"), testresp=str(tmp_path / "resp_te.txt"),
                                       alg="blr", model_path=str(mdir), trbefile=str(tmp_path / "be_ad.txt"),
                                       tsbefile=str(tmp_path / "be_te.txt"), inputsuffix="estimate",
                                       outputsuffix="transfer")

    z = ours.pivot(index="subject", columns="tract", values="z").loc[te["subject"], model_set.tracts]
    np.testing.assert_allclose(z.to_numpy(), Z, rtol=0, atol=1e-10)


@pytest.mark.parametrize("make, message", [
    (lambda te, ad: (te, None), "not in the training set.*pass --adapt"),
    (lambda te, ad: (te, ad.iloc[:1]), r"too few for: NEW_SITE \(1\)"),
    # once one test site is new, every test site is adapted, training sites too
    (lambda te, ad: (pd.concat([te, te.iloc[:1].assign(subject="x", site="HCPA_3002")]), ad),
     r"too few for: HCPA_3002 \(0\)"),
])
def test_adaptation_input_errors(model_set, test_new, adapt, make, message):
    te, ad = make(test_new, adapt)
    with pytest.raises(CliError, match=message):
        score(model_set, te, ad)


def test_too_few_values_for_one_tract_leaves_it_blank(model_set, test_new, adapt):
    tract = model_set.tracts[5]
    ad = adapt.iloc[:2].copy()
    ad.loc[ad.index[0], tract] = np.nan
    s = score(model_set, test_new, ad)
    assert s.results.loc[s.results["tract"] == tract, "z"].isna().all()
    assert s.results.loc[s.results["tract"] != tract, "z"].notna().all()
    assert s.adaptation.set_index("tract").loc[tract, "n"] == 1
    assert any(tract in w and "need 2" in w for w in s.warnings)


def test_adaptation_subjects_from_other_sites_are_ignored(model_set, test_new, adapt, scores):
    other = adapt.iloc[:3].assign(subject=["o1", "o2", "o3"], site="OTHER_SITE")
    s = score(model_set, test_new, pd.concat([adapt, other], ignore_index=True))
    assert any("not used: OTHER_SITE" in w for w in s.warnings)
    pd.testing.assert_frame_equal(s.results, scores.results)


def test_cli_with_adaptation(tmp_path):
    out = tmp_path / "out"
    assert main(["score", "--test", str(TEST), "--adapt", str(ADAPT), "-o", str(out)]) == 0
    params = pd.read_csv(out / "adaptation_params.csv")
    assert list(params.columns) == ADAPTATION_COLUMNS and len(params) == 48
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["mode"] == "site-adaptation" and prov["unseen_sites"] == ["NEW_SITE"]
    assert prov["adaptation_subjects"] == {"NEW_SITE": 12} and prov["warnings"] == []
    log_text = (out / "score.log").read_text()
    assert "INFO score: site adaptation" in log_text and "NEW_SITE: over 48 tracts, median offset" in log_text
    assert "wrote z_test.csv, centiles_test.csv, adaptation_params.csv, provenance.json" in log_text
