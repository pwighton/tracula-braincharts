import itertools

import numpy as np
import pytest

from conftest import requires_models, requires_pcntoolkit
from tracula_braincharts import models
from tracula_braincharts.config import COVARIATE_COLUMNS, DESIGN_WIDTH


def test_template_columns():
    cols = models.template_columns()
    assert len(cols) == 52
    assert tuple(cols[:4]) == COVARIATE_COLUMNS
    tracts = cols[4:]
    assert len(set(tracts)) == 48
    assert all(t.startswith("FA_mean_of_") for t in tracts)


@requires_models
def test_tracts_match_template_order():
    assert models.tract_names() == models.template_columns()[4:]


@requires_models
def test_training_sites():
    sites = models.training_sites()
    assert len(sites) == 19 and len(set(sites)) == 19
    assert sites == sorted(sites) and sites[0] == "HCPA_3002" and sites[-1] == "UKB_11028"


@requires_models
def test_site_columns_group_by_study(model_set):
    """Sites of the same study (same protocol) have similar fitted site effects, if the column labels are right.

    Distance between two site columns: the difference of their weights over the 48 tracts, in noise-SD units.
    Same-study pairs average 0.30 of the all-pairs mean with the sorted labels; with the published file's
    order they average 0.75, and none of 20000 random labellings went below 0.5.
    """
    W = np.array([np.ravel(nm.blr.m)[3:22] / np.exp(-np.ravel(nm.blr.hyp)[0] / 2)
                  for nm in model_set.nms.values()]).T

    def mean_distance(same_study):
        return np.mean([np.linalg.norm(W[i] - W[j]) for i, j in itertools.combinations(range(19), 2)
                        if same_study is None or same_study(i, j)])

    study = [s.split("_")[0] for s in model_set.sites]
    assert mean_distance(lambda i, j: study[i] == study[j]) < 0.5 * mean_distance(None)


@requires_models
@pytest.mark.parametrize("tract", models.template_columns()[4:])
def test_model_loads(tract):
    nm = models.load_model(tract)
    assert type(nm).__name__ == "NormBLR"
    assert nm.D == DESIGN_WIDTH and nm.blr.D == DESIGN_WIDTH
    assert type(nm.blr.warp).__name__ == "WarpSinArcsinh"
    meta = models.load_meta(tract)
    assert meta["inscaler"] == "None" and meta["outscaler"] == "None"


@requires_models
def test_design_matrix_layout():
    sites = models.training_sites()
    X = models.design_matrix(age=[30.0, 70.5, 45.0], sex=[0, 1, 1], site=[sites[0], sites[18], "NEW_SITE"],
                             sites=sites)
    assert X.shape == (3, DESIGN_WIDTH)
    assert np.all(X[:, 0] == 1)
    np.testing.assert_array_equal(X[:, 1], [30.0, 70.5, 45.0])
    np.testing.assert_array_equal(X[:, 2], [0, 1, 1])
    site_cols = X[:, 3:22]
    np.testing.assert_array_equal(site_cols[0], np.eye(19)[0])
    np.testing.assert_array_equal(site_cols[1], np.eye(19)[18])
    assert not site_cols[2].any()  # unseen site: no site column set
    # B-spline basis functions form a partition of unity inside [xmin, xmax]
    np.testing.assert_allclose(X[:, 22:].sum(axis=1), 1.0)


@requires_models
def test_predict_smoke():
    sites = models.training_sites()
    nm = models.load_model(models.tract_names()[0])
    X = models.design_matrix(age=[25.0, 65.0], sex=[0, 1], site=[sites[0], sites[0]], sites=sites)
    yhat, s2 = nm.predict(X)
    yhat, s2 = np.ravel(yhat), np.ravel(s2)
    assert yhat.shape == (2,) and s2.shape == (2,)
    assert np.all(np.isfinite(yhat)) and np.all(s2 > 0)


@requires_pcntoolkit
def test_design_matrix_rejects_wrong_width():
    with pytest.raises(ValueError, match="columns"):
        models.design_matrix(age=[30.0], sex=[0], site=["A"], sites=["A", "B"])
