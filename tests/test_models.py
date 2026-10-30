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
    assert sites[0] == "UKB_11025" and sites[-1] == "HCPEP_3013"


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
