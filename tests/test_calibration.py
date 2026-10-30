"""Small-sample correction of adapted z-scores (``scoring.calibrate``, ``scoring.quantile_z``). No models needed."""

import numpy as np
from scipy import stats

from tracula_braincharts.scoring import calibrate, quantile_z


def _adapted_z(m: int, reps: int, rng) -> np.ndarray:
    """z of new subjects as PCNtoolkit's adaptation computes it: minus the mean, over the SD (ddof 0), of m values."""
    ref, new = rng.standard_normal((reps, m)), rng.standard_normal(reps)
    return (new - ref.mean(axis=1)) / ref.std(axis=1)


def test_uncorrected_z_is_too_wide():
    z = _adapted_z(11, 100_000, np.random.default_rng(0))
    assert abs(z.std() - np.sqrt(12 / 8)) < 0.01  # sqrt((m+1)/(m-3))
    assert abs(np.mean(np.abs(z) > 2) - 0.098) < 0.003


def test_calibrated_z_is_standard_normal():
    rng = np.random.default_rng(1)
    for m in (4, 11, 30):
        z = _adapted_z(m, 100_000, rng)
        z_cal, p = calibrate(z, np.full(len(z), float(m)))
        assert stats.kstest(z_cal, "norm").statistic < 0.01
        np.testing.assert_allclose(p, stats.norm.cdf(z_cal), atol=1e-12)


def test_no_adaptation_leaves_z_unchanged():
    z = np.array([-2.5, 0.0, 1.0, np.nan])
    z_cal, p = calibrate(z, np.full(4, np.nan))
    np.testing.assert_array_equal(z_cal, z)
    np.testing.assert_allclose(p[:3], stats.norm.cdf(z[:3]))


def test_quantiles_round_trip_and_tails():
    m = np.array([np.nan, 5.0, 12.0, 100.0])
    q = quantile_z(0.05, m)
    assert q[0] == stats.norm.ppf(0.05) and q[1] < q[2] < q[3] < stats.norm.ppf(0.05)
    np.testing.assert_allclose(calibrate(q, m)[0], stats.norm.ppf(0.05), atol=1e-10)
    z_cal, _ = calibrate(np.array([40.0, -40.0]), np.array([12.0, 12.0]))
    assert np.isfinite(z_cal).all() and np.isclose(z_cal[0], -z_cal[1]) and 0 < z_cal[0] < 40
