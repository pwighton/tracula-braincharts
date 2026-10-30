"""Normative scores for test subjects, computed as PCNtoolkit's ``normative.predict`` does for BLR models.

For each tract the model predicts a mean ``yhat`` and variance ``s2`` in warped (Gaussian) space; the
observed FA is warped with the model's SinArcsinh warp and ``z = (warp(fa) - yhat) / sqrt(s2)``.
Centiles in FA units come from un-warping ``yhat + z_p * sqrt(s2)``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import models, provenance
from .cli_errors import CliError
from .cohort import ID_COLUMNS
from .config import MODEL_DIR, MODELS_SUBDIR

log = logging.getLogger(__name__)

# Training age range (portal model README: 0-100 years)
TRAINING_AGE_RANGE = (0.0, 100.0)

# Centiles reported in FA units: column -> z
CENTILE_COLUMNS = {"fa_p05": norm.ppf(0.05), "fa_p50": 0.0, "fa_p95": norm.ppf(0.95)}

RESULT_COLUMNS = ["subject", "tract", "fa", "z", "centile", *CENTILE_COLUMNS, "yhat_warped", "s2_warped"]


class ModelSet:
    """The 48 tract models and the training site list, loaded once."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        if not (self.model_dir / MODELS_SUBDIR).is_dir():
            raise CliError(f"model files not found in {self.model_dir} (run inside the container, or pass "
                           "--model-dir pointing at a directory made by tools/fetch_models.sh)")
        self.tracts = models.tract_names(self.model_dir)
        self.sites = models.training_sites(self.model_dir)
        self.nms = {t: models.load_model(t, self.model_dir) for t in self.tracts}


def warp_params(nm) -> np.ndarray:
    return nm.blr.hyp[1:nm.blr.warp.get_n_params() + 1]


def score_tract(nm, X: np.ndarray, fa: np.ndarray):
    """Warped-space prediction and z-score for one tract. Returns ``(yhat, s2, z)``."""
    yhat, s2 = nm.predict(X)
    yhat, s2 = np.ravel(yhat), np.ravel(s2)
    z = (nm.blr.warp.f(fa, warp_params(nm)) - yhat) / np.sqrt(s2)
    return yhat, s2, z


def score(model_set: ModelSet, test: pd.DataFrame, adapt: pd.DataFrame | None = None):
    """Score every test row and tract.

    Returns ``(results, warnings)``: results in long format (one row per subject and tract, columns
    ``RESULT_COLUMNS``) and a list of warning strings.
    """
    warnings = []
    unseen = sorted(set(test["site"]) - set(model_set.sites))
    if unseen:
        raise CliError(f"site(s) not in the training set: {', '.join(unseen)}; "
                       "site adaptation is not implemented yet (plan step 3)")
    if adapt is not None:
        warnings.append("all test sites are training sites, so the adaptation data is not used (as on the portal)")
    lo, hi = TRAINING_AGE_RANGE
    outside = test.loc[~test["age"].between(lo, hi), "subject"]
    if not outside.empty:
        warnings.append(f"age outside the training range {lo:g}-{hi:g} years: {', '.join(outside)}")

    X = models.design_matrix(test["age"], test["sex"], test["site"], model_set.sites)
    parts = []
    for tract in model_set.tracts:
        nm = model_set.nms[tract]
        fa = test[tract].to_numpy(dtype=float)
        yhat, s2, z = score_tract(nm, X, fa)
        part = pd.DataFrame({"subject": test["subject"].to_numpy(), "tract": tract, "fa": fa, "z": z,
                             "centile": 100 * norm.cdf(z)})
        for col, zp in CENTILE_COLUMNS.items():
            part[col] = nm.blr.warp.invf(yhat + zp * np.sqrt(s2), warp_params(nm))
        part["yhat_warped"] = yhat
        part["s2_warped"] = s2
        parts.append(part)
    return pd.concat(parts, ignore_index=True)[RESULT_COLUMNS], warnings


def z_wide(test: pd.DataFrame, results: pd.DataFrame, tracts: list[str]) -> pd.DataFrame:
    """One row per test subject: ``subject, sex, age, site`` + one z column per tract (template order)."""
    z = results.pivot(index="subject", columns="tract", values="z").reindex(index=test["subject"], columns=tracts)
    return pd.concat([test[ID_COLUMNS].reset_index(drop=True), z.reset_index(drop=True)], axis=1)


def run(test_csv: Path, out_dir: Path, adapt_csv: Path | None = None, model_dir: Path = MODEL_DIR) -> None:
    """The ``score`` subcommand: read, score, write ``z_test.csv``, ``centiles_test.csv``, ``provenance.json``.

    Progress and warnings are logged; the warnings are also recorded in the provenance.
    """
    from .cohort import read_cohort

    test = read_cohort(test_csv)
    sites = sorted(set(test["site"]))
    log.info("score: test %s: %d subject(s), site(s) %s", test_csv, len(test), ", ".join(sites))
    adapt = read_cohort(adapt_csv) if adapt_csv is not None else None
    if adapt is not None:
        log.info("score: adapt %s: %d subject(s)", adapt_csv, len(adapt))
    model_set = ModelSet(model_dir)
    log.info("score: loaded %d tract models from %s", len(model_set.tracts), model_set.model_dir)
    results, warnings = score(model_set, test, adapt)
    log.info("score: all test sites are training sites: scored without site adaptation")
    for warning in warnings:
        log.warning("score: %s", warning)
    scored = results["z"].notna()
    if not scored.all():
        log.info("score: %d of %d FA values missing; their z-scores are left blank", (~scored).sum(), len(results))
    log.info("score: |z| > 2 for %d of %d z-scores", (results.loc[scored, "z"].abs() > 2).sum(), scored.sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    z_wide(test, results, model_set.tracts).to_csv(out_dir / "z_test.csv", index=False)
    results.to_csv(out_dir / "centiles_test.csv", index=False)

    record = provenance.base_record("score", model_set.model_dir)
    record["inputs"] = {"test": {**provenance.file_info(test_csv), "n_rows": len(test)},
                        "adapt": ({**provenance.file_info(adapt_csv), "n_rows": len(adapt)}
                                  if adapt is not None else None)}
    record["mode"] = "training-sites"
    record["sites"] = sorted(set(test["site"]))
    record["warnings"] = warnings
    provenance.write(record, out_dir / "provenance.json")
    log.info("score: wrote z_test.csv, centiles_test.csv, provenance.json to %s", out_dir)
