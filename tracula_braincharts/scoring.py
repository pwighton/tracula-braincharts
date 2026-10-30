"""Normative scores for test subjects, computed as PCNtoolkit 0.35 and the portal do for BLR models.

For each tract the model predicts a mean ``yhat`` and variance ``s2`` in warped (Gaussian) space; the
observed FA is warped with the model's SinArcsinh warp and ``z = (warp(fa) - yhat) / sqrt(s2)``.
Centiles in FA units come from un-warping ``yhat + z_p * sqrt(s2)``.

Site adaptation (PCNtoolkit ``BLR.predict_and_adjust``, which the portal's transfer path uses): when any test site
is not a training site, every test site is adapted with the adaptation subjects from that site. Per tract, their
residuals ``warp(fa) - yhat`` shift the predicted mean by the residuals' mean, and the residuals' SD (ddof 0)
replaces the predictive SD.
"""

import logging
from pathlib import Path
from typing import NamedTuple

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

# predict_and_adjust needs at least 2 adaptation values per site to estimate a variance
MIN_ADAPT = 2

# One row per tract and test site. Residual = warp(fa) - unadapted yhat, for the site's adaptation subjects.
#   offset_warped     mean residual: how far the site sits above (+) or below (-) the model; added to yhat
#   offset_se_warped  its standard error, resid_sd_warped / sqrt(n)
#   offset_model_sd   offset_warped in units of the model's predictive SD
#   resid_sd_warped   SD of the residuals (ddof 0): the adapted predictive SD
#   model_sd_warped   the model's own predictive SD at the adaptation subjects (mean)
#   sd_ratio          resid_sd_warped / model_sd_warped: > 1 means more spread at this site than the model expects
#   fa_offset         mean FA minus mean predicted median FA of the adaptation subjects (FA units)
ADAPTATION_COLUMNS = ["tract", "site", "n", "offset_warped", "offset_se_warped", "offset_model_sd",
                      "resid_sd_warped", "model_sd_warped", "sd_ratio", "fa_offset"]


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


class Scores(NamedTuple):
    results: pd.DataFrame  # long format, RESULT_COLUMNS
    adaptation: pd.DataFrame | None  # ADAPTATION_COLUMNS; None when scored without site adaptation
    warnings: list[str]


def warp_params(nm) -> np.ndarray:
    return nm.blr.hyp[1:nm.blr.warp.get_n_params() + 1]


def predict(nm, X: np.ndarray, **adaptation) -> tuple[np.ndarray, np.ndarray]:
    """Warped-space ``(yhat, s2)`` for one tract; ``adaptation`` is passed on to ``NormBLR.predict``."""
    yhat, s2 = nm.predict(X, **adaptation)
    return np.ravel(yhat), np.ravel(s2)


def adapt_tract(nm, X_te: np.ndarray, group_te: np.ndarray, X_ad: np.ndarray, fa_ad: np.ndarray,
                group_ad: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Site-adapted ``(yhat, s2)`` for one tract, and the adaptation parameters of each site group.

    Groups are integer codes ``0..n_groups-1`` (one per test site; -1 for adaptation subjects from other sites).
    Adaptation subjects missing this tract's FA are left out. Test rows of a group with fewer than ``MIN_ADAPT``
    remaining values get NaN (PCNtoolkit would raise an error for the whole run).
    """
    wp = warp_params(nm)
    valid = ~np.isnan(fa_ad) & (group_ad >= 0)
    params, adaptable = [], []
    for g in range(n_groups):
        sel = valid & (group_ad == g)
        p = {"n": int(sel.sum())}
        if p["n"] >= MIN_ADAPT:
            adaptable.append(g)
            yhat_ref, s2_ref = predict(nm, X_ad[sel])
            resid = nm.blr.warp.f(fa_ad[sel], wp) - yhat_ref
            model_sd = np.sqrt(s2_ref).mean()
            p.update(offset_warped=resid.mean(), offset_se_warped=resid.std() / np.sqrt(p["n"]),
                     offset_model_sd=resid.mean() / model_sd, resid_sd_warped=resid.std(), model_sd_warped=model_sd,
                     sd_ratio=resid.std() / model_sd,
                     fa_offset=fa_ad[sel].mean() - nm.blr.warp.invf(yhat_ref, wp).mean())
        params.append(p)

    yhat, s2 = np.full(len(X_te), np.nan), np.full(len(X_te), np.nan)
    te = np.isin(group_te, adaptable)
    if te.any():
        yhat[te], s2[te] = predict(nm, X_te[te], adaptcov=X_ad[valid], adaptresp=fa_ad[valid],
                                   adaptvargroup=group_ad[valid], testvargroup=group_te[te])
    return yhat, s2, params


def _rows_in_both(test: pd.DataFrame, adapt: pd.DataFrame) -> list[str]:
    """Test subjects whose row (everything but the subject id) also appears in the adaptation data."""
    cols = [c for c in test.columns if c != "subject"]
    return test.merge(adapt[cols].drop_duplicates(), on=cols, how="inner")["subject"].tolist()


def score(model_set: ModelSet, test: pd.DataFrame, adapt: pd.DataFrame | None = None) -> Scores:
    """Score every test row and tract, with site adaptation when a test site is not a training site."""
    warnings = []
    test_sites = sorted(set(test["site"]))
    unseen = [s for s in test_sites if s not in model_set.sites]
    adapting = bool(unseen)
    if adapting:
        if adapt is None:
            raise CliError(f"site(s) not in the training set: {', '.join(unseen)}; scoring them needs site "
                           "adaptation: pass --adapt with healthy controls from the test sites")
        counts = adapt["site"].value_counts()
        too_few = [f"{s} ({counts.get(s, 0)})" for s in test_sites if counts.get(s, 0) < MIN_ADAPT]
        if too_few:
            raise CliError(f"site adaptation needs at least {MIN_ADAPT} adaptation subjects from every test site "
                           f"(as on the portal, all test sites are adapted once one is new); too few for: "
                           f"{', '.join(too_few)}")
        unused = sorted(set(adapt["site"]) - set(test_sites))
        if unused:
            warnings.append(f"adaptation subjects from site(s) without test subjects are not used: {', '.join(unused)}")
        both = _rows_in_both(test, adapt)
        if both:
            warnings.append(f"test subject(s) also in the adaptation data, so their own FA set their norms "
                            f"(score controls with x-val): {', '.join(both)}")
    elif adapt is not None:
        warnings.append("all test sites are training sites, so the adaptation data is not used (as on the portal)")
    lo, hi = TRAINING_AGE_RANGE
    for label, df in [("test", test), ("adaptation", adapt if adapting else None)]:
        outside = df.loc[~df["age"].between(lo, hi), "subject"] if df is not None else []
        if len(outside):
            warnings.append(f"{label} age outside the training range {lo:g}-{hi:g} years: {', '.join(outside)}")

    X_te = models.design_matrix(test["age"], test["sex"], test["site"], model_set.sites)
    if adapting:
        # Site groups as the portal numbers them: test sites in sorted order
        group = {s: i for i, s in enumerate(test_sites)}
        group_te = test["site"].map(group).to_numpy(dtype=float)
        group_ad = adapt["site"].map(group).fillna(-1).to_numpy(dtype=float)
        X_ad = models.design_matrix(adapt["age"], adapt["sex"], adapt["site"], model_set.sites)

    parts, params = [], []
    for tract in model_set.tracts:
        nm = model_set.nms[tract]
        wp = warp_params(nm)
        fa = test[tract].to_numpy(dtype=float)
        if adapting:
            yhat, s2, tract_params = adapt_tract(nm, X_te, group_te, X_ad, adapt[tract].to_numpy(dtype=float),
                                                 group_ad, len(test_sites))
            for site, p in zip(test_sites, tract_params):
                params.append({"tract": tract, "site": site, **p})
                if p["n"] < MIN_ADAPT:
                    warnings.append(f"{tract} at {site}: {p['n']} adaptation value(s), need {MIN_ADAPT}; "
                                    "z-scores left blank")
        else:
            yhat, s2 = predict(nm, X_te)
        z = (nm.blr.warp.f(fa, wp) - yhat) / np.sqrt(s2)
        part = pd.DataFrame({"subject": test["subject"].to_numpy(), "tract": tract, "fa": fa, "z": z,
                             "centile": 100 * norm.cdf(z)})
        for col, zp in CENTILE_COLUMNS.items():
            part[col] = nm.blr.warp.invf(yhat + zp * np.sqrt(s2), wp)
        part["yhat_warped"] = yhat
        part["s2_warped"] = s2
        parts.append(part)
    results = pd.concat(parts, ignore_index=True)[RESULT_COLUMNS]
    adaptation = pd.DataFrame(params).reindex(columns=ADAPTATION_COLUMNS) if adapting else None
    return Scores(results, adaptation, warnings)


def z_wide(test: pd.DataFrame, results: pd.DataFrame, tracts: list[str]) -> pd.DataFrame:
    """One row per test subject: ``subject, sex, age, site`` + one z column per tract (template order)."""
    z = results.pivot(index="subject", columns="tract", values="z").reindex(index=test["subject"], columns=tracts)
    return pd.concat([test[ID_COLUMNS].reset_index(drop=True), z.reset_index(drop=True)], axis=1)


def _site_counts(df: pd.DataFrame) -> str:
    return ", ".join(f"{s} ({n})" for s, n in df["site"].value_counts().sort_index().items())


def run(test_csv: Path, out_dir: Path, adapt_csv: Path | None = None, model_dir: Path = MODEL_DIR) -> None:
    """The ``score`` subcommand: read, score, write ``z_test.csv``, ``centiles_test.csv``, ``provenance.json``
    and, with site adaptation, ``adaptation_params.csv``.

    Progress and warnings are logged; the warnings are also recorded in the provenance.
    """
    from .cohort import read_cohort

    test = read_cohort(test_csv)
    log.info("score: test %s: %d subject(s), site(s) %s", test_csv, len(test), _site_counts(test))
    adapt = read_cohort(adapt_csv) if adapt_csv is not None else None
    if adapt is not None:
        log.info("score: adapt %s: %d subject(s), site(s) %s", adapt_csv, len(adapt), _site_counts(adapt))
    model_set = ModelSet(model_dir)
    log.info("score: loaded %d tract models from %s", len(model_set.tracts), model_set.model_dir)
    results, adaptation, warnings = score(model_set, test, adapt)
    if adaptation is None:
        log.info("score: all test sites are training sites: scored without site adaptation")
    else:
        log.info("score: site adaptation (a test site is not a training site, so as on the portal every test site "
                 "is adapted)")
        for site, rows in adaptation.groupby("site"):
            log.info("score: %s: over %d tracts, median offset %+.2f model SD, median SD ratio %.2f", site,
                     len(rows), rows["offset_model_sd"].median(), rows["sd_ratio"].median())
    for warning in warnings:
        log.warning("score: %s", warning)
    missing = results["fa"].isna()
    if missing.any():
        log.info("score: %d of %d FA values missing; their z-scores are left blank", missing.sum(), len(results))
    scored = results["z"].notna()
    log.info("score: |z| > 2 for %d of %d z-scores", (results.loc[scored, "z"].abs() > 2).sum(), scored.sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = ["z_test.csv", "centiles_test.csv"]
    z_wide(test, results, model_set.tracts).to_csv(out_dir / "z_test.csv", index=False)
    results.to_csv(out_dir / "centiles_test.csv", index=False)
    if adaptation is not None:
        adaptation.to_csv(out_dir / "adaptation_params.csv", index=False)
        outputs.append("adaptation_params.csv")

    record = provenance.base_record("score", model_set.model_dir)
    record["inputs"] = {"test": {**provenance.file_info(test_csv), "n_rows": len(test)},
                        "adapt": ({**provenance.file_info(adapt_csv), "n_rows": len(adapt)}
                                  if adapt is not None else None)}
    record["mode"] = "training-sites" if adaptation is None else "site-adaptation"
    record["sites"] = sorted(set(test["site"]))
    record["unseen_sites"] = [s for s in record["sites"] if s not in model_set.sites]
    record["adaptation_subjects"] = ({s: int(n) for s, n in adapt["site"].value_counts().sort_index().items()}
                                     if adaptation is not None else None)
    record["warnings"] = warnings
    provenance.write(record, out_dir / "provenance.json")
    outputs.append("provenance.json")
    log.info("score: wrote %s to %s", ", ".join(outputs), out_dir)
