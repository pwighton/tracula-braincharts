"""Cross-validation of site adaptation on the controls (the ``x-val`` subcommand).

Scoring the controls against norms adapted with their own data gives z-scores with mean 0 and SD 1 per tract by
construction, so those say nothing. Here each control is scored against norms adapted with the *other* controls
from its site (leave-one-out by default, or K-fold within each site). That gives an out-of-sample z for every
control and tract. Adaptation only shifts each tract's norms and sets their spread, so if it suits the site these
z-scores look like N(0, 1), with no trend with age or sex.

The QA uses ``z_cal``, which corrects for the number of controls each fold adapts with (see ``scoring``). The
uncorrected z is wider than N(0, 1) even when adaptation fits perfectly: SD 1.22 with 11 controls.
"""

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import provenance, scoring
from .cli_errors import CliError
from .cohort import ID_COLUMNS
from .config import MODEL_DIR

log = logging.getLogger(__name__)

# One row per tract and site, from the cross-validated z_cal of that site's controls (target in brackets):
#   n                         controls with a z for this tract
#   z_mean, z_sd              mean (0) and SD (1, ddof 1)
#   z_skew(_se), z_kurtosis(_se)  skew (0) and excess kurtosis (0) with standard errors, as braincharts'
#                             calibration_descriptives (the FA paper's test set: -0.09 and 0.42)
#   frac_abs_z_gt2            fraction of |z| > 2 (0.046)
#   shapiro_p                 Shapiro-Wilk normality test
#   r_age, p_age              Pearson correlation with age (0): adaptation can't correct a mismatch in age trend
#   r_sex, p_sex              the same with sex (0/1)
#   rho, p_rho, rmse, smse, expv  PCNtoolkit's evaluate() in warped space, as the portal reports them
QA_COLUMNS = ["tract", "site", "n", "z_mean", "z_sd", "z_skew", "z_skew_se", "z_kurtosis", "z_kurtosis_se",
              "frac_abs_z_gt2", "shapiro_p", "r_age", "p_age", "r_sex", "p_sex", "rho", "p_rho", "rmse", "smse",
              "expv"]


def assign_folds(adapt: pd.DataFrame, n_folds: int | None = None, seed: int = 0) -> np.ndarray:
    """Fold number of each row, assigned within each site: leave-one-out when ``n_folds`` is None, else K-fold
    (rows shuffled with ``seed``, folds as equal in size as possible). Fold k holds the k-th row(s) of every site."""
    folds = np.empty(len(adapt), dtype=int)
    rng = np.random.default_rng(seed)
    for idx in adapt.groupby("site").indices.values():
        folds[idx] = np.arange(len(idx)) if n_folds is None else rng.permutation(len(idx)) % n_folds
    return folds


def check(model_set: scoring.ModelSet, adapt: pd.DataFrame, n_folds: int | None) -> None:
    """Reject inputs x-val can't cross-validate."""
    trained = sorted(set(adapt["site"]) & set(model_set.sites))
    if trained:
        raise CliError(f"x-val cross-validates site adaptation, but these are training sites, which score without "
                       f"adaptation: {', '.join(trained)}")
    if n_folds is not None and n_folds < 2:
        raise CliError("--folds must be at least 2")
    counts = adapt["site"].value_counts().sort_index()
    for site, n in counts.items():
        k = n if n_folds is None else n_folds
        if k > n or n - math.ceil(n / k) < scoring.MIN_ADAPT:
            need = scoring.MIN_ADAPT + 1 if n_folds is None else max(n_folds, math.ceil(scoring.MIN_ADAPT * k / (k - 1)))
            raise CliError(f"site {site} has {n} control(s); "
                           f"{'leave-one-out' if n_folds is None else f'{n_folds}-fold'} cross-validation needs at "
                           f"least {need}, so that every fold leaves {scoring.MIN_ADAPT} or more for adaptation")


def cross_validate(model_set: scoring.ModelSet, adapt: pd.DataFrame, folds: np.ndarray) -> scoring.Scores:
    """Score each fold's controls with the other controls of their site as adaptation data.

    Returns ``Scores`` with an extra ``fold`` column in the results (one row per control and tract) and in the
    adaptation table (each fold's adaptation parameters); warnings are de-duplicated.
    """
    parts, params, warnings = [], [], []
    for k in np.unique(folds):
        test = adapt[folds == k]
        ad = adapt[(folds != k) & adapt["site"].isin(test["site"])]
        s = scoring.score(model_set, test, ad)
        parts.append(s.results.assign(fold=int(k)))
        params.append(s.adaptation.assign(fold=int(k)))
        warnings += [w for w in s.warnings if w not in warnings]
    results = pd.concat(parts, ignore_index=True)[["subject", "fold", *scoring.RESULT_COLUMNS[1:]]]
    adaptation = pd.concat(params, ignore_index=True)[["fold", *scoring.ADAPTATION_COLUMNS]]
    return scoring.Scores(results, adaptation, warnings)



def _calibration_descriptives(z: np.ndarray) -> tuple[float, float, float, float]:
    """Skew, its SE, excess kurtosis, its SE: braincharts ``scripts/nm_utils.py::calibration_descriptives``."""
    n = len(z)
    d = z - z.mean()
    m2, m3, m4 = (d ** 2).sum(), (d ** 3).sum(), (d ** 4).sum()
    s1 = z.std()
    skew = n * m3 / (n - 1) / (n - 2) / s1 ** 3
    skew_se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    kurtosis = (n * (n + 1) * m4 - 3 * m2 ** 2 * (n - 1)) / ((n - 1) * (n - 2) * (n - 3) * s1 ** 4)
    kurtosis_se = np.sqrt(4 * (n ** 2 - 1) * skew_se ** 2 / ((n - 3) * (n + 5)))
    return skew, skew_se, kurtosis, kurtosis_se


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    r = stats.pearsonr(x, y)
    return float(r.statistic), float(r.pvalue)


def qa_adaptation(cv: pd.DataFrame, adapt: pd.DataFrame, tracts: list[str]) -> pd.DataFrame:
    """``QA_COLUMNS`` for every tract and site, from ``cross_validate``'s results."""
    from pcntoolkit.normative import evaluate

    data = cv.merge(adapt[["subject", "sex", "age", "site"]], on="subject").dropna(subset=["z"])
    rows = []
    for site in sorted(set(adapt["site"])):
        for tract in tracts:
            g = data[(data["site"] == site) & (data["tract"] == tract)]
            z = g["z_cal"].to_numpy()
            row = {"tract": tract, "site": site, "n": len(z)}
            if len(z) >= 3:
                row.update(z_mean=z.mean(), z_sd=z.std(ddof=1), frac_abs_z_gt2=np.mean(np.abs(z) > 2),
                           shapiro_p=stats.shapiro(z).pvalue)
                row["r_age"], row["p_age"] = _pearson(g["age"].to_numpy(dtype=float), z)
                row["r_sex"], row["p_sex"] = _pearson(g["sex"].to_numpy(dtype=float), z)
                # warped FA = yhat + (uncorrected) z * sd, exactly
                yhat = g["yhat_warped"].to_numpy()
                y = yhat + g["z"].to_numpy() * np.sqrt(g["s2_warped"].to_numpy())
                m = evaluate(y[:, None], yhat[:, None], metrics=["Rho", "RMSE", "SMSE", "EXPV"])
                row.update(rho=m["Rho"][0], p_rho=m["pRho"][0], rmse=m["RMSE"][0], smse=m["SMSE"][0],
                           expv=m["EXPV"][0])
            if len(z) >= 4:
                row["z_skew"], row["z_skew_se"], row["z_kurtosis"], row["z_kurtosis_se"] = _calibration_descriptives(z)
            rows.append(row)
    return pd.DataFrame(rows).reindex(columns=QA_COLUMNS)


def centile_curves(model_set: scoring.ModelSet, adapt_site: pd.DataFrame, n_ages: int = 60) -> pd.DataFrame:
    """Centile curves adapted with all of one site's controls, over their age range (±2 years), for both sexes.

    Long format: ``tract, age, sex`` + ``scoring.CENTILE_COLUMNS`` + ``scoring.CAL_CENTILE_COLUMNS``.
    """
    lo = max(scoring.TRAINING_AGE_RANGE[0], adapt_site["age"].min() - 2)
    hi = min(scoring.TRAINING_AGE_RANGE[1], adapt_site["age"].max() + 2)
    grid = pd.DataFrame([(sex, age) for sex in (0, 1) for age in np.linspace(lo, hi, n_ages)], columns=["sex", "age"])
    grid.insert(0, "subject", [f"grid{i}" for i in range(len(grid))])
    grid["site"] = adapt_site["site"].iloc[0]
    grid = grid.reindex(columns=list(adapt_site.columns))  # FA columns empty: only the centiles are needed
    res = scoring.score(model_set, grid, adapt_site).results
    res = res.merge(grid[["subject", "sex", "age"]], on="subject")
    return res[["tract", "age", "sex", *scoring.CENTILE_COLUMNS, *scoring.CAL_CENTILE_COLUMNS]]


def _file_name(site: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", site)


def run(adapt_csv: Path, out_dir: Path, n_folds: int | None = None, seed: int = 0,
        model_dir: Path = MODEL_DIR) -> None:
    """The ``x-val`` subcommand: write ``z_adapt_cv.csv``, ``z_cal_adapt_cv.csv``, ``centiles_adapt_cv.csv``,
    ``qa_adaptation.csv``, ``plots/centiles_<site>.png`` and ``provenance.json``."""
    from . import plots
    from .cohort import read_cohort

    adapt = read_cohort(adapt_csv)
    counts = adapt["site"].value_counts().sort_index()
    log.info("x-val: adapt %s: %d control(s), site(s) %s", adapt_csv, len(adapt),
             ", ".join(f"{s} ({n})" for s, n in counts.items()))
    model_set = scoring.ModelSet(model_dir)
    log.info("x-val: loaded %d tract models from %s", len(model_set.tracts), model_set.model_dir)
    check(model_set, adapt, n_folds)
    scheme = "leave-one-out" if n_folds is None else f"{n_folds}-fold within each site (seed {seed})"
    folds = assign_folds(adapt, n_folds, seed)
    log.info("x-val: %s cross-validation: %d fold(s)", scheme, len(np.unique(folds)))
    cv = cross_validate(model_set, adapt, folds)
    for warning in cv.warnings:
        log.warning("x-val: %s", warning)
    qa = qa_adaptation(cv.results, adapt, model_set.tracts)
    for site, g in qa.groupby("site"):
        rows = cv.results[cv.results["subject"].isin(adapt.loc[adapt["site"] == site, "subject"])].dropna(subset=["z"])
        log.info("x-val: %s: z_cal over %d tracts: median mean %+.2f (0 expected), median SD %.2f (1 expected); "
                 "|z_cal| > 2 for %.1f%% (4.6%% expected; uncorrected |z| > 2 for %.1f%%); correlates with age at "
                 "p < 0.05 in %d tracts (%.1f expected by chance)", site, len(g), g["z_mean"].median(),
                 g["z_sd"].median(), 100 * np.mean(rows["z_cal"].abs() > 2), 100 * np.mean(rows["z"].abs() > 2),
                 (g["p_age"] < 0.05).sum(), 0.05 * g["p_age"].notna().sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    wide = scoring.z_wide(adapt, cv.results, model_set.tracts)
    wide.insert(len(ID_COLUMNS), "fold", folds)
    wide.to_csv(out_dir / "z_adapt_cv.csv", index=False)
    wide_cal = scoring.z_wide(adapt, cv.results, model_set.tracts, "z_cal")
    wide_cal.insert(len(ID_COLUMNS), "fold", folds)
    wide_cal.to_csv(out_dir / "z_cal_adapt_cv.csv", index=False)
    cv.results.to_csv(out_dir / "centiles_adapt_cv.csv", index=False)
    qa.to_csv(out_dir / "qa_adaptation.csv", index=False)
    outputs = ["z_adapt_cv.csv", "z_cal_adapt_cv.csv", "centiles_adapt_cv.csv", "qa_adaptation.csv"]

    (out_dir / "plots").mkdir(exist_ok=True)
    points = cv.results.merge(adapt[["subject", "sex", "age"]], on="subject")
    for site in counts.index:
        site_adapt = adapt[adapt["site"] == site]
        curves = centile_curves(model_set, site_adapt)
        name = f"plots/centiles_{_file_name(site)}.png"
        plots.centile_panels(curves, points[points["subject"].isin(site_adapt["subject"])], model_set.tracts,
                             f"{site}: {len(site_adapt)} controls, cross-validated, over centile curves adapted "
                             f"with all of them (5th/95th corrected for {len(site_adapt)} controls)", out_dir / name)
        outputs.append(name)

    record = provenance.base_record("x-val", model_set.model_dir)
    record["inputs"] = {"adapt": {**provenance.file_info(adapt_csv), "n_rows": len(adapt)}}
    record["cross_validation"] = {"scheme": "leave-one-out" if n_folds is None else "k-fold", "k": n_folds,
                                  "seed": seed if n_folds is not None else None}
    record["sites"] = {s: int(n) for s, n in counts.items()}
    record["warnings"] = cv.warnings
    provenance.write(record, out_dir / "provenance.json")
    outputs.append("provenance.json")
    log.info("x-val: wrote %s to %s", ", ".join(outputs), out_dir)
