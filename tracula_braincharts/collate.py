"""The ``collate`` subcommand: from a manifest of ``extract`` outputs to ``adapt.csv`` / ``test.csv`` and cohort QA
tables.

Manifest columns: ``subject, extract_dir, age, sex, site, role`` (role ``adapt``, ``test`` or ``exclude``; other
columns are ignored). ``extract_dir`` is an ``extract`` output folder (``features.json``, ``provenance.json``,
``label_qc.csv``), relative to the manifest's folder unless absolute. Excluded subjects need no outputs; if they
have them, they appear in the QA tables.

``adapt.csv`` / ``test.csv`` are in the portal's template format (``sex,age,site,race`` + the 48 tracts in template
order, ``race`` empty) with a leading ``subject`` column, which the portal ignores (it picks the tract columns by
name, in file order) and ``score`` / ``x-val`` read as ids.

QA tables, for visual review rather than automatic exclusion: ``qa_summary.csv`` (one row per subject, the scalar
metrics from ``provenance.json``), ``qa_labels.csv`` (``label_qc.csv`` of every subject) and ``qa_flags.csv``:
``extract``'s hard flags and warnings, plus cohort-relative flags against the adaptation subjects of the same site
(each adaptation subject against the others): selected QA metrics, and the tract values (ENIGMA's eDTI_outliers
rule), as prediction t statistics, which are exact for a few controls where percentiles and robust z aren't.
"""

import json
import logging
import warnings as pywarnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from . import models, provenance
from .cli_errors import CliError
from .cohort import TRACTS, _list, check_cohort, read_cohort
from .config import MODEL_DIR, WARN_ADAPT_BELOW
from .scoring import MIN_ADAPT

log = logging.getLogger(__name__)

ROLES = ("adapt", "test", "exclude")
MANIFEST_COLUMNS = ["subject", "extract_dir", "age", "sex", "site", "role"]
COHORT_COLUMNS = ["subject", "sex", "age", "site", "race", *TRACTS]

# Cohort-relative flags, against the adaptation subjects of the subject's site (when it has at least MIN_REFERENCE;
# an adaptation subject is compared with the others). The statistic is the prediction t of a new observation,
# (x - mean) / (SD sqrt(1 + 1/n)) ~ t(n - 1) for normal data, so its tails are right for a few controls; a robust z
# (median / MAD) from 12 controls flagged ~1 in 5 typical subjects in simulation.
MIN_REFERENCE = 5
# QA metrics: flagged beyond the one-sided tail probability of 3 SD
QA_P = norm.sf(3)
# QA metric (dotted key in provenance.json's "qa") -> the direction that is a problem
QA_METRICS = {
    "fa.mask_volume_ml": "both",
    "fa.fa_above_1_pct": "high",
    "fa.wm_median_fa": "both",
    "fa.b0_tsnr_median": "low",
    "eddy.movement_rms.fa_volumes.abs_mean_mm": "high",
    "eddy.movement_rms.fa_volumes.rel_mean_mm": "high",
    "eddy.movement_rms.fa_volumes.rel_max_mm": "high",
    "eddy.restricted_movement_rms.fa_volumes.rel_mean_mm": "high",
    "eddy.outliers.fa_dwi_slices_pct": "high",
    "eddy.cnr.shell_cnr_median": "low",
    "registration.r_wm": "low",
    "registration.template_brain_coverage_pct": "low",
    "registration.flirt.average_scaling": "both",
    "registration.jacobian.p01": "low",
    "registration.jacobian.p99": "high",
    "skeleton.zero_pct": "high",
    "skeleton.below_threshold_pct": "high",
    "skeleton.projection_gain_mean": "high",
}
# Tract values: ENIGMA's eDTI_outliers flags a subject with >= 5% of its tract values outside the 0.1-99.9
# percentiles of the cohort, or >= 10% outside 0.5-99.5. Those percentiles need hundreds of subjects; for a site's
# few controls they are just the min and max, outside which a typical new subject has ~2/(n+1) of its tracts. So
# "outside the percentiles" is a two-sided prediction p below 0.002 / 0.01 instead: (percentiles, share of tracts)
TRACT_RULES = [((0.1, 99.9), 0.05), ((0.5, 99.5), 0.10)]
FLAG_COLUMNS = ["subject", "role", "site", "source", "flag", "detail"]


@dataclass
class Extracted:
    """One subject's ``extract`` outputs."""
    dir: Path
    features: dict
    provenance: dict
    label_qc: pd.DataFrame | None


@dataclass
class Collated:
    adapt: pd.DataFrame
    test: pd.DataFrame
    qa_summary: pd.DataFrame
    qa_labels: pd.DataFrame
    qa_flags: pd.DataFrame
    settings: dict
    adapting_sites: list[str]
    warnings: list[str] = field(default_factory=list)


def read_manifest(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"subject": str, "site": str, "extract_dir": str, "role": str})
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise CliError(f"{path}: missing columns: {_list(missing)}")
    if df.empty:
        raise CliError(f"{path}: no rows")
    bad = df.index[df["subject"].isna() | (df["subject"].str.strip() == "")]
    if len(bad):
        raise CliError(f"{path}: empty subject id in row(s) {_list(bad + 1)}")
    df["subject"] = df["subject"].str.strip()
    dup = df.loc[df["subject"].duplicated(), "subject"]
    if not dup.empty:
        raise CliError(f"{path}: duplicate subject ids (one row per subject, one role each): {_list(dup.unique())}")
    df["role"] = df["role"].fillna("").str.strip().str.lower()
    bad = df.loc[~df["role"].isin(ROLES), "subject"]
    if not bad.empty:
        raise CliError(f"{path}: role must be one of {', '.join(ROLES)}; bad rows: {_list(bad)}")
    return df.reset_index(drop=True)


def load_extracted(path: Path) -> Extracted:
    def read_json(name):
        try:
            return json.loads((path / name).read_text())
        except (OSError, ValueError) as exc:
            raise CliError(f"cannot read {path / name} (run extract first?): {exc}") from exc

    features, prov = read_json("features.json"), read_json("provenance.json")
    if set(features.get("features", {})) != set(TRACTS):
        raise CliError(f"{path / 'features.json'} doesn't have the template's 48 tracts")
    qc_path = path / "label_qc.csv"
    return Extracted(path, features, prov, pd.read_csv(qc_path) if qc_path.is_file() else None)


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested dict -> {dotted key: value} for numbers, booleans and nulls; lists of numbers become key.0, key.1, ...
    Strings (paths, reasons) are left out."""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(value, name + ".")
        elif isinstance(value, list):
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                out |= {f"{name}.{i}": v for i, v in enumerate(value)}
        elif value is None or isinstance(value, (bool, int, float)):
            out[name] = value
    return out


def prediction_t(x, ref) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise (x - mean) / (SD sqrt(1 + 1/n)) of ``ref`` (subjects in rows, missing values left out): t(n - 1)
    for a new observation from ``ref``'s normal distribution. Returns (t, n, mean, SD); t is NaN where n < 3 or the
    SD is 0 (to rounding)."""
    ref = np.asarray(ref, dtype=float)
    n = np.isfinite(ref).sum(axis=0)
    with pywarnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        pywarnings.simplefilter("ignore", RuntimeWarning)
        mean, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0, ddof=1)
        t = (np.asarray(x, dtype=float) - mean) / (sd * np.sqrt(1 + 1 / n))
    varies = sd > 1e-9 * np.maximum(1, np.abs(mean))
    return np.where((n >= 3) & varies, t, np.nan), n, mean, np.where(varies, sd, 0.0)


def tract_outliers(values: np.ndarray, ref: np.ndarray) -> list[tuple[str, str]]:
    """``TRACT_RULES`` on one subject's tract values against the reference subjects' (rows): (flag, detail) for
    each rule met. Missing tracts are left out."""
    t, n, _mean, _sd = prediction_t(values, ref)
    ok = np.isfinite(t)
    p = 2 * student_t.sf(np.abs(t[ok]), n[ok] - 1)
    out = []
    for (lo, hi), share in TRACT_RULES:
        k = int((p < 2 * lo / 100).sum())
        if ok.any() and k >= share * ok.sum():
            out.append(("tract_outliers", f"{k} of {int(ok.sum())} tracts outside the adaptation subjects' "
                                          f"{lo:g}-{hi:g} percentile prediction interval (rule: >= {share:.0%})"))
    return out


def metric_outlier(x: float, ref: np.ndarray, direction: str) -> str | None:
    """Detail text when ``x`` is beyond the ``QA_P`` tail of ``ref`` in a problem ``direction`` (high, low, both).
    When the reference subjects all have the same value, any difference in a problem direction is reported."""
    t, n, mean, sd = (float(v) for v in prediction_t(x, ref))
    if n >= MIN_REFERENCE - 1 and sd == 0:
        diff = x - mean
        bad = abs(diff) > 1e-9 * max(1, abs(mean)) and ((diff > 0 and direction in ("high", "both")) or
                                                          (diff < 0 and direction in ("low", "both")))
        return f"{x:g}; every adaptation subject has {mean:g} (n {int(n)})" if bad else None
    if not np.isfinite(t):
        return None
    p = student_t.sf(abs(t), n - 1)
    if p < QA_P and ((t > 0 and direction in ("high", "both")) or (t < 0 and direction in ("low", "both"))):
        return f"{x:g} vs adaptation subjects {mean:g} +/- {sd:g} (t {t:+.1f}, n {int(n)}, p {p:.1g})"
    return None


def _settings(prov: dict, features: dict) -> dict:
    """What has to match across subjects for their features to be comparable."""
    tbss = prov.get("tbss", {})
    return {"skeleton": features.get("skeleton"), "fsl_version": tbss.get("fsl_version"),
            "threshold": tbss.get("threshold"),
            "atlas_and_templates": {k: v.get("sha256") for k, v in sorted(tbss.get("files", {}).items())}}


def _protocol(prov: dict) -> str:
    acq = prov.get("acquisition", {})
    shells = ", ".join(f"{s['b']:.0f}" for s in acq.get("shells", []))
    delta = acq.get("diffusion_time", {}).get("delta_ms")
    return f"b={shells or '?'} (Delta {delta if delta is not None else '?'} ms)"


def collate(manifest: pd.DataFrame, base: Path, training_sites: list[str], source: str = "manifest",
            warn_adapt_below: int = WARN_ADAPT_BELOW) -> Collated:
    """The tables ``run`` writes, from a validated manifest (``read_manifest``); ``base`` resolves relative
    ``extract_dir``s, ``source`` names the manifest in errors."""
    warnings: list[str] = []
    used = manifest[manifest["role"] != "exclude"]
    if used.empty:
        raise CliError(f"{source}: no adapt or test rows")

    # extract outputs: required for adapt/test rows, used when present for excluded ones
    extracted: dict[str, Extracted] = {}
    dirs = extract_dirs(manifest, base)
    for row in manifest.itertuples():
        path = dirs.get(row.subject)
        if row.role == "exclude":
            if path is not None and (path / "provenance.json").is_file():
                extracted[row.subject] = load_extracted(path)
            continue
        if path is None:
            raise CliError(f"{source}: no extract_dir for {row.subject}")
        extracted[row.subject] = load_extracted(path)
        named = extracted[row.subject].features.get("subject")
        if named != row.subject:
            warnings.append(f"{row.subject}: extract_dir {path} holds subject {named!r}")

    # extraction settings must match across adapt and test subjects
    settings = {s: _settings(extracted[s].provenance, extracted[s].features) for s in used["subject"]}
    for key in ("skeleton", "fsl_version", "threshold", "atlas_and_templates"):
        groups: dict[str, list[str]] = {}
        for subject, st in settings.items():
            value = st[key]
            if isinstance(value, dict):  # file -> sha256
                value = ", ".join(f"{Path(f).name} {h[:12] if h else h}" for f, h in value.items())
            groups.setdefault(str(value), []).append(subject)
        if len(groups) > 1:
            detail = "; ".join(f"{k} ({_list(v, 3)})" for k, v in groups.items())
            raise CliError(f"subjects were extracted with different {key}: {detail}; re-run extract the same way")
    versions = {extracted[s].provenance.get("version") for s in used["subject"]}
    if len(versions) > 1:
        warnings.append(f"extracted with different tracula-braincharts versions: {', '.join(map(str, versions))}")

    # cohort table, validated as score / x-val will read it
    rows = []
    for row in used.itertuples():
        feats = extracted[row.subject].features["features"]
        rows.append({"subject": row.subject, "sex": row.sex, "age": row.age, "site": row.site, "race": np.nan,
                     **{t: np.nan if feats[t] is None else feats[t] for t in TRACTS}})
    cohort = check_cohort(pd.DataFrame(rows, columns=COHORT_COLUMNS), source)
    cohort.insert(4, "race", np.nan)
    role = dict(zip(manifest["subject"], manifest["role"]))
    adapt = cohort[cohort["subject"].map(role) == "adapt"].reset_index(drop=True)
    test = cohort[cohort["subject"].map(role) == "test"].reset_index(drop=True)

    # site adaptation, as score will do it: once one test site is new, every test site is adapted
    test_sites = sorted(set(test["site"]))
    counts = adapt["site"].value_counts().to_dict()
    unseen = [s for s in test_sites if s not in training_sites]
    adapting = test_sites if unseen else []
    too_few = [f"{s} ({counts.get(s, 0)})" for s in adapting if counts.get(s, 0) < MIN_ADAPT]
    if too_few:
        raise CliError(f"site(s) {', '.join(unseen)} aren't training sites, so every test site is adapted and needs "
                       f"at least {MIN_ADAPT} adapt rows; too few for: {', '.join(too_few)}")
    few = [f"{s} ({counts[s]})" for s in adapting if counts[s] < warn_adapt_below]
    if few:
        warnings.append(f"fewer than {warn_adapt_below} adaptation subjects at: {', '.join(few)} (the adapted "
                        "offsets and SDs are uncertain; z_cal accounts for the number, not for unrepresentative "
                        "controls)")
    if not test.empty and not adapting and not adapt.empty:
        warnings.append("every test site is a training site: score won't use adapt.csv (x-val can)")
    unused = sorted(set(adapt["site"]) - set(test_sites)) if not test.empty else []
    if unused:
        warnings.append(f"adapt rows from site(s) without test subjects: {', '.join(unused)} (used by x-val only)")
    n_missing = int(adapt[TRACTS].isna().sum().sum())
    if n_missing:
        warnings.append(f"{n_missing} missing tract value(s) in adapt.csv: score leaves those subjects out of that "
                        "tract's adaptation; the portal would blank the whole site x tract")
    for site in adapting:
        ad, te = adapt[adapt["site"] == site], test[test["site"] == site]
        outside = te.loc[~te["age"].between(ad["age"].min(), ad["age"].max()), "subject"]
        if not outside.empty:
            warnings.append(f"{site}: test subject(s) outside the adaptation ages {ad['age'].min():g}-"
                            f"{ad['age'].max():g}: {_list(outside)}")
        protocols = {s: _protocol(extracted[s].provenance) for s in pd.concat([ad, te])["subject"]}
        ad_protocols = {protocols[s] for s in ad["subject"]}
        other = [s for s in te["subject"] if protocols[s] not in ad_protocols]
        if other:
            warnings.append(f"{site}: test subject(s) acquired differently from the adaptation subjects "
                            f"({', '.join(sorted(ad_protocols))}): " + _list(f"{s} {protocols[s]}" for s in other))

    qa_summary, qa_labels, qa_flags = qa_tables(manifest, extracted, adapt)
    return Collated(adapt, test, qa_summary, qa_labels, qa_flags,
                    settings=next(iter(settings.values())), adapting_sites=adapting, warnings=warnings)


def qa_tables(manifest: pd.DataFrame, extracted: dict[str, Extracted],
              adapt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = manifest.set_index("subject")
    summary, labels, flags = [], [], []
    for subject, ex in extracted.items():
        site, role = info.at[subject, "site"], info.at[subject, "role"]
        qa = ex.provenance.get("qa", {})
        found = qa.get("flags", {})
        summary.append({"subject": subject, "role": role, "site": site, "age": info.at[subject, "age"],
                        "sex": info.at[subject, "sex"], "n_extract_flags": len(found),
                        "extract_flags": ";".join(found),
                        **flatten({k: v for k, v in qa.items() if k not in ("flags", "images")})})
        if ex.label_qc is not None:
            labels.append(ex.label_qc.assign(subject=subject, role=role, site=site))
        flags += [(subject, role, site, "extract", name, detail) for name, detail in found.items()]
        flags += [(subject, role, site, "extract", "warning", w) for w in ex.provenance.get("warnings", [])
                  if not w.startswith("QA ")]  # the QA flags are already listed
    summary = pd.DataFrame(summary)

    # cohort-relative flags against each site's adaptation subjects (each of them against the others)
    for site, ref_subjects in adapt.groupby("site")["subject"]:
        if len(ref_subjects) < MIN_REFERENCE:
            log.info("collate: %s has %d adaptation subjects (< %d): no cohort-relative QA flags there", site,
                     len(ref_subjects), MIN_REFERENCE)
            continue
        ref_summary = summary[summary["subject"].isin(ref_subjects)]
        ref_fa = adapt[adapt["site"] == site].set_index("subject")[TRACTS]
        for row in summary[summary["site"] == site].to_dict("records"):
            s = row["subject"]
            others = ref_summary[ref_summary["subject"] != s]
            for metric, direction in QA_METRICS.items():
                if metric in summary.columns and pd.notna(row.get(metric)):
                    detail = metric_outlier(float(row[metric]), others[metric].to_numpy(dtype=float), direction)
                    if detail:
                        flags.append((s, row["role"], site, "cohort", metric, detail))
            fa = np.array([np.nan if v is None else v for v in map(extracted[s].features["features"].get, TRACTS)])
            flags += [(s, row["role"], site, "cohort", name, detail)
                      for name, detail in tract_outliers(fa, ref_fa.drop(index=s, errors="ignore").to_numpy())]
    flags = pd.DataFrame(flags, columns=FLAG_COLUMNS)
    labels = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame()
    if not labels.empty:
        ids = ["subject", "role", "site"]
        labels = labels[ids + [c for c in labels.columns if c not in ids]]
    return summary, labels, flags


OUTPUTS = ["adapt.csv", "test.csv", "qa_summary.csv", "qa_labels.csv", "qa_flags.csv", "provenance.json"]


def run(manifest_path: Path, out_dir: Path, model_dir: Path = MODEL_DIR,
        warn_adapt_below: int = WARN_ADAPT_BELOW) -> None:
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    counts = manifest["role"].value_counts()
    log.info("collate: %s: %s", manifest_path, ", ".join(f"{counts[r]} {r}" for r in ROLES if r in counts))
    result = collate(manifest, manifest_path.parent, models.training_sites(model_dir), source=str(manifest_path),
                     warn_adapt_below=warn_adapt_below)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in (("adapt.csv", result.adapt), ("test.csv", result.test)):
        path = out_dir / name
        if df.empty:
            if path.exists():
                path.unlink()
                log.info("collate: removed %s from an earlier run (no %s rows now)", name, name.split(".")[0])
            continue
        df[COHORT_COLUMNS].to_csv(path, index=False)
        read_cohort(path)  # as score / x-val will read it
        written.append(name)
    for name, df in (("qa_summary.csv", result.qa_summary), ("qa_labels.csv", result.qa_labels),
                     ("qa_flags.csv", result.qa_flags)):
        df.to_csv(out_dir / name, index=False)
        written.append(name)
    for w in result.warnings:
        log.warning("collate: %s", w)

    record = provenance.base_record("collate", model_dir)
    record["manifest"] = provenance.file_info(manifest_path)
    record["subjects"] = {r: {s: int(n) for s, n in g["site"].value_counts().sort_index().items()}
                          for r, g in manifest.groupby("role")}
    record["extraction"] = result.settings
    record["adapting_sites"] = result.adapting_sites
    record["extract_outputs"] = {s: provenance.file_info(p / "features.json")
                                 for s, p in extract_dirs(manifest, manifest_path.parent).items()
                                 if (p / "features.json").is_file()}
    record["qa_flags"] = result.qa_flags.groupby("source").size().to_dict() if not result.qa_flags.empty else {}
    record["warnings"] = result.warnings
    provenance.write(record, out_dir / "provenance.json")
    written.append("provenance.json")

    flagged = result.qa_flags["subject"].nunique() if not result.qa_flags.empty else 0
    log.info("collate: %d adapt, %d test subjects; %s; %d QA flag(s) on %d subject(s) (qa_flags.csv); wrote %s to %s",
             len(result.adapt), len(result.test),
             f"adapting {', '.join(result.adapting_sites)}" if result.adapting_sites else "no site adaptation",
             len(result.qa_flags), flagged, ", ".join(written), out_dir)


def extract_dirs(manifest: pd.DataFrame, base: Path) -> dict[str, Path]:
    """subject -> extract output folder (relative paths are relative to ``base``, the manifest's folder)."""
    out = {}
    for row in manifest.itertuples():
        if pd.notna(row.extract_dir):
            path = Path(str(row.extract_dir))
            out[row.subject] = path if path.is_absolute() else base / path
    return out
