"""collate: manifest -> adapt.csv / test.csv and the cohort QA tables.

The extract outputs are synthetic, made from the NEW_SITE fixtures (12 controls, one missing value; 3 test
subjects), so adapt.csv / test.csv must give those files back.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from tracula_braincharts import collate
from tracula_braincharts.cli import main
from tracula_braincharts.cli_errors import CliError
from tracula_braincharts.cohort import TRACTS, read_cohort

ADAPT, TEST = read_cohort(FIXTURES / "adapt-new-site.csv"), read_cohort(FIXTURES / "test-new-site.csv")
TRAINING = ["HCPA_3002", "HCPYA_Washington"]


def write_extract(path, subject, row, i=0, skeleton="fmrib58", delta=49.0, flags=None, warnings=(), **qa):
    """An extract output folder with features from ``row`` and plausible QA metrics varying with ``i``; ``qa``
    overrides metrics by dotted key (``registration.r_wm=0.5``)."""
    path.mkdir(parents=True)
    features = {t: None if pd.isna(row[t]) else float(row[t]) for t in TRACTS}
    (path / "features.json").write_text(json.dumps({"subject": subject, "skeleton": skeleton, "features": features}))
    rng = np.random.default_rng(i)
    q = {"fa": {"mask_volume_ml": 2200 + 40 * rng.normal(), "nonfinite_voxels": 0, "fa_above_1_pct": 0.4,
                "wm_median_fa": 0.32 + 0.01 * rng.normal(), "b0_tsnr_median": 22 + rng.normal(), "mask": "dtifit -m"},
         "eddy": {"movement_rms": {"fa_volumes": {"abs_mean_mm": 1 + 0.2 * rng.normal(),
                                                  "rel_mean_mm": 0.7 + 0.05 * rng.normal(), "rel_max_mm": 2.0}}},
         "registration": {"r_wm": 0.86 + 0.01 * rng.normal(), "template_brain_coverage_pct": 99.7,
                          "flirt": {"scales": [0.88, 0.96, 0.86], "average_scaling": 0.9 + 0.01 * rng.normal()},
                          "jacobian": {"p01": 0.2 + 0.02 * rng.normal(), "p99": 1.6 + 0.05 * rng.normal(),
                                       "file": "FA/x_jac.nii.gz"}},
         "skeleton": {"zero_pct": 0.06, "below_threshold_pct": 8.5 + rng.normal(), "projection_gain_mean": 0.033},
         "flags": flags or {}, "images": ["qc/registration.png", "qc/skeleton.png"]}
    for key, value in qa.items():
        *parents, leaf = key.split("__")
        d = q
        for k in parents:
            d = d[k]
        d[leaf] = value
    prov = {"version": "0.1.0.dev0", "warnings": [*warnings, *(f"QA {k}: {v}" for k, v in (flags or {}).items())],
            "acquisition": {"shells": [{"b": 950.0, "n": 32}], "diffusion_time": {"delta_ms": delta}},
            "tbss": {"fsl_version": "6.0.5.2:dc6f4207", "threshold": 0.2,
                     "files": {"data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz": {"sha256": "abc"}}},
            "qa": q}
    (path / "provenance.json").write_text(json.dumps(prov))
    pd.DataFrame({"tract": TRACTS[:2], "label": [1, 2], "coverage_pct": [100.0, 100.0]}).to_csv(
        path / "label_qc.csv", index=False)
    return path


@pytest.fixture
def cohort_dir(tmp_path):
    """extract outputs under tmp_path/extract/<subject>, and the manifest rows for them (extract_dir relative)."""
    rows = []
    for role, df in (("adapt", ADAPT), ("test", TEST)):
        for i, row in df.iterrows():
            write_extract(tmp_path / "extract" / row["subject"], row["subject"], row, i=i + (100 if role == "test"
                                                                                           else 0))
            rows.append({"subject": row["subject"], "extract_dir": f"extract/{row['subject']}", "age": row["age"],
                         "sex": row["sex"], "site": row["site"], "role": role})
    return tmp_path, pd.DataFrame(rows)


def run(base, manifest, **kw):
    manifest.to_csv(base / "manifest.csv", index=False)
    return collate.collate(collate.read_manifest(base / "manifest.csv"), base, TRAINING, **kw)


def test_round_trip(cohort_dir):
    base, manifest = cohort_dir
    out = run(base, manifest)
    pd.testing.assert_frame_equal(out.adapt.drop(columns="race"), ADAPT, check_dtype=False)
    pd.testing.assert_frame_equal(out.test.drop(columns="race"), TEST, check_dtype=False)
    assert list(out.adapt.columns) == collate.COHORT_COLUMNS and out.adapt["race"].isna().all()
    assert out.adapting_sites == ["NEW_SITE"] and out.settings["skeleton"] == "fmrib58"
    assert any("1 missing tract value" in w for w in out.warnings)
    assert len(out.qa_summary) == 15 and out.qa_summary["n_extract_flags"].eq(0).all()
    assert "registration.flirt.scales.2" in out.qa_summary.columns and "fa.mask" not in out.qa_summary.columns
    assert set(out.qa_labels.columns[:3]) == {"subject", "role", "site"} and len(out.qa_labels) == 30


def test_cli(cohort_dir, tmp_path):
    base, manifest = cohort_dir
    manifest.to_csv(base / "manifest.csv", index=False)
    model_dir = tmp_path / "models"
    (model_dir / "docs").mkdir(parents=True)
    (model_dir / "docs" / "site_ids_fa_19sites.txt").write_text("\n".join(TRAINING) + "\n")
    out = tmp_path / "out"
    (out).mkdir()
    (out / "test.csv").write_text("stale")
    assert main(["collate", str(base / "manifest.csv"), "-o", str(out), "--model-dir", str(model_dir)]) == 0
    pd.testing.assert_frame_equal(read_cohort(out / "adapt.csv"), ADAPT)
    pd.testing.assert_frame_equal(read_cohort(out / "test.csv"), TEST)
    for name in ("qa_summary.csv", "qa_labels.csv", "qa_flags.csv", "provenance.json", "collate.log"):
        assert (out / name).is_file()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["subjects"] == {"adapt": {"NEW_SITE": 12}, "test": {"NEW_SITE": 3}}
    assert prov["adapting_sites"] == ["NEW_SITE"] and len(prov["extract_outputs"]) == 15
    assert "collate: 12 adapt, 3 test subjects; adapting NEW_SITE" in (out / "collate.log").read_text()

    # no test rows now: the old test.csv goes
    manifest[manifest["role"] == "adapt"].to_csv(base / "manifest.csv", index=False)
    assert main(["collate", str(base / "manifest.csv"), "-o", str(out), "--model-dir", str(model_dir)]) == 0
    assert (out / "adapt.csv").is_file() and not (out / "test.csv").exists()


@pytest.mark.parametrize("change, message", [
    (lambda m, b: m.drop(columns="role"), "missing columns: role"),
    (lambda m, b: pd.concat([m, m.iloc[[0]]]), "duplicate subject ids"),
    (lambda m, b: m.assign(role=m["role"].where(m.index != 0, "control")), "role must be one of"),
    (lambda m, b: m.assign(role="exclude"), "no adapt or test rows"),
    (lambda m, b: m.assign(extract_dir=m["extract_dir"].where(m.index != 0, "extract/nobody")),
     "cannot read .*nobody.*features.json"),
    (lambda m, b: m.assign(sex=m["sex"].where(m.index != 0, 2)), "sex must be 0"),
    (lambda m, b: m[(m["role"] == "test") | (m.index == 0)], r"too few for: NEW_SITE \(1\)"),
    # a training site among the test sites is adapted too, so it needs adapt rows
    (lambda m, b: pd.concat([m, m[m["role"] == "test"].iloc[[0]].assign(subject="t9", site="HCPA_3002")]),
     r"too few for: HCPA_3002 \(0\)"),
])
def test_errors(cohort_dir, change, message):
    base, manifest = cohort_dir
    with pytest.raises(CliError, match=message):
        run(base, change(manifest, base))


def test_extraction_settings_must_match(cohort_dir):
    base, manifest = cohort_dir
    row = TEST.iloc[0]
    write_extract(base / "extract" / "odd", row["subject"], row, skeleton="subject")
    m = manifest.assign(extract_dir=manifest["extract_dir"].where(manifest["subject"] != row["subject"],
                                                                  "extract/odd"))
    with pytest.raises(CliError, match="different skeleton: .*fmrib58.*subject"):
        run(base, m)


def test_warnings(cohort_dir):
    base, manifest = cohort_dir
    row = TEST.iloc[0]
    write_extract(base / "extract" / "late", "someone_else", row, delta=19.0)
    m = manifest.copy()
    m.loc[m["subject"] == row["subject"], ["extract_dir", "age"]] = ["extract/late", 90.0]
    m = pd.concat([m, m[m["role"] == "adapt"].iloc[:2].assign(subject=["x1", "x2"], site="OTHER")])
    w = "\n".join(run(base, m, warn_adapt_below=20).warnings)
    assert f"{row['subject']}: extract_dir" in w and "holds subject 'someone_else'" in w
    assert "fewer than 20 adaptation subjects at: NEW_SITE (12)" in w
    assert "outside the adaptation ages" in w and row["subject"] in w
    assert "acquired differently" in w and "Delta 19.0 ms" in w
    assert "adapt rows from site(s) without test subjects: OTHER" in w


def test_excluded_subjects(cohort_dir):
    base, manifest = cohort_dir
    row = TEST.iloc[0]
    m = pd.concat([manifest.assign(role=manifest["role"].where(manifest["subject"] != row["subject"], "exclude")),
                   pd.DataFrame([{"subject": "gone", "extract_dir": np.nan, "age": 40, "sex": 1, "site": "NEW_SITE",
                                  "role": "exclude"}])])
    out = run(base, m)
    assert row["subject"] not in set(out.test["subject"]) and len(out.test) == 2
    assert row["subject"] in set(out.qa_summary["subject"]) and "gone" not in set(out.qa_summary["subject"])


def test_qa_flags(cohort_dir):
    base, manifest = cohort_dir
    bad, odd = TEST.iloc[0], TEST.iloc[1]
    write_extract(base / "extract" / "bad", bad["subject"], bad, registration__r_wm=0.5,
                  flags={"label_coverage": "1 label(s) ..."}, warnings=["acquisition: shell b = 700"])
    typical = odd.copy()
    typical[TRACTS] = ADAPT[TRACTS].median()
    write_extract(base / "extract" / "typical", odd["subject"], typical)
    m = manifest.copy()
    m.loc[m["subject"] == odd["subject"], "extract_dir"] = "extract/typical"
    assert not (run(base, m).qa_flags["flag"] == "tract_outliers").any()
    shifted = typical.copy()
    shifted[TRACTS[:6]] = 0.95  # 6 of 48 tracts far outside the controls: both rules
    write_extract(base / "extract" / "odd", odd["subject"], shifted)
    m = manifest.copy()
    m.loc[m["subject"] == bad["subject"], "extract_dir"] = "extract/bad"
    m.loc[m["subject"] == odd["subject"], "extract_dir"] = "extract/odd"
    flags = run(base, m).qa_flags
    mine = flags[flags["subject"] == bad["subject"]].set_index("flag")
    assert mine.loc["registration.r_wm", "source"] == "cohort" and "(t -" in mine.loc["registration.r_wm", "detail"]
    assert mine.loc["label_coverage", "source"] == "extract"
    assert mine.loc["warning", "detail"] == "acquisition: shell b = 700"  # the QA warning isn't listed twice
    outliers = flags[(flags["subject"] == odd["subject"]) & (flags["flag"] == "tract_outliers")]
    assert len(outliers) == 2 and outliers["detail"].str.startswith("6 of 48 tracts").all()
    assert flags.loc[flags["role"] == "adapt", "source"].eq("cohort").all()


def test_no_cohort_flags_with_few_controls(cohort_dir):
    base, manifest = cohort_dir
    m = pd.concat([manifest[manifest["role"] == "test"], manifest[manifest["role"] == "adapt"].iloc[:4]])
    assert run(base, m).qa_flags.empty


def test_prediction_t_and_flatten():
    ref = np.array([[1.0, 5], [2, 5], [3, 5], [np.nan, 5]])
    t, n, mean, sd = collate.prediction_t([4.0, 6.0], ref)
    assert n.tolist() == [3, 4] and t[0] == pytest.approx((4 - 2) / (1 * np.sqrt(1 + 1 / 3))) and np.isnan(t[1])
    assert collate.metric_outlier(2.5, ref[:3, 0], "both") is None
    # n = 3: t(2) has heavy tails, so it takes a lot
    assert collate.metric_outlier(20.0, ref[:3, 0], "high") is None
    assert "t +" in collate.metric_outlier(50.0, ref[:3, 0], "high") and collate.metric_outlier(50.0, ref[:3, 0],
                                                                                                 "low") is None
    constant = np.full(11, 0.06) + np.array([0, 1e-17] * 5 + [0])  # rounding noise isn't variation
    assert np.isnan(collate.prediction_t(0.07, constant)[0])
    assert collate.metric_outlier(0.0, np.zeros(11), "high") is None
    assert collate.metric_outlier(2.5, np.zeros(11), "high") == "2.5; every adaptation subject has 0 (n 11)"
    assert collate.metric_outlier(-1.0, np.zeros(11), "high") is None
    assert collate.flatten({"a": {"b": 1, "c": "text", "d": [1.5, 2]}, "e": None, "f": True}) == {
        "a.b": 1, "a.d.0": 1.5, "a.d.1": 2, "e": None, "f": True}


def test_flag_rates_for_typical_subjects():
    """New subjects from the controls' distribution are rarely flagged, although with 12 controls they fall outside
    the controls' range on ~15% of tracts (which is what ENIGMA's percentiles amount to here)."""
    rng = np.random.default_rng(0)
    ref = rng.normal(0.5, 0.02, (12, 48))
    new = rng.normal(0.5, 0.02, (500, 48))
    assert ((new < ref.min(axis=0)) | (new > ref.max(axis=0))).mean() > 0.1
    assert sum(bool(collate.tract_outliers(v, ref)) for v in new) <= 5
    metric = rng.normal(1.0, 0.2, 12)
    assert sum(collate.metric_outlier(x, metric, "both") is not None for x in rng.normal(1.0, 0.2, 2000)) < 20


def test_tract_outliers():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.5, 0.02, (12, 48))
    values = ref.mean(axis=0)
    assert collate.tract_outliers(values, ref) == []
    values[:3] = 0.65  # 3 of 48 far out: the >= 5% rule only
    out = collate.tract_outliers(values, ref)
    assert len(out) == 1 and out[0][1].startswith("3 of 48 tracts outside the adaptation subjects' 0.1-99.9")
    values[3:5] = 0.35  # 5 of 48: both
    assert len(collate.tract_outliers(values, ref)) == 2
    values[:] = np.nan
    assert collate.tract_outliers(values, ref) == []
