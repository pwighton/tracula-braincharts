import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from tracula_braincharts.cli_errors import CliError
from tracula_braincharts.cohort import ID_COLUMNS, TRACTS, read_cohort

FIXTURE = FIXTURES / "test-training-sites.csv"


def _write(tmp_path, df):
    path = tmp_path / "cohort.csv"
    df.to_csv(path, index=False)
    return path


def test_reads_fixture():
    df = read_cohort(FIXTURE)
    assert list(df.columns) == ID_COLUMNS + TRACTS
    assert list(df["subject"]) == ["sub-ukb60f", "sub-hcpya28m", "sub-hcpa75f"]
    assert df["sex"].tolist() == [0, 1, 0]
    assert np.isnan(df.loc[2, "FA_mean_of_Fornix-column_and_body"])
    assert df[TRACTS].notna().sum().sum() == 3 * 48 - 1


def test_subject_column_is_optional(tmp_path):
    raw = pd.read_csv(FIXTURE).drop(columns="subject")
    df = read_cohort(_write(tmp_path, raw))
    assert list(df["subject"]) == ["row1", "row2", "row3"]


def test_extra_columns_are_ignored(tmp_path):
    raw = pd.read_csv(FIXTURE).assign(scanner="Connectom", race="White")
    assert list(read_cohort(_write(tmp_path, raw)).columns) == ID_COLUMNS + TRACTS


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns="FA_mean_of_Tapetum_L"), "missing columns"),
    (lambda d: d.drop(columns="age"), "missing columns"),
    (lambda d: d.assign(subject=["a", "a", "b"]), "duplicate subject"),
    (lambda d: d.assign(sex=[0, 2, 1]), "sex must be"),
    (lambda d: d.assign(sex=["F", "M", "F"]), "sex must be"),
    (lambda d: d.assign(age=[30, 120, 40]), "age must be"),
    (lambda d: d.assign(age=[30, None, 40]), "age must be"),
    (lambda d: d.assign(site=["A", None, "B"]), "empty site"),
    (lambda d: d.assign(FA_mean_of_Tapetum_L=[0.5, 1.2, 0.5]), r"outside \(0, 1\)"),
    (lambda d: d.assign(FA_mean_of_Tapetum_L=[0.5, 0.0, 0.5]), r"outside \(0, 1\)"),
    (lambda d: d.assign(FA_mean_of_Tapetum_L=[0.5, "x", 0.5]), "non-numeric"),
    (lambda d: d.iloc[0:0], "no rows"),
])
def test_rejects_bad_input(tmp_path, change, message):
    with pytest.raises(CliError, match=message):
        read_cohort(_write(tmp_path, change(pd.read_csv(FIXTURE))))


def test_missing_file(tmp_path):
    with pytest.raises(CliError, match="cannot read"):
        read_cohort(tmp_path / "nope.csv")
