"""Reading and validating cohort csv files.

A cohort csv uses the portal's template format (``sex,age,site,race`` + 48 tract columns) with an optional
leading ``subject`` column. ``race`` and any other extra columns are ignored. Tract values may be empty
(missing); present values must be FA values in (0, 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .cli_errors import CliError
from .config import AGE_XMAX, AGE_XMIN
from .models import template_columns

TRACTS = template_columns()[4:]
ID_COLUMNS = ["subject", "sex", "age", "site"]


def _list(values, limit=5) -> str:
    values = [str(v) for v in values]
    return ", ".join(values[:limit]) + (f" (+{len(values) - limit} more)" if len(values) > limit else "")


def read_cohort(path: Path) -> pd.DataFrame:
    """Read and validate a cohort csv; returns ``subject, sex, age, site`` + the 48 tracts, in template order."""
    try:
        df = pd.read_csv(path, dtype={"subject": str, "site": str})
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    return check_cohort(df, str(path))


def check_cohort(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Validate a cohort table (``path`` names it in errors); returns ``subject, sex, age, site`` + the 48 tracts."""
    missing = [c for c in ["sex", "age", "site", *TRACTS] if c not in df.columns]
    if missing:
        raise CliError(f"{path}: missing columns: {_list(missing)}")
    if df.empty:
        raise CliError(f"{path}: no rows")

    if "subject" not in df.columns:
        df.insert(0, "subject", [f"row{i}" for i in range(1, len(df) + 1)])
    if df["subject"].isna().any():
        raise CliError(f"{path}: empty subject id in row(s) {_list(np.flatnonzero(df['subject'].isna()) + 1)}")
    dup = df.loc[df["subject"].duplicated(), "subject"]
    if not dup.empty:
        raise CliError(f"{path}: duplicate subject ids: {_list(dup.unique())}")

    sex = pd.to_numeric(df["sex"], errors="coerce")
    bad = df.loc[~sex.isin([0, 1]), "subject"]
    if not bad.empty:
        raise CliError(f"{path}: sex must be 0 (female) or 1 (male); bad rows: {_list(bad)}")

    age = pd.to_numeric(df["age"], errors="coerce")
    bad = df.loc[~age.between(AGE_XMIN, AGE_XMAX), "subject"]
    if not bad.empty:
        raise CliError(f"{path}: age must be a number in [{AGE_XMIN}, {AGE_XMAX}] years; bad rows: {_list(bad)}")

    bad = df.loc[df["site"].isna() | (df["site"].astype(str).str.strip() == ""), "subject"]
    if not bad.empty:
        raise CliError(f"{path}: empty site in rows: {_list(bad)}")

    fa = df[TRACTS].apply(pd.to_numeric, errors="coerce")
    not_numeric = fa.isna() & df[TRACTS].notna()
    out_of_range = fa.notna() & ~((fa > 0) & (fa < 1))
    for problem, mask in (("non-numeric", not_numeric), ("outside (0, 1)", out_of_range)):
        if mask.any().any():
            cells = [f"{df['subject'].iloc[r]}/{TRACTS[c]}" for r, c in zip(*np.nonzero(mask.to_numpy()))]
            raise CliError(f"{path}: {problem} FA values: {_list(cells)}")

    out = pd.DataFrame({"subject": df["subject"].astype(str), "sex": sex.astype(int), "age": age.astype(float),
                        "site": df["site"].astype(str).str.strip()})
    return pd.concat([out, fa.astype(float)], axis=1).reset_index(drop=True)
