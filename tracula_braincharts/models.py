"""Access to the upstream FA normative models (PCNtoolkit 0.35 NormBLR pickles)."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (AGE_XMAX, AGE_XMIN, DESIGN_WIDTH, MODEL_DIR, MODELS_SUBDIR, SITES_FILE,
                     TEMPLATE_COLUMNS_FILE, TRACTS_FILE)


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def template_columns() -> list[str]:
    """All columns of the portal's csv template, in order."""
    return _read_lines(TEMPLATE_COLUMNS_FILE)


def tract_names(model_dir: Path = MODEL_DIR) -> list[str]:
    """The 48 tract (IDP) names, in model / template order."""
    return _read_lines(model_dir / TRACTS_FILE)


def training_sites(model_dir: Path = MODEL_DIR) -> list[str]:
    """The 19 training site ids, in the order of the design matrix's site columns.

    The models were fitted without ``all_sites``, so ``create_design_matrix`` ordered the site columns
    alphabetically (``sorted(pd.unique(site_ids))``). The published site list is in a different order, so it is
    sorted here. Using it as published assigns each site another site's effect.
    """
    return sorted(_read_lines(model_dir / SITES_FILE))


def model_file(tract: str, model_dir: Path, filename: str) -> Path:
    return model_dir / MODELS_SUBDIR / tract / "Models" / filename


def _alias_norm_blr() -> None:
    # The pickles reference the top-level module ``norm_blr`` (PCNtoolkit was run from its source tree when
    # they were saved); point that name at the packaged module so they can be unpickled.
    from pcntoolkit.normative_model import norm_blr
    sys.modules.setdefault("norm_blr", norm_blr)


def load_model(tract: str, model_dir: Path = MODEL_DIR):
    """Load one tract's ``NormBLR`` model."""
    _alias_norm_blr()
    with open(model_file(tract, model_dir, "NM_0_0_estimate.pkl"), "rb") as f:
        return pickle.load(f)


def load_meta(tract: str, model_dir: Path = MODEL_DIR) -> dict:
    """Load one tract's metadata (a pickle named ``meta_data.md``): training mean/sd and scaler settings."""
    with open(model_file(tract, model_dir, "meta_data.md"), "rb") as f:
        return pickle.load(f)


def design_matrix(age, sex, site, sites: list[str]) -> np.ndarray:
    """Design matrix as used for training and by the portal.

    Columns: intercept, age, sex, one-hot over ``sites`` (a site not in the list gets all zeros),
    cubic B-spline over age (5 knots on [AGE_XMIN, AGE_XMAX]).
    """
    from pcntoolkit.util.utils import create_design_matrix

    covariates = pd.DataFrame({"age": np.asarray(age, dtype=float), "sex": np.asarray(sex, dtype=float)})
    X = create_design_matrix(covariates, site_ids=pd.Series(list(site)), all_sites=sites,
                             basis="bspline", xmin=AGE_XMIN, xmax=AGE_XMAX)
    if X.shape[1] != DESIGN_WIDTH:
        raise ValueError(f"design matrix has {X.shape[1]} columns, the models expect {DESIGN_WIDTH}")
    return X
