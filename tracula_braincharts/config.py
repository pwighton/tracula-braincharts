"""Constants shared across the pipeline."""

import os
from pathlib import Path

# Upstream braincharts repository and the commit the model files are fetched from.
UPSTREAM_REPO = "https://github.com/predictive-clinical-neuroscience/braincharts"
UPSTREAM_COMMIT = "c017a3e0beca4a62982ebf6563a19fa5c0b7284c"

# Where the Dockerfile puts the braincharts files; overridable for local runs.
MODEL_DIR = Path(os.environ.get("TRACULA_BRAINCHARTS_MODEL_DIR", "/opt/tracula-braincharts/braincharts"))

# Paths inside MODEL_DIR (same layout as the braincharts repo; the zip unpacks to FA_50K_19sites/).
TRACTS_FILE = "docs/phenotypes_fa.txt"
SITES_FILE = "docs/site_ids_fa_19sites.txt"
MODELS_SUBDIR = "models/FA_50K_19sites"

# Header of the portal's csv template (dMRI_BLR_fa_lifespan_24K_19sites_template.csv), one name per line.
TEMPLATE_COLUMNS_FILE = Path(__file__).parent / "data" / "template-columns.txt"

# Template columns before the 48 tract columns. Only age, sex (F=0, M=1) and site are used by the model.
COVARIATE_COLUMNS = ("sex", "age", "site", "race")

# Age range of the cubic B-spline basis used when the models were trained (02.normative_models_fa.ipynb).
AGE_XMIN = -5
AGE_XMAX = 110

# Design matrix width: intercept + age + sex + 19 site columns + 7 B-spline columns.
DESIGN_WIDTH = 29

# extract --skeleton choice -> tbss_3_postreg option: the FMRIB58 standard skeleton (as UKB and, by its text, the
# FA paper) or the subject's own mean-FA skeleton (as the training notebook's per-subject cell)
SKELETONS = {"fmrib58": "-T", "subject": "-S"}

# collate warns when an adapted site has fewer adaptation subjects than this (fewer than 2 is an error)
WARN_ADAPT_BELOW = 10
