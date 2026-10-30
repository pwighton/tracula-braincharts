import importlib.util
from pathlib import Path

import pytest

from tracula_braincharts.config import MODEL_DIR, MODELS_SUBDIR

FIXTURES = Path(__file__).parent / "fixtures"


def _pcntoolkit_available() -> bool:
    return importlib.util.find_spec("pcntoolkit") is not None


requires_pcntoolkit = pytest.mark.skipif(not _pcntoolkit_available(),
                                         reason="needs pcntoolkit 0.35 (run inside the container)")

requires_models = pytest.mark.skipif(not (_pcntoolkit_available() and (MODEL_DIR / MODELS_SUBDIR).is_dir()),
                                     reason="needs pcntoolkit 0.35 and the model files (run inside the container)")


@pytest.fixture(scope="session")
def model_set():
    from tracula_braincharts.scoring import ModelSet
    return ModelSet()


def link_models(model_set, mdir: Path) -> Path:
    """A ``Models/`` dir as pcntoolkit's ``normative.predict``/``transfer`` want it: ``NM_0_<i>_estimate.pkl`` per
    tract, in model order. No ``meta_data.md``: they would index its training mean/sd by model number, but each
    tract's file covers only that tract. The models use no scalers, so they compute the same without it."""
    from tracula_braincharts import models

    mdir.mkdir()
    for i, tract in enumerate(model_set.tracts):
        (mdir / f"NM_0_{i}_estimate.pkl").symlink_to(models.model_file(tract, model_set.model_dir, "NM_0_0_estimate.pkl"))
    return mdir
