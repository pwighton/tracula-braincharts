import importlib.util

import pytest

from tracula_braincharts.config import MODEL_DIR, MODELS_SUBDIR


def _pcntoolkit_available() -> bool:
    return importlib.util.find_spec("pcntoolkit") is not None


requires_pcntoolkit = pytest.mark.skipif(not _pcntoolkit_available(),
                                         reason="needs pcntoolkit 0.35 (run inside the container)")

requires_models = pytest.mark.skipif(not (_pcntoolkit_available() and (MODEL_DIR / MODELS_SUBDIR).is_dir()),
                                     reason="needs pcntoolkit 0.35 and the model files (run inside the container)")
