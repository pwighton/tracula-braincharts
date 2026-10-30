"""Provenance records written next to every output."""

import hashlib
import json
import os
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path

from . import __version__
from .config import MODEL_DIR, UPSTREAM_COMMIT, UPSTREAM_REPO


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _version(dist: str):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def base_record(command: str, model_dir: Path = MODEL_DIR) -> dict:
    return {
        "tool": "tracula-braincharts",
        "version": __version__,
        "git_sha": os.environ.get("TRACULA_BRAINCHARTS_GIT_SHA"),
        "image": os.environ.get("TRACULA_BRAINCHARTS_IMAGE"),
        "command": command,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {"upstream_repo": UPSTREAM_REPO, "upstream_commit": UPSTREAM_COMMIT, "model_dir": str(model_dir)},
        "software": {"python": platform.python_version(),
                     **{name: _version(name) for name in ("pcntoolkit", "numpy", "pandas", "scipy")}},
    }


def write(record: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(record, indent=2) + "\n")
