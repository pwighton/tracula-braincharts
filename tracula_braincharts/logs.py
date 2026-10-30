"""Logging for a subcommand run: timestamped messages on stderr and in a log file in the output directory."""

import logging
from pathlib import Path

from .cli_errors import CliError

FORMAT = "%(asctime)s %(levelname)s %(message)s"

# Pass as ``extra=`` to write a record to the log file only (the CLI prints its own one-line error message).
FILE_ONLY = {"file_only": True}

_handlers: list[logging.Handler] = []


def start(out_dir: Path, filename: str) -> Path:
    """Create ``out_dir`` and log to stderr and ``out_dir/filename``. Returns the log file's path.

    The log file is overwritten, so it describes only the latest run. Messages from this package are logged
    from INFO up; other libraries' from WARNING up. Python warnings are logged too.
    """
    log_file = out_dir / filename
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
    except OSError as exc:
        raise CliError(f"cannot write to output directory {out_dir}: {exc}") from exc
    stop()
    console = logging.StreamHandler()
    console.addFilter(lambda record: not getattr(record, "file_only", False))
    _handlers.extend([console, file_handler])
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in _handlers:
        handler.setFormatter(logging.Formatter(FORMAT))
        root.addHandler(handler)
    logging.getLogger("tracula_braincharts").setLevel(logging.INFO)
    logging.captureWarnings(True)
    return log_file


def stop() -> None:
    """Remove and close the handlers added by ``start``."""
    root = logging.getLogger()
    for handler in _handlers:
        root.removeHandler(handler)
        handler.close()
    _handlers.clear()
    logging.captureWarnings(False)
