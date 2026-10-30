"""Finding a Tracula subject's FA map and the b-values it was fitted with, and checking the acquisition.

The lab's Tracula wrapper writes ``dti/<prefix>_dti_FA.nii.gz`` from ``dtifit``, and logs every command in
``scripts/log.txt`` with the absolute paths of the machine it ran on. File names are discovered, not assumed:
1. explicit ``--fa`` and ``--bval``;
2. otherwise the last ``dtifit`` in ``scripts/log.txt`` whose output is ``.../dti/<prefix>_dti``: its ``-k``, ``-m``
   and ``-b`` files are looked up by their last two path components (``dti/...``, ``bmask-eddy/...``) in the
   subject directory;
3. otherwise exactly one ``dti/*_dti_FA.nii.gz`` and exactly one ``dti/*.bval``.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cli_errors import CliError

# b-values at or below this count as b=0 (some scanners store b=0 as 5 or 10)
B0_MAX = 20
# b-values within this distance belong to the same shell
SHELL_TOLERANCE = 50
# the training data used each dataset's lowest shell, b ~ 1000-1500
SHELL_RANGE = (800, 1600)

DTIFIT_OPTIONS = {"-k": "data", "--data": "data", "-o": "out", "--out": "out", "-m": "mask", "--mask": "mask",
                  "-b": "bvals", "--bvals": "bvals"}


@dataclass
class Inputs:
    subject: str
    subject_dir: Path
    fa: Path
    bval: Path
    dwi: Path | None  # the dtifit -k image, used to check the number of volumes
    mask: Path | None  # the dtifit -m brain mask
    method: str  # "explicit", "log" or "glob"
    log_line: int | None = None


def _local(subject_dir: Path, logged: str) -> Path:
    """A path from the log, found by its last two components (e.g. ``dti/x.bval``) under the subject directory."""
    parts = Path(logged).parts
    return subject_dir.joinpath(*parts[-2:])


def _image(path: Path) -> Path | None:
    """``path`` as given, or with ``.nii.gz`` / ``.nii`` appended (dtifit takes image names without extension)."""
    for candidate in (path, path.with_name(path.name + ".nii.gz"), path.with_name(path.name + ".nii")):
        if candidate.is_file():
            return candidate
    return None


def parse_dtifit(line: str) -> dict[str, str] | None:
    """The data/out/mask/bvals arguments of a logged ``dtifit`` command, or None if the line isn't one."""
    try:
        words = shlex.split(line)
    except ValueError:
        return None
    if not words or Path(words[0]).name != "dtifit":
        return None
    args = {}
    for i, word in enumerate(words[1:], start=1):
        key, _, value = word.partition("=")
        if key in DTIFIT_OPTIONS:
            if not value and i + 1 < len(words):
                value = words[i + 1]
            args[DTIFIT_OPTIONS[key]] = value
    return args


def _from_log(subject_dir: Path) -> Inputs | None:
    log_txt = subject_dir / "scripts" / "log.txt"
    if not log_txt.is_file():
        return None
    found = None
    for number, line in enumerate(log_txt.read_text(errors="replace").splitlines(), start=1):
        args = parse_dtifit(line)
        if args and "out" in args and "bvals" in args:
            out = Path(args["out"])
            if out.parent.name == "dti" and out.name.endswith("_dti"):
                found = (number, args)
    if found is None:
        return None
    number, args = found
    out = _local(subject_dir, args["out"])
    fa = out.with_name(out.name + "_FA.nii.gz")
    bval = _local(subject_dir, args["bvals"])
    for label, path in (("FA map", fa), ("b-values", bval)):
        if not path.is_file():
            raise CliError(f"{log_txt} line {number}: dtifit's {label} not found at {path} "
                           f"(pass --fa and --bval to choose the files)")
    dwi = _image(_local(subject_dir, args["data"])) if "data" in args else None
    mask = _image(_local(subject_dir, args["mask"])) if "mask" in args else None
    return Inputs(subject_dir.name, subject_dir, fa, bval, dwi, mask, "log", number)


def _from_glob(subject_dir: Path) -> Inputs:
    dti = subject_dir / "dti"
    fas, bvals = sorted(dti.glob("*_dti_FA.nii.gz")), sorted(dti.glob("*.bval"))
    if len(fas) != 1 or len(bvals) != 1:
        raise CliError(f"cannot tell which FA map and b-values to use in {dti} ({len(fas)} *_dti_FA.nii.gz, "
                       f"{len(bvals)} *.bval, and no dtifit in scripts/log.txt); pass --fa and --bval")
    return Inputs(subject_dir.name, subject_dir, fas[0], bvals[0], None, None, "glob")


def resolve(subject_dir: Path, fa: Path | None = None, bval: Path | None = None) -> Inputs:
    """Find the FA map and its b-values (see the module docstring)."""
    subject_dir = Path(subject_dir)
    if not subject_dir.is_dir():
        raise CliError(f"subject directory not found: {subject_dir}")
    if (fa is None) != (bval is None):
        raise CliError("pass both --fa and --bval, or neither")
    if fa is not None:
        for path in (fa, bval):
            if not Path(path).is_file():
                raise CliError(f"file not found: {path}")
        return Inputs(subject_dir.name, subject_dir, Path(fa), Path(bval), None, None, "explicit")
    return _from_log(subject_dir) or _from_glob(subject_dir)


@dataclass
class Shells:
    n_b0: int
    shells: list[dict]  # [{"b": mean b, "n": volumes}], ascending


def read_bvals(path: Path) -> np.ndarray:
    try:
        return np.atleast_1d(np.loadtxt(path, dtype=float)).ravel()
    except ValueError as exc:
        raise CliError(f"cannot read b-values from {path}: {exc}") from exc


def shells(bvals: np.ndarray) -> Shells:
    """Group b-values into b=0 and shells (sorted values within ``SHELL_TOLERANCE`` of the shell's first value)."""
    nonzero = np.sort(bvals[bvals > B0_MAX])
    groups: list[list[float]] = []
    for b in nonzero:
        if groups and b - groups[-1][0] <= SHELL_TOLERANCE:
            groups[-1].append(b)
        else:
            groups.append([b])
    return Shells(int((bvals <= B0_MAX).sum()), [{"b": float(np.mean(g)), "n": len(g)} for g in groups])


def check_shells(s: Shells, allow: bool = False) -> list[str]:
    """Problems with the FA's acquisition: it should come from b=0 plus one shell in ``SHELL_RANGE``, like the
    training data. Raises unless ``allow``; returns the problems as warnings when allowed."""
    problems = []
    if s.n_b0 == 0:
        problems.append("no b=0 volumes")
    if len(s.shells) != 1:
        found = ", ".join(f"{x['b']:.0f}" for x in s.shells) or "none"
        problems.append(f"{len(s.shells)} non-zero shells (b = {found}), expected 1")
    elif not SHELL_RANGE[0] <= s.shells[0]["b"] <= SHELL_RANGE[1]:
        problems.append(f"shell b = {s.shells[0]['b']:.0f} is outside {SHELL_RANGE[0]}-{SHELL_RANGE[1]}, the range "
                        "of the training data's lowest shells")
    if problems and not allow:
        raise CliError("the FA map's b-values don't match the training data: " + "; ".join(problems) +
                       " (pass --allow-shells to extract anyway)")
    return problems


def diffusion_time(subject_dir: Path, b: float) -> dict:
    """Diffusion time(s) Delta of the volumes acquired at b-value ``b``, from the wrapper's ``diff/*_diff.bval`` and
    ``diff/*_diff.delta`` (one value per volume). ``{"delta": ..., "source": ...}``, with delta None and a reason when
    the files are missing or volumes at this b-value have more than one Delta."""
    diff = subject_dir / "diff"
    bvals, deltas = sorted(diff.glob("*_diff.bval")), sorted(diff.glob("*_diff.delta"))
    if len(bvals) != 1 or len(deltas) != 1:
        return {"delta_ms": None, "reason": f"need one diff/*_diff.bval and one diff/*_diff.delta, found "
                                            f"{len(bvals)} and {len(deltas)}"}
    b_all, d_all = read_bvals(bvals[0]), read_bvals(deltas[0])
    if len(b_all) != len(d_all):
        return {"delta_ms": None, "reason": f"{bvals[0].name} has {len(b_all)} values, {deltas[0].name} {len(d_all)}"}
    found = sorted({float(x) for x in d_all[np.abs(b_all - b) <= SHELL_TOLERANCE].round(3)})
    source = f"{bvals[0].relative_to(subject_dir)}, {deltas[0].relative_to(subject_dir)}"
    if len(found) == 1:
        return {"delta_ms": found[0], "source": source}
    return {"delta_ms": None, "source": source,
            "reason": f"volumes at b = {b:.0f} have Delta {found}" if found else f"no volumes at b = {b:.0f}"}
