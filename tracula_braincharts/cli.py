"""Command-line interface.

    tracula-braincharts extract ...   # per subject: FA features, provenance and QA metrics
    tracula-braincharts collate ...   # adapt.csv / test.csv and cohort QA tables from a manifest
    tracula-braincharts score ...     # z-scores and centiles for test.csv, adapted with adapt.csv
    tracula-braincharts x-val ...     # cross-validated z-scores and adaptation QA for adapt.csv

Subcommands that are still stubs say so and exit 2.
"""

import argparse
import logging
import sys
import traceback
import warnings
from pathlib import Path

from . import __version__, logs
from .cli_errors import CliError
from .config import MODEL_DIR, SKELETONS

log = logging.getLogger("tracula_braincharts")

# Exit codes: 0 success, 1 unexpected error, 2 usage/input error.
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2

# Subcommand -> (help text, implementation step in the plan; None once implemented)
SUBCOMMANDS = {
    "extract": ("per subject: TBSS and mean skeleton FA per JHU label, with provenance and QA", None),
    "collate": ("build adapt.csv / test.csv and cohort QA tables from a manifest", 6),
    "score": ("z-scores and centiles for test.csv, adapted with adapt.csv", None),
    "x-val": ("cross-validated z-scores and adaptation QA for adapt.csv", None),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracula-braincharts",
        description="Apply the BLR_fa_lifespan_24K_19sites FA normative model to Tracula subjects.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(SUBCOMMANDS) + "}", required=True)
    parsers = {name: sub.add_parser(name, help=help_text) for name, (help_text, _step) in SUBCOMMANDS.items()}

    score = parsers["score"]
    score.add_argument("--test", required=True, type=Path, metavar="TEST.csv",
                       help="subjects to score: portal template columns, optional leading 'subject' column")
    score.add_argument("--adapt", type=Path, metavar="ADAPT.csv",
                       help="healthy controls for site adaptation, needed when a test site is not a training site; "
                            "then every test site is adapted (as on the portal) and needs at least 2 controls")
    score.add_argument("-o", "--out", required=True, type=Path, metavar="DIR",
                       help="output directory (created if needed; existing outputs, including score.log, "
                            "are overwritten)")
    score.add_argument("--model-dir", type=Path, default=MODEL_DIR, metavar="DIR",
                       help=f"braincharts model files (default: {MODEL_DIR})")

    extract = parsers["extract"]
    extract.add_argument("subject_dir", type=Path, metavar="SUBJECT_DIR", help="Tracula subject directory")
    extract.add_argument("-o", "--out", type=Path, metavar="DIR",
                         help="write to DIR/<subject>/ (default: SUBJECT_DIR/fa-normative/); existing outputs, "
                              "including tbss/ and extract.log, are overwritten")
    extract.add_argument("--fa", type=Path, metavar="FA.nii.gz",
                         help="FA map (with --bval; default: found from scripts/log.txt or dti/)")
    extract.add_argument("--bval", type=Path, metavar="BVALS", help="b-values the FA map was fitted with")
    extract.add_argument("--skeleton", choices=list(SKELETONS), default="fmrib58",
                         help="skeleton for tbss_3_postreg: fmrib58 (-T, default) or subject (-S, the subject's own "
                              "mean FA); controls and test subjects must use the same")
    extract.add_argument("--allow-shells", action="store_true",
                         help="extract even if the FA map isn't from b=0 + one shell in 800-1600")

    xval = parsers["x-val"]
    xval.add_argument("--adapt", required=True, type=Path, metavar="ADAPT.csv",
                      help="healthy controls (portal template columns, optional leading 'subject' column); every "
                           "site must be new (not a training site)")
    xval.add_argument("-o", "--out", required=True, type=Path, metavar="DIR",
                      help="output directory (created if needed; existing outputs, including x-val.log, "
                           "are overwritten)")
    xval.add_argument("--folds", type=int, metavar="K",
                      help="K-fold cross-validation within each site (default: leave-one-out)")
    xval.add_argument("--seed", type=int, default=0, help="seed for assigning K-fold folds (default: 0)")
    xval.add_argument("--model-dir", type=Path, default=MODEL_DIR, metavar="DIR",
                      help=f"braincharts model files (default: {MODEL_DIR})")
    return parser


def _score(args) -> None:
    from . import scoring

    logs.start(args.out, "score.log")
    scoring.run(args.test, args.out, adapt_csv=args.adapt, model_dir=args.model_dir)


def _extract(args) -> None:
    from . import extract, inputs

    inp = inputs.resolve(args.subject_dir, fa=args.fa, bval=args.bval)
    out_dir = extract.output_dir(inp, args.out)
    logs.start(out_dir, "extract.log")
    extract.run_extract(inp, out_dir, skeleton=args.skeleton, allow_shells=args.allow_shells)


def _x_val(args) -> None:
    from . import xval

    logs.start(args.out, "x-val.log")
    xval.run(args.adapt, args.out, n_folds=args.folds, seed=args.seed, model_dir=args.model_dir)


HANDLERS = {"extract": _extract, "score": _score, "x-val": _x_val}


def main(argv=None) -> int:
    # arviz (imported via pymc) announces its 1.0 refactor once a day, i.e. on every run in a fresh container
    warnings.filterwarnings("ignore", message=r"\s*ArviZ is undergoing", category=FutureWarning)
    args = build_parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        _help_text, step = SUBCOMMANDS[args.command]
        print(f"tracula-braincharts: '{args.command}' is not implemented yet (plan step {step})", file=sys.stderr)
        return EXIT_USAGE
    try:
        handler(args)
    except CliError as exc:
        log.error("%s", exc, extra=logs.FILE_ONLY)
        print(f"tracula-braincharts: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        log.exception("unexpected error", extra=logs.FILE_ONLY)
        traceback.print_exc()
        return EXIT_UNEXPECTED
    finally:
        logs.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
