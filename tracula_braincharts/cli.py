"""Command-line interface.

    tracula-braincharts extract ...   # per subject: FA features, provenance and QA metrics
    tracula-braincharts collate ...   # adapt.csv / test.csv and cohort QA tables from a manifest
    tracula-braincharts score ...     # z-scores and centiles for test.csv, adapted with adapt.csv
    tracula-braincharts x-val ...     # cross-validated z-scores and adaptation QA for adapt.csv

Subcommands are stubs until their implementation step lands.
"""

import argparse
import sys

from . import __version__

# Exit codes: 0 success, 1 unexpected error, 2 usage/input error.
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2

# Subcommand -> (help text, implementation step in the plan)
SUBCOMMANDS = {
    "extract": ("per subject: FA features, provenance and QA metrics", 5),
    "collate": ("build adapt.csv / test.csv and cohort QA tables from a manifest", 6),
    "score": ("z-scores and centiles for test.csv, adapted with adapt.csv", 2),
    "x-val": ("cross-validated z-scores and adaptation QA for adapt.csv", 4),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracula-braincharts",
        description="Apply the BLR_fa_lifespan_24K_19sites FA normative model to Tracula subjects.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{" + ",".join(SUBCOMMANDS) + "}", required=True)
    for name, (help_text, _step) in SUBCOMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _help_text, step = SUBCOMMANDS[args.command]
    print(f"tracula-braincharts: '{args.command}' is not implemented yet (plan step {step})", file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
