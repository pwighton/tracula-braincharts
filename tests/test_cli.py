from pathlib import Path

import pytest

from tracula_braincharts import __version__
from tracula_braincharts.cli import EXIT_USAGE, SUBCOMMANDS, build_parser, main

STUBS = sorted(name for name, (_help, step) in SUBCOMMANDS.items() if step is not None)


def test_subcommands_are_registered():
    assert set(SUBCOMMANDS) == {"extract", "collate", "score", "x-val"}
    for name in STUBS:
        assert build_parser().parse_args([name]).command == name
    args = build_parser().parse_args(["score", "--test", "t.csv", "-o", "out"])
    assert args.command == "score" and args.adapt is None
    assert build_parser().parse_args(["x-val", "--adapt", "a.csv", "-o", "out"]).command == "x-val"


def test_subcommand_is_required():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_score_requires_test_and_out():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["score", "--test", "t.csv"])
    assert exc.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("name", STUBS)
def test_stubs_report_not_implemented(name, capsys):
    assert main([name]) == EXIT_USAGE
    assert "not implemented yet" in capsys.readouterr().err


def test_score_input_error_exits_2(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["score", "--test", str(tmp_path / "missing.csv"), "-o", str(out)]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "tracula-braincharts: error: cannot read" in err and err.count("cannot read") == 1
    assert " ERROR cannot read" in (out / "score.log").read_text()


def test_score_unwritable_out_dir_exits_2(tmp_path, capsys):
    (tmp_path / "file").touch()
    fixture = Path(__file__).parent / "fixtures" / "test-training-sites.csv"
    assert main(["score", "--test", str(fixture), "-o", str(tmp_path / "file" / "out")]) == EXIT_USAGE
    assert "cannot write to output directory" in capsys.readouterr().err


def test_score_without_models_exits_2(tmp_path, capsys):
    fixture = Path(__file__).parent / "fixtures" / "test-training-sites.csv"
    argv = ["score", "--test", str(fixture), "-o", str(tmp_path / "out"), "--model-dir", str(tmp_path / "none")]
    assert main(argv) == EXIT_USAGE
    assert "model files not found" in capsys.readouterr().err
