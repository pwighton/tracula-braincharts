import pytest

from tracula_braincharts import __version__
from tracula_braincharts.cli import EXIT_USAGE, SUBCOMMANDS, build_parser, main


def test_subcommands_are_registered():
    assert set(SUBCOMMANDS) == {"extract", "collate", "score", "x-val"}
    for name in SUBCOMMANDS:
        assert build_parser().parse_args([name]).command == name


def test_subcommand_is_required():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args([])
    assert exc.value.code == 2


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(SUBCOMMANDS))
def test_stubs_report_not_implemented(name, capsys):
    assert main([name]) == EXIT_USAGE
    assert "not implemented yet" in capsys.readouterr().err
