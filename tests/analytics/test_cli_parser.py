"""CLI parser flag tests for function parser selection."""

from __future__ import annotations

import argparse

import pytest

from codeintel.cli.main import make_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = make_parser()
    return parser.parse_args(argv)


def test_function_parser_invalid_exits() -> None:
    """Invalid function parser flag should exit with error."""
    with pytest.raises(SystemExit):
        _parse(
            [
                "pipeline",
                "run",
                "--repo",
                "r",
                "--commit",
                "c",
                "--function-parser",
                "bogus",
            ]
        )


def test_function_parser_default_none() -> None:
    """When not provided, function parser flag remains None."""
    args = _parse(["pipeline", "run", "--repo", "r", "--commit", "c"])
    if args.function_parser is not None:
        pytest.fail("function_parser should default to None when flag is omitted")
