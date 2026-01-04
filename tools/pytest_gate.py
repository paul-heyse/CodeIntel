"""Run targeted and segmented pytest suites for regression gating."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LOG = logging.getLogger(__name__)

DEFAULT_TARGETED = (
    "tests/ingestion/test_symtable_extract.py",
    "tests/ingestion/test_dis_extract_cfg.py",
    "tests/ingestion/test_dis_extract_defuse.py",
    "tests/ingestion/test_ast_span_joins.py",
    "tests/ingestion/test_inspect_overlay.py",
)

DEFAULT_SEGMENTS = (
    "tests/ingestion",
    "tests/build",
    "tests/graphs",
    "tests/storage",
    "tests/serving",
    "tests/runtime",
    "tests/analytics",
)


@dataclass(frozen=True, slots=True)
class PytestRun:
    """Describe a pytest run segment."""

    label: str
    paths: tuple[str, ...]


def _normalize_paths(paths: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in paths:
        if not raw:
            continue
        normalized.append(str(Path(raw)))
    return tuple(normalized)


def _validate_paths(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        message = f"Missing pytest paths: {missing}"
        raise FileNotFoundError(message)


def _run_pytest(run: PytestRun, *, pytest_args: Sequence[str]) -> int:
    cmd = ["uv", "run", "pytest", "-q", *pytest_args, *run.paths]
    LOG.info("pytest_gate.start label=%s cmd=%s", run.label, shlex.join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOG.error("pytest_gate.failed label=%s code=%s", run.label, result.returncode)
    else:
        LOG.info("pytest_gate.ok label=%s", run.label)
    return result.returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted + segmented pytest suites.")
    parser.add_argument(
        "--targeted",
        nargs="*",
        default=None,
        help="Explicit list of targeted test files to run.",
    )
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Explicit list of pytest directories to run as segments.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=(),
        help="Extra pytest arguments to pass through.",
    )
    parser.add_argument(
        "--skip-targeted",
        action="store_true",
        help="Skip the targeted test subset.",
    )
    parser.add_argument(
        "--skip-segments",
        action="store_true",
        help="Skip segmented pytest runs.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running segments after a failure.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    targeted = (
        DEFAULT_TARGETED if args.targeted is None or not args.targeted else tuple(args.targeted)
    )
    segments = DEFAULT_SEGMENTS if args.segments is None or not args.segments else tuple(
        args.segments
    )
    pytest_args = tuple(args.pytest_args)

    runs: list[PytestRun] = []
    if not args.skip_targeted and targeted:
        runs.append(PytestRun(label="targeted", paths=_normalize_paths(targeted)))
    if not args.skip_segments and segments:
        for segment in segments:
            runs.append(PytestRun(label=f"segment:{segment}", paths=_normalize_paths([segment])))

    if not runs:
        LOG.info("pytest_gate.skip reason=no_runs_configured")
        return 0

    for run in runs:
        _validate_paths(run.paths)
        return_code = _run_pytest(run, pytest_args=pytest_args)
        if return_code != 0 and not args.continue_on_failure:
            return return_code

    return 0


if __name__ == "__main__":
    sys.exit(main())
