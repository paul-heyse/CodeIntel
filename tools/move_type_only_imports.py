"""Automatically move type-only imports under ``if TYPE_CHECKING:`` blocks."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import libcst as cst
from libcst import metadata

if TYPE_CHECKING:
    from collections.abc import Iterable


def _simple_import_from(stmt: cst.BaseStatement) -> cst.ImportFrom | None:
    """
    Return the ImportFrom node if present in a simple statement.

    Returns
    -------
    cst.ImportFrom | None
        ImportFrom node when present, otherwise None.
    """
    if not isinstance(stmt, cst.SimpleStatementLine):
        return None
    for small in stmt.body:
        if isinstance(small, cst.ImportFrom):
            return small
    return None


def _import_module_name(import_from: cst.ImportFrom) -> str | None:
    """
    Extract the module name from an ImportFrom node.

    Returns
    -------
    str | None
        Module name if resolvable, otherwise None.
    """
    if isinstance(import_from.module, cst.Name):
        return import_from.module.value
    if isinstance(import_from.module, cst.Attribute):
        return import_from.module.attr.value
    return None


def _has_type_checking_alias(names: list[cst.ImportAlias | cst.ImportStar]) -> bool:
    """
    Return True when TYPE_CHECKING alias is present in an import list.

    Returns
    -------
    bool
        True if TYPE_CHECKING is imported.
    """
    for alias in names:
        if (
            isinstance(alias, cst.ImportAlias)
            and isinstance(alias.name, cst.Name)
            and alias.name.value == "TYPE_CHECKING"
        ):
            return True
    return False


def _has_type_checking_import(body: Iterable[cst.BaseStatement]) -> bool:
    for stmt in body:
        import_from = _simple_import_from(stmt)
        module_name = _import_module_name(import_from) if import_from else None
        if module_name == "typing" and _has_type_checking_alias(import_from.names):
            return True
    return False


def _import_insertion_index(body: list[cst.BaseStatement]) -> int:
    """
    Determine insertion index after module docstring and initial imports.

    Returns
    -------
    int
        Index where new imports should be inserted.
    """
    idx = 0
    if (
        body
        and isinstance(body[0], cst.SimpleStatementLine)
        and len(body[0].body) == 1
        and isinstance(body[0].body[0], cst.Expr)
        and isinstance(body[0].body[0].value, cst.SimpleString)
    ):
        idx = 1
    while idx < len(body):
        stmt = body[idx]
        if not isinstance(stmt, cst.SimpleStatementLine):
            break
        if all(isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body):
            idx += 1
            continue
        break
    return idx


def _find_type_checking_block(
    body: list[cst.BaseStatement],
) -> tuple[int | None, cst.If | None]:
    for index, stmt in enumerate(body):
        if (
            isinstance(stmt, cst.If)
            and isinstance(stmt.test, cst.Name)
            and stmt.test.value == "TYPE_CHECKING"
        ):
            return index, stmt
    return None, None


def _line_number(
    stmt: cst.BaseStatement,
    positions: dict[cst.CSTNode, metadata.Position],
) -> int | None:
    position = positions.get(stmt)
    return position.start.line if position else None


def _is_import_statement(stmt: cst.BaseStatement) -> bool:
    return isinstance(stmt, cst.SimpleStatementLine) and all(
        isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
    )


def _collect_movable_imports(
    body: list[cst.BaseStatement],
    target_lines: set[int],
    positions: dict[cst.CSTNode, metadata.Position],
) -> tuple[list[cst.BaseSmallStatement], list[cst.BaseStatement]]:
    moved: list[cst.BaseSmallStatement] = []
    remaining: list[cst.BaseStatement] = []
    for stmt in body:
        line_no = _line_number(stmt, positions)
        if line_no in target_lines and _is_import_statement(stmt):
            if isinstance(stmt, cst.SimpleStatementLine):
                moved.extend(
                    small for small in stmt.body if isinstance(small, (cst.Import, cst.ImportFrom))
                )
            continue
        remaining.append(stmt)
    return moved, remaining


def _run_ruff(argv: list[str]) -> tuple[int, str, str]:
    """
    Run Ruff programmatically and capture output streams.

    Returns
    -------
    tuple[int, str, str]
        Exit code, stdout, and stderr content.
    """
    ruff_module = importlib.import_module("ruff.__main__")
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = ruff_module.main(argv)
    return exit_code, stdout.getvalue(), stderr.getvalue()


def _collect_tch_issues(paths: list[str]) -> dict[Path, set[int]]:
    args = ["check", "--select", "TCH", "--output-format", "json", *paths]
    exit_code, stdout, stderr = _run_ruff(args)

    if exit_code not in {0, 1}:
        sys.stderr.write(stderr)
        raise SystemExit(exit_code)

    try:
        issues = json.loads(stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to parse ruff output: {exc}\n")
        raise SystemExit(1) from exc

    locations: dict[Path, set[int]] = defaultdict(set)
    for issue in issues:
        if not str(issue.get("code", "")).startswith("TCH"):
            continue
        filename = issue.get("filename")
        line = issue.get("location", {}).get("row")
        if filename and line is not None:
            locations[Path(filename)].add(int(line))
    return locations


def _apply_moves(path: Path, lines: set[int]) -> bool:
    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    wrapper = metadata.MetadataWrapper(module)
    positions = wrapper.resolve(metadata.PositionProvider)

    moved_imports, remaining_body = _collect_movable_imports(list(module.body), lines, positions)
    if not moved_imports:
        return False

    body = list(remaining_body)
    if not _has_type_checking_import(body):
        type_import = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Name("typing"),
                    names=[cst.ImportAlias(name=cst.Name("TYPE_CHECKING"))],
                )
            ]
        )
        insert_at = _import_insertion_index(body)
        body.insert(insert_at, type_import)

    moved_lines = [cst.SimpleStatementLine(body=[imp]) for imp in moved_imports]
    tc_index, tc_if = _find_type_checking_block(body)
    if tc_index is not None and tc_if is not None:
        new_body = list(tc_if.body.body) + moved_lines
        body[tc_index] = tc_if.with_changes(body=cst.IndentedBlock(body=new_body))
    else:
        tc_if = cst.If(test=cst.Name("TYPE_CHECKING"), body=cst.IndentedBlock(body=moved_lines))
        insert_at = _import_insertion_index(body)
        body.insert(insert_at + 1, tc_if)

    new_module = module.with_changes(body=body)
    if new_module.code == source:
        return False

    path.write_text(new_module.code, encoding="utf-8")
    return True


def main() -> int:
    """
    CLI entrypoint.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Move type-only imports under TYPE_CHECKING using Ruff TCH diagnostics."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["src", "tests"],
        help="Paths to scan (default: src tests)",
    )
    args = parser.parse_args()

    locations = _collect_tch_issues(args.paths)
    if not locations:
        sys.stdout.write("No TCH issues detected; no changes made.\n")
        return 0

    changed = 0
    for path, lines in locations.items():
        if not path.exists():
            continue
        if _apply_moves(path, lines):
            changed += 1
            sys.stdout.write(f"Moved type-only imports in {path}\n")

    sys.stdout.write(f"Completed import moves. Files updated: {changed}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
