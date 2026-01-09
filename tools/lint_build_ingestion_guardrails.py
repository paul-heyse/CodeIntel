"""Combined guardrails for build/ingestion node anti-patterns."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ast_grep_py import SgRoot

from tools.lint_file_utils import find_literal_candidates, list_python_files

_ALLOWLIST_RAW_COMPUTE: frozenset[str] = frozenset(
    {
        "src/codeintel/build/tabular/arrow_ops.py",
        "src/codeintel/build/tabular/array_ops.py",
    }
)
_ALLOWLIST_MATERIALIZE: frozenset[str] = frozenset(
    {
        "src/codeintel/ingestion/compute/cst_extract.py",
        "src/codeintel/ingestion/compute/tree_sitter_index.py",
    }
)

_SCAN_DIRS: tuple[str, ...] = (
    "src/codeintel/build/hamilton",
    "src/codeintel/ingestion/compute",
)

_RAW_COMPUTE_MESSAGE = (
    "Raw pyarrow.compute import detected; use core DSL helpers "
    "(codeintel.core.columnar.expr_vocab/kernels) instead."
)
_MATERIALIZE_MESSAGE = (
    "Materialization via to_table() detected; keep readers streaming "
    "and finalize at explicit boundaries."
)
_COMPUTE_NAME_RE = re.compile(r"\bcompute\b")


@dataclass(frozen=True)
class Violation:
    """Single lint violation discovered during scanning."""

    path: Path
    lineno: int
    message: str


@dataclass(frozen=True)
class BuildIngestionScan:
    """Container for build/ingestion lint findings."""

    raw_compute: list[Violation]
    materialize: list[Violation]


def _candidate_paths(root: Path) -> set[Path]:
    include_globs = tuple(f"{dirname}/**/*.py" for dirname in _SCAN_DIRS)
    return find_literal_candidates(
        root,
        patterns=("pyarrow.compute", "from pyarrow import", "to_table"),
        include_globs=include_globs,
    )


def _iter_python_files(root: Path) -> Iterable[Path]:
    yield from list_python_files(root, _SCAN_DIRS)


def _parse_root(path: Path) -> SgRoot | None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return SgRoot(source, "python")
    except (RuntimeError, ValueError):
        return None


def _rel_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _scan_raw_compute(root: SgRoot, *, path: Path, repo_root: Path) -> list[Violation]:
    rel = _rel_path(path, root=repo_root)
    if rel in _ALLOWLIST_RAW_COMPUTE:
        return []
    violations: list[Violation] = []
    tree = root.root()
    for node in tree.find_all(kind="import_statement"):
        text = " ".join(node.text().split())
        if "pyarrow.compute" in text:
            violations.append(
                Violation(
                    path=path,
                    lineno=node.range().start.line + 1,
                    message=_RAW_COMPUTE_MESSAGE,
                )
            )
    for node in tree.find_all(kind="import_from_statement"):
        text = " ".join(node.text().split())
        if text.startswith("from pyarrow.compute"):
            violations.append(
                Violation(
                    path=path,
                    lineno=node.range().start.line + 1,
                    message=_RAW_COMPUTE_MESSAGE,
                )
            )
            continue
        if text.startswith("from pyarrow import") and _COMPUTE_NAME_RE.search(text):
            violations.append(
                Violation(
                    path=path,
                    lineno=node.range().start.line + 1,
                    message=_RAW_COMPUTE_MESSAGE,
                )
            )
    return violations


def _scan_materialize(root: SgRoot, *, path: Path, repo_root: Path) -> list[Violation]:
    rel = _rel_path(path, root=repo_root)
    if rel in _ALLOWLIST_MATERIALIZE:
        return []
    tree = root.root()
    return [
        Violation(
            path=path,
            lineno=node.range().start.line + 1,
            message=_MATERIALIZE_MESSAGE,
        )
        for node in tree.find_all(pattern="$OBJ.to_table($$$ARGS)")
    ]


def scan_build_ingestion(repo_root: Path) -> BuildIngestionScan:
    """Scan build/ingestion directories for guarded patterns.

    Parameters
    ----------
    repo_root
        Repository root for path resolution.

    Returns
    -------
    BuildIngestionScan
        Aggregated findings for raw compute imports and materialization calls.
    """
    candidate_paths = _candidate_paths(repo_root)
    if not candidate_paths:
        return BuildIngestionScan(raw_compute=[], materialize=[])
    raw_violations: list[Violation] = []
    materialize_violations: list[Violation] = []
    for path in _iter_python_files(repo_root):
        if path not in candidate_paths:
            continue
        parsed_root = _parse_root(path)
        if parsed_root is None:
            continue
        raw_violations.extend(_scan_raw_compute(parsed_root, path=path, repo_root=repo_root))
        materialize_violations.extend(
            _scan_materialize(parsed_root, path=path, repo_root=repo_root)
        )
    return BuildIngestionScan(raw_compute=raw_violations, materialize=materialize_violations)


def _emit_violations(violations: list[Violation], *, root: Path) -> None:
    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    sys.stderr.write("\n".join(output_lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the combined build/ingestion guardrails.

    Parameters
    ----------
    argv
        Optional CLI args, with the repo root as the first entry.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    repo_root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    findings = scan_build_ingestion(repo_root)
    had_violations = False
    if findings.raw_compute:
        _emit_violations(findings.raw_compute, root=repo_root)
        sys.stderr.write(
            f"{len(findings.raw_compute)} raw pyarrow.compute import(s) detected.\n"
        )
        had_violations = True
    if findings.materialize:
        _emit_violations(findings.materialize, root=repo_root)
        sys.stderr.write(
            f"{len(findings.materialize)} materialization call(s) detected.\n"
        )
        had_violations = True
    return 1 if had_violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
