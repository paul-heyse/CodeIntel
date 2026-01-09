"""Guardrail to prevent to_pylist usage in build code."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.lint_file_utils import find_literal_candidates


def _scan_file(path: Path) -> list[str]:
    """Return guardrail match lines for a single file.

    Returns
    -------
    list[str]
        Violation entries for the file.
    """
    matches: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"{path}:0:0:read_error:{exc}"]
    for index, line in enumerate(lines, start=1):
        if "to_pylist(" in line:
            matches.append(f"{path}:{index}:1:{line.strip()}")
    return matches


def _candidate_paths(repo_root: Path) -> list[Path]:
    candidates = find_literal_candidates(
        repo_root,
        patterns=("to_pylist(",),
        include_globs=("src/codeintel/build/**/*.py",),
    )
    return sorted(candidates)


def main() -> int:
    """Run the guardrail and return the exit status.

    Returns
    -------
    int
        Zero when clean, non-zero when violations are found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    target_root = repo_root / "src" / "codeintel" / "build"
    if not target_root.exists():
        sys.stderr.write(f"Guardrail: missing target root {target_root}\n")
        return 1
    violations: list[str] = []
    for path in _candidate_paths(repo_root):
        violations.extend(_scan_file(path))
    if violations:
        sys.stderr.write("Guardrail: to_pylist usage is not allowed in build code.\n")
        for entry in violations:
            sys.stderr.write(f"{entry}\n")
        return 1
    sys.stdout.write("Guardrail: no to_pylist usage found in build code.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
