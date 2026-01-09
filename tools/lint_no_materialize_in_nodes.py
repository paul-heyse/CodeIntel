"""Lint for table materialization in build/ingestion nodes."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from tools.lint_build_ingestion_guardrails import scan_build_ingestion


def main(argv: Sequence[str] | None = None) -> int:
    """Run the materialization lint.

    Parameters
    ----------
    argv
        Optional CLI arguments, with the repo root as the first entry.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    findings = scan_build_ingestion(root)
    violations = findings.materialize

    if not violations:
        return 0

    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    output_lines.append(f"{len(violations)} materialization call(s) detected.")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
