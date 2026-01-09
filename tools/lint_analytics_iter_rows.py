"""Guardrail for iter_rows usage in analytics modules."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from tools.lint_analytics_guardrails import scan_analytics


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analytics iter_rows guardrail.

    Returns
    -------
    int
        Exit code (0 for success, 1 for violations).
    """
    args = list(argv) if argv is not None else []
    root = Path(args[0]).resolve() if args else Path.cwd().resolve()
    findings = scan_analytics(root)
    violations = findings.iter_rows

    if not violations:
        return 0

    output_lines = [
        f"{violation.path.relative_to(root)}:{violation.lineno}: {violation.message}"
        for violation in violations
    ]
    output_lines.append(f"{len(violations)} analytics iter_rows guardrail violation(s).")
    sys.stderr.write("\n".join(output_lines) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
