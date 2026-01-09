"""Guardrail: prohibit raw pyarrow.compute imports outside core helpers."""

from __future__ import annotations

import re
from pathlib import Path

ALLOWLIST = (
    Path("src/codeintel/core"),
    Path("src/codeintel/build/tabular"),
)


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / "pyproject.toml").exists():
            return root
        root = root.parent
    msg = "Unable to locate repository root for guardrail test."
    raise RuntimeError(msg)


def _is_allowed(path: Path) -> bool:
    return any(path.is_relative_to(allowed) for allowed in ALLOWLIST)


def test_no_raw_pc_imports() -> None:
    """Reject raw pyarrow.compute imports outside allowed paths."""
    root = _repo_root()
    bad: list[str] = []
    pattern = re.compile(r"^\s*import\s+pyarrow\.compute\s+as\s+pc", re.MULTILINE)
    for file_path in root.glob("src/codeintel/**/*.py"):
        rel_path = file_path.relative_to(root)
        if _is_allowed(rel_path):
            continue
        text = file_path.read_text(encoding="utf-8")
        if pattern.search(text):
            bad.append(str(rel_path))
    assert not bad, f"Raw pyarrow.compute imports found: {', '.join(sorted(bad))}"
