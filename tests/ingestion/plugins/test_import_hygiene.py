"""Guardrails for plugin test imports."""

from __future__ import annotations

from pathlib import Path


def test_plugin_tests_avoid_cross_test_imports() -> None:
    """
    Plugin tests should rely on helpers, not other test modules.

    Raises
    ------
    AssertionError
        If a plugin test imports another test module directly.
    """
    plugin_tests_dir = Path(__file__).parent
    repo_root = Path().resolve()
    banned_marker = "tests.ingestion.test_"

    offending_files: list[str] = []
    for path in plugin_tests_dir.glob("test_*.py"):
        rel_path = path.relative_to(repo_root)
        if rel_path == Path("tests/ingestion/plugins/test_import_hygiene.py"):
            continue
        content = path.read_text(encoding="utf8")
        if banned_marker in content:
            offending_files.append(str(rel_path))

    if offending_files:
        message = (
            "Plugin tests must not import other test modules: "
            f"{', '.join(sorted(offending_files))}"
        )
        raise AssertionError(message)
