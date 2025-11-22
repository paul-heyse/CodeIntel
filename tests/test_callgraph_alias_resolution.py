"""Call graph resolution tests for import alias heuristics."""

from __future__ import annotations

from pathlib import Path

from codeintel.graphs.callgraph_builder import resolve_callee_for_testing


def test_callgraph_resolves_import_alias(tmp_path: Path) -> None:
    """
    Call graph builder should resolve calls made through an import alias.

    Raises
    ------
    AssertionError
        If alias-based resolution fails to return the expected GOID.
    """
    repo_root = tmp_path / "repo"
    _callee_path, _caller_path = _write_repo_files(repo_root)
    callee_by_name = {"pkg.callee.target": 1}
    global_callee_by_name = {"pkg.callee.target": 1, "target": 1}
    import_aliases = {"cc": "pkg.callee"}

    goid, resolved_via, confidence = resolve_callee_for_testing(
        callee_name="cc",
        attr_chain=["cc", "target"],
        callee_by_name=callee_by_name,
        global_callee_by_name=global_callee_by_name,
        import_aliases=import_aliases,
    )

    if goid != 1:
        message = f"Expected callee GOID 1, got {goid}"
        raise AssertionError(message)
    if resolved_via not in {"import_alias", "local_attr", "import_alias_global"}:
        message = f"Unexpected resolution mode: {resolved_via}"
        raise AssertionError(message)
    if confidence <= 0.0:
        message = "Confidence should be positive for alias resolution"
        raise AssertionError(message)


def _write_repo_files(repo_root: Path) -> tuple[Path, Path]:
    """
    Create minimal caller/callee modules for alias resolution.

    Returns
    -------
    tuple[Path, Path]
        Paths to the callee and caller Python files.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    callee_path = pkg_dir / "callee.py"
    callee_path.write_text(
        "def target():\n    return 1\n",
        encoding="utf-8",
    )
    caller_path = pkg_dir / "caller.py"
    caller_path.write_text(
        "import pkg.callee as cc\n\ndef caller():\n    return cc.target()\n",
        encoding="utf-8",
    )
    return callee_path, caller_path
