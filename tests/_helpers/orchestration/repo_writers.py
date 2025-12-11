"""Filesystem setup helpers for test repositories.

This module provides functions that write Python source files to disk
for test repository setup, including sample packages and coverage drivers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def write_sample_repo(repo_root: Path) -> list[Path]:
    """Create a minimal but realistic Python package for ingestion.

    Parameters
    ----------
    repo_root
        Root directory where the package will be created.

    Returns
    -------
    list[Path]
        Paths for the files created under the repo root.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")

    mod_path = pkg_dir / "mod.py"
    mod_path.write_text(
        "\n".join(
            [
                "def hello(name: str) -> str:",
                '    """Return greeting."""',
                '    return f"hi {name}"',
                "",
                "def adder(x: int, y: int) -> int:",
                "    return x + y",
            ]
        ),
        encoding="utf8",
    )
    files.append(mod_path)

    util_path = pkg_dir / "util.py"
    util_path.write_text(
        "\n".join(
            [
                "from pkg.mod import hello",
                "",
                "def loud(name: str) -> str:",
                "    msg = hello(name)",
                "    return msg.upper()",
            ]
        ),
        encoding="utf8",
    )
    files.append(util_path)

    return files


def write_callgraph_alias_repo(repo_root: Path) -> list[Path]:
    """Create a repo exercising alias/relative-import callgraph paths.

    Parameters
    ----------
    repo_root
        Root directory where the package will be created.

    Returns
    -------
    list[Path]
        Paths of the files written under the repo root.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    callee_path = pkg_dir / "a.py"
    callee_path.write_text(
        "\n".join(
            [
                "def foo():",
                "    return 1",
                "",
                "class C:",
                "    def helper(self):",
                "        return foo()",
            ]
        ),
        encoding="utf8",
    )
    files.append(callee_path)

    caller_path = pkg_dir / "b.py"
    caller_path.write_text(
        "\n".join(
            [
                "from .a import foo as f, C",
                "import pkg.a as pa",
                "",
                "def caller():",
                "    f()",
                "    obj = C()",
                "    obj.helper()",
                "    pa.foo()",
                "    unknown_call()",
            ]
        ),
        encoding="utf8",
    )
    files.append(caller_path)

    return files


def write_graph_metrics_repo(repo_root: Path) -> list[Path]:
    """Write a simple repo suitable for graph metrics computation.

    Parameters
    ----------
    repo_root
        Root directory where the package will be created.

    Returns
    -------
    list[Path]
        Paths of the files written under the repo root.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    mod_a = pkg_dir / "mod_a.py"
    mod_a.write_text(
        "\n".join(
            [
                "import pkg.mod_b",
                "",
                "def a(x: int) -> int:",
                "    return pkg.mod_b.b(x) + 1",
            ]
        ),
        encoding="utf8",
    )
    files.append(mod_a)
    mod_b = pkg_dir / "mod_b.py"
    mod_b.write_text(
        "\n".join(
            [
                "def b(x: int) -> int:",
                "    return x * 2",
            ]
        ),
        encoding="utf8",
    )
    files.append(mod_b)
    return files


def write_coverage_driver(repo_root: Path, files: list[Path]) -> Path:
    """Write a driver that imports repo modules to generate real coverage data.

    Parameters
    ----------
    repo_root
        Root directory of the repository.
    files
        List of Python files to import in the driver.

    Returns
    -------
    Path
        Path to the generated driver module.
    """
    driver_path = repo_root / "_coverage_driver.py"
    module_names: list[str] = []
    for path in files:
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        module = rel.with_suffix("").as_posix().replace("/", ".")
        if module.endswith(".__init__"):
            module = module.rsplit(".", 1)[0]
        if module:
            module_names.append(module)
    module_names = sorted(set(module_names))

    lines: list[str] = ["import importlib", "from contextlib import suppress"]
    if module_names:
        lines.append(f"MODULES = {module_names!r}")
        lines.append("for name in MODULES:")
        lines.append("    with suppress(Exception):")
        lines.append("        importlib.import_module(name)")
    else:
        lines.append("pass")

    driver_path.write_text("\n".join(lines), encoding="utf8")
    return driver_path


__all__ = [
    "write_callgraph_alias_repo",
    "write_coverage_driver",
    "write_graph_metrics_repo",
    "write_sample_repo",
]
