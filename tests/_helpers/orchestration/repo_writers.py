"""Filesystem setup helpers for test repositories.

This module provides functions that write Python source files to disk
for test repository setup, including sample packages and coverage drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class RepoFixture:
    """Repo fixture metadata with expected module inventory."""

    files: tuple[Path, ...]
    module_map: dict[str, str]

    def module_paths(self) -> list[str]:
        """Return sorted module paths from the expected module map.

        Returns
        -------
        list[str]
            Sorted module paths from the expected map.
        """
        return sorted(self.module_map.keys())


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf8")
    return path


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


def write_monorepo_fixture(
    repo_root: Path,
    *,
    include_tests: bool = True,
) -> RepoFixture:
    """Create a multi-language monorepo with deterministic Python modules.

    Returns
    -------
    RepoFixture
        Fixture metadata including expected module inventory.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "services" / "py_service" / "src" / "app.py",
            "\n".join(
                [
                    "def run() -> int:",
                    "    return 42",
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "services" / "py_service" / "src" / "__init__.py",
            "",
        )
    )
    files.append(
        _write_file(
            repo_root / "libs" / "shared" / "util.py",
            "\n".join(
                [
                    "def greet(name: str) -> str:",
                    '    return f"hi {name}"',
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "apps" / "web" / "index.ts",
            "export const value = 1;",
        )
    )
    if include_tests:
        files.append(
            _write_file(
                repo_root / "services" / "py_service" / "tests" / "test_app.py",
                "\n".join(
                    [
                        "from services.py_service.src.app import run",
                        "",
                        "def test_run() -> None:",
                        "    assert run() == 42",
                    ]
                ),
            )
        )

    options = ModuleIngestOptions(include_tests=include_tests)
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_generated_noise_fixture(
    repo_root: Path,
    *,
    include_generated: bool = False,
) -> RepoFixture:
    """Create a repo with generated file noise.

    Returns
    -------
    RepoFixture
        Fixture metadata including expected module inventory.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "main.py",
            "\n".join(
                [
                    "def main() -> None:",
                    "    return None",
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "generated" / "service_pb2.py",
            "class Stub: pass",
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "models_generated.py",
            "class Generated: pass",
        )
    )

    options = ModuleIngestOptions(include_generated=include_generated)
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_large_file_fixture(
    repo_root: Path,
    *,
    max_file_size_kb: int = 1,
) -> RepoFixture:
    """Create a repo with a file exceeding max size limits.

    Returns
    -------
    RepoFixture
        Fixture metadata including expected module inventory.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "small.py",
            "VALUE = 1",
        )
    )
    large_payload = "x" * (max_file_size_kb * 1024 + 10)
    files.append(
        _write_file(
            repo_root / "src" / "large.py",
            large_payload,
        )
    )

    options = ModuleIngestOptions(max_file_size_kb=max_file_size_kb)
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_scoped_paths_fixture(
    repo_root: Path,
    *,
    scope_paths: list[str],
) -> RepoFixture:
    """Create a repo with scoped paths for module discovery.

    Returns
    -------
    RepoFixture
        Fixture metadata including expected module inventory.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "pkg_a" / "__init__.py",
            "",
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "pkg_a" / "mod.py",
            "VALUE = 'a'",
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "pkg_b" / "mod.py",
            "VALUE = 'b'",
        )
    )

    options = ModuleIngestOptions(scope_paths=scope_paths)
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


__all__ = [
    "RepoFixture",
    "write_callgraph_alias_repo",
    "write_coverage_driver",
    "write_generated_noise_fixture",
    "write_graph_metrics_repo",
    "write_large_file_fixture",
    "write_monorepo_fixture",
    "write_sample_repo",
    "write_scoped_paths_fixture",
]
