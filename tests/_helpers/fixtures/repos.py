"""Unified repository fixtures for tests."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from codeintel.build.hamilton.native.options.ingestion import ModuleIngestOptions
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


RepoFixtureKind = Literal["canonical", "callgraph_alias", "graph_metrics", "sample", "custom"]


MOD_A_PATH: Final[str] = "pkg/mod_a.py"
MOD_B_PATH: Final[str] = "pkg/mod_b.py"
MOD_C_PATH: Final[str] = "pkg/mod_c.py"
MOD_UTIL_PATH: Final[str] = "pkg/util.py"

MOD_A_FQN: Final[str] = "pkg.mod_a"
MOD_B_FQN: Final[str] = "pkg.mod_b"
MOD_C_FQN: Final[str] = "pkg.mod_c"
MOD_UTIL_FQN: Final[str] = "pkg.util"

GOID_FUNC_A: Final[int] = 1001
GOID_FUNC_B: Final[int] = 1002
GOID_FUNC_C: Final[int] = 1003
GOID_HELPER: Final[int] = 1004
GOID_CALLER: Final[int] = 1005
GOID_CALLEE: Final[int] = 1006


@dataclass(frozen=True)
class CanonicalFunction:
    """Function metadata for canonical repo fixtures."""

    qualname: str
    goid: int
    rel_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CanonicalRepo:
    """Canonical repo layout including module paths and GOID metadata."""

    module_paths: dict[str, str]
    goids: dict[str, int]
    functions: dict[str, CanonicalFunction]


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
            Sorted module paths.
        """
        return sorted(self.module_map.keys())


@dataclass(frozen=True)
class RepoFixtureSpec:
    """Specification for writing repo fixtures."""

    kind: RepoFixtureKind
    repo_root: Path
    files: Mapping[str, str] | None = None
    module_map: Mapping[str, str] | None = None


class RepoFixtureWriter:
    """Writer for standardized repository fixtures."""

    @staticmethod
    def write(spec: RepoFixtureSpec) -> RepoFixture:
        """Write a repo fixture from a specification.

        Returns
        -------
        RepoFixture
            Written fixture metadata.

        Raises
        ------
        ValueError
            If the fixture kind is unsupported.
        """
        if spec.kind == "canonical":
            fixture = write_canonical_repo(spec.repo_root)
            return RepoFixture(
                files=tuple(spec.repo_root / path for path in fixture.module_paths.values()),
                module_map=fixture.module_paths,
            )
        if spec.kind == "callgraph_alias":
            return _write_callgraph_alias_repo(spec.repo_root)
        if spec.kind == "graph_metrics":
            return _write_graph_metrics_repo(spec.repo_root)
        if spec.kind == "sample":
            return _write_sample_repo(spec.repo_root)
        if spec.kind == "custom":
            return write_tree(spec.repo_root, spec.files or {})
        message = f"Unsupported repo fixture kind: {spec.kind}"
        raise ValueError(message)

    @staticmethod
    def write_tree(root: Path, files: Mapping[str, str]) -> RepoFixture:
        """Write a custom repo tree from path/content pairs.

        Returns
        -------
        RepoFixture
            Written fixture metadata.
        """
        return write_tree(root, files)


def write_tree(root: Path, files: Mapping[str, str]) -> RepoFixture:
    """Write a repo tree and return fixture metadata.

    Returns
    -------
    RepoFixture
        Written fixture metadata.
    """
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    module_map = modules_expected_from_repo_tree(root)
    return RepoFixture(files=tuple(written), module_map=module_map)


def _write_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_canonical_repo(repo_root: Path) -> CanonicalRepo:
    """Write canonical pkg/ modules and return metadata.

    Returns
    -------
    CanonicalRepo
        Canonical repo metadata and module paths.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    module_sources = {
        MOD_A_PATH: "\n".join(
            [
                "def func_a(x: int, y: int) -> int:",
                "    from pkg.mod_b import func_b",
                "    return func_b(x) + y",
            ]
        ),
        MOD_B_PATH: "\n".join(
            [
                "def func_b(x: int) -> int:",
                "    from pkg.mod_c import func_c",
                "    total = x * 2",
                "    for value in func_c():",
                "        total += value",
                "    return total",
            ]
        ),
        MOD_C_PATH: "\n".join(
            [
                "def func_c() -> range:",
                "    return range(1, 4)",
            ]
        ),
        MOD_UTIL_PATH: "\n".join(
            [
                "def helper(value: int) -> int:",
                "    return value",
            ]
        ),
    }
    written_paths: dict[str, Path] = {}
    for rel_path, source in module_sources.items():
        abs_path = repo_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(source + "\n", encoding="utf-8")
        written_paths[rel_path] = abs_path

    locations = _compute_function_meta(written_paths)
    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    func_meta: dict[str, CanonicalFunction] = {}
    for qualname, meta in locations.items():
        goid = goids.get(qualname)
        if goid is None:
            continue
        func_meta[qualname] = CanonicalFunction(
            qualname=qualname,
            goid=goid,
            rel_path=meta.rel_path,
            start_line=meta.start_line,
            end_line=meta.end_line,
        )

    module_map = {
        MOD_A_FQN: MOD_A_PATH,
        MOD_B_FQN: MOD_B_PATH,
        MOD_C_FQN: MOD_C_PATH,
        MOD_UTIL_FQN: MOD_UTIL_PATH,
    }
    return CanonicalRepo(
        module_paths=module_map,
        goids=goids,
        functions=func_meta,
    )


@dataclass(frozen=True)
class FunctionLocation:
    """Function start/end locations within a file."""

    rel_path: str
    start_line: int
    end_line: int


def _compute_function_meta(written_paths: dict[str, Path]) -> dict[str, FunctionLocation]:
    meta: dict[str, FunctionLocation] = {}
    for rel_path, abs_path in written_paths.items():
        source = abs_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                qualname = node.name
                start_line = getattr(node, "lineno", 1)
                end_line = getattr(node, "end_lineno", start_line)
                meta[qualname] = FunctionLocation(
                    rel_path=rel_path,
                    start_line=start_line,
                    end_line=end_line,
                )
    return meta


def _write_sample_repo(repo_root: Path) -> RepoFixture:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")

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
        encoding="utf-8",
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
        encoding="utf-8",
    )
    files.append(util_path)

    module_map = modules_expected_from_repo_tree(repo_root)
    return RepoFixture(files=tuple(files), module_map=module_map)


def _write_callgraph_alias_repo(repo_root: Path) -> RepoFixture:
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
        encoding="utf-8",
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
        encoding="utf-8",
    )
    files.append(caller_path)

    module_map = modules_expected_from_repo_tree(repo_root)
    return RepoFixture(files=tuple(files), module_map=module_map)


def _write_graph_metrics_repo(repo_root: Path) -> RepoFixture:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
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
        encoding="utf-8",
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
        encoding="utf-8",
    )
    files.append(mod_b)
    module_map = modules_expected_from_repo_tree(repo_root)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_sample_repo(repo_root: Path) -> list[Path]:
    """Create a minimal but realistic Python package for ingestion.

    Returns
    -------
    list[Path]
        Paths written to the repository root.
    """
    return list(_write_sample_repo(repo_root).files)


def write_callgraph_alias_repo(repo_root: Path) -> list[Path]:
    """Create a repo exercising alias/relative-import callgraph paths.

    Returns
    -------
    list[Path]
        Paths written to the repository root.
    """
    return list(_write_callgraph_alias_repo(repo_root).files)


def write_graph_metrics_repo(repo_root: Path) -> list[Path]:
    """Write a simple repo suitable for graph metrics computation.

    Returns
    -------
    list[Path]
        Paths written to the repository root.
    """
    return list(_write_graph_metrics_repo(repo_root).files)


def write_coverage_driver(repo_root: Path, files: list[Path]) -> Path:
    """Write a driver that imports repo modules to generate real coverage data.

    Returns
    -------
    Path
        Driver file path.
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

    driver_path.write_text("\n".join(lines), encoding="utf-8")
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
        Written fixture metadata.
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
        Written fixture metadata.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "main.py",
            "\n".join(
                [
                    "def main() -> int:",
                    "    return 0",
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "__init__.py",
            "",
        )
    )
    if include_generated:
        files.append(
            _write_file(
                repo_root / "src" / "generated" / "gen.py",
                "# generated file\n",
            )
        )
    module_map = modules_expected_from_repo_tree(repo_root)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_large_file_fixture(
    repo_root: Path,
    *,
    max_bytes: int,
) -> RepoFixture:
    """Create a repo with files above and below a size threshold.

    Returns
    -------
    RepoFixture
        Written fixture metadata.

    Raises
    ------
    ValueError
        If ``max_bytes`` is not positive.
    """
    if max_bytes <= 0:
        message = "max_bytes must be positive"
        raise ValueError(message)
    limit_kb = max(1, math.ceil(max_bytes / 1024))
    max_bytes_limit = limit_kb * 1024
    small_target = max(1, min(128, max_bytes_limit - 1))

    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "small_module.py",
            _pad_source(["def keep() -> int:", "    return 1"], small_target),
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "large_module.py",
            _pad_source(["def big() -> int:", "    return 1"], max_bytes_limit + 1),
        )
    )

    options = ModuleIngestOptions(max_file_size_kb=limit_kb)
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


def write_scoped_paths_fixture(
    repo_root: Path,
    *,
    scope_paths: Sequence[str],
) -> RepoFixture:
    """Create a repo that exercises scoped path filtering.

    Returns
    -------
    RepoFixture
        Written fixture metadata.
    """
    files: list[Path] = []
    files.append(
        _write_file(
            repo_root / "src" / "core" / "alpha.py",
            "\n".join(
                [
                    "def alpha() -> int:",
                    "    return 1",
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "src" / "extras" / "beta.py",
            "\n".join(
                [
                    "def beta() -> int:",
                    "    return 2",
                ]
            ),
        )
    )
    files.append(
        _write_file(
            repo_root / "tools" / "gamma.py",
            "\n".join(
                [
                    "def gamma() -> int:",
                    "    return 3",
                ]
            ),
        )
    )

    options = ModuleIngestOptions(scope_paths=list(scope_paths))
    module_map = modules_expected_from_repo_tree(repo_root, options=options)
    return RepoFixture(files=tuple(files), module_map=module_map)


def _pad_source(lines: list[str], target_bytes: int) -> str:
    base = "\n".join(lines) + "\n"
    if len(base) >= target_bytes:
        return base
    pad_line = "# padding"
    pad_len = len(pad_line) + 1
    remaining = target_bytes - len(base)
    pad_count = math.ceil(remaining / pad_len)
    return base + ("\n".join([pad_line] * pad_count)) + "\n"


__all__ = [
    "GOID_CALLEE",
    "GOID_CALLER",
    "GOID_FUNC_A",
    "GOID_FUNC_B",
    "GOID_FUNC_C",
    "GOID_HELPER",
    "MOD_A_FQN",
    "MOD_A_PATH",
    "MOD_B_FQN",
    "MOD_B_PATH",
    "MOD_C_FQN",
    "MOD_C_PATH",
    "MOD_UTIL_FQN",
    "MOD_UTIL_PATH",
    "CanonicalFunction",
    "CanonicalRepo",
    "RepoFixture",
    "RepoFixtureSpec",
    "RepoFixtureWriter",
    "write_callgraph_alias_repo",
    "write_canonical_repo",
    "write_coverage_driver",
    "write_generated_noise_fixture",
    "write_graph_metrics_repo",
    "write_large_file_fixture",
    "write_monorepo_fixture",
    "write_sample_repo",
    "write_scoped_paths_fixture",
    "write_tree",
]
