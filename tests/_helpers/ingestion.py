"""Shared helpers for ingestion plugin tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.config import BuildConfig
from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.hamilton.helpers import paths_to_modules
from codeintel.build.providers import create_default_providers
from codeintel.build.schemas import column_names_for_table_key
from codeintel.build.targets import OutputTarget
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import ScanProfile, default_code_profile
from codeintel.storage.warehouse import MaterializeOptions, Warehouse
from tests._helpers.assertions.modules import ModulesAssertions
from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.factories import make_snapshot
from tests._helpers.fakes.tools import write_dummy_scip_files
from tests._helpers.fixtures.repos import write_tree
from tests._helpers.gateway import GatewayFactory
from tests._helpers.harnesses.hamilton_build import BuildEnvSpec, build_test_env
from tests._helpers.modules_expectations import (
    module_paths_expected_from_repo_tree,
    modules_expected_from_env,
)
from tests._helpers.scip_proto import ensure_proto_module
from tests._helpers.scip_proto import write_scip_index as write_proto_index
from tests._helpers.tool_payloads import pytest_report_payload

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.providers import Providers
    from codeintel.ingestion.compute.repo_scan import RepoScanResult
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway


def _make_ingestion_target(name: str, description: str = "") -> OutputTarget:
    """Create a minimal OutputTarget for ingestion testing.

    Parameters
    ----------
    name
        Target name (e.g., "repo_scan", "tests_ingest").
    description
        Optional description.

    Returns
    -------
    OutputTarget
        Minimal target suitable for test execution.
    """
    return OutputTarget(
        name=name,
        module="ingestion",
        contract=EMPTY_CONTRACT,
        dependencies=(),
        description=description or f"Test target for {name}",
    )


__all__ = [
    "IngestionContextBundle",
    "ModuleInventoryContext",
    "RepoVariantOptions",
    "ScanSetup",
    "ScanSetupOptions",
    "ScipIngestContext",
    "SeedIngestionConfig",
    "TargetContextConfig",
    "build_ingestion_adapters",
    "build_ingestion_context_bundle",
    "build_repo_with_configs",
    "build_repo_with_variants",
    "build_scan_profile",
    "build_scan_setup",
    "build_scip_ingest_context",
    "build_scip_repo_fixture",
    "build_target_context_for_target",
    "closing_gateway",
    "create_scan_and_docstring_steps",
    "create_scan_step",
    "make_resource_case_params",
    "make_scan_setup",
    "materialize_repo_scan_result",
    "materialize_rows_for_snapshot",
    "module_inventory_context",
    "module_paths_from_context",
    "module_records_for_paths",
    "repo_variants",
    "seed_foreign_key_tables",
    "seed_ingestion_tables",
    "seed_inventory_from_paths",
    "seed_modules_and_repo_map",
    "seed_numeric_table",
    "seed_varchar_table",
    "write_coverage_file",
    "write_dummy_scip_files",
    "write_pytest_report",
    "write_scip_index",
    "write_tree",
]


@dataclass(frozen=True)
class TargetContextConfig:
    """Configuration for building a Hamilton ``BuildEnv`` for tests."""

    repo_root: Path | None = None
    providers: Providers | None = None
    gateway: StorageGateway | None = None
    gateway_factory: GatewayFactory | None = None
    snapshot: SnapshotRef | tuple[str, str] | None = None
    paths: BuildPaths | None = None
    profile: str | None = None


@dataclass(frozen=True)
class IngestionContextBundle:
    """Bundle of adapters, gateway, and module metadata for ingestion tests."""

    repo_root: Path
    gateway: StorageGateway
    ctx: BuildEnv
    storage: DuckDBStorageAdapter
    discovery: FilesystemDiscoveryAdapter
    change_detection: HashChangeDetectionAdapter
    tools: ToolRunnerAdapter
    module_paths: tuple[str, ...]


@dataclass(frozen=True)
class RepoVariantOptions:
    """Options for constructing sample repositories."""

    repo_structure: Mapping[str, str] | None = None
    include_invalid: bool = False
    include_macros: bool = False
    include_symlinks: bool = False
    module_paths: Sequence[str] | None = None


@dataclass(frozen=True)
class ScanSetup:
    """Shared scan/profile wiring for ingestion pipelines."""

    repo_root: Path
    gateway: StorageGateway
    profile: ScanProfile
    scan_step: RepoScanStep
    storage: DuckDBStorageAdapter
    discovery: FilesystemDiscoveryAdapter


@dataclass(frozen=True)
class ModuleInventoryContext:
    """Pre-baked context for module inventory round-trip tests."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    profile: ScanProfile
    scan_step: RepoScanStep
    storage: DuckDBStorageAdapter
    discovery: FilesystemDiscoveryAdapter


@dataclass(frozen=True)
class ScanSetupOptions:
    """Options controlling scan/profile setup."""

    repo_structure: Mapping[str, str] | None = None
    include_invalid: bool = False
    include_globs: Sequence[str] = ("*.py",)
    ignore_dirs: Sequence[str] = ()
    log_every: int | None = None
    log_interval: float | None = None
    gateway_factory: GatewayFactory | None = None


def build_target_context_for_target(
    target: OutputTarget,
    tmp_path: Path,
    *,
    config: TargetContextConfig | None = None,
) -> BuildEnv:
    """Construct a Hamilton ``BuildEnv`` wired to real adapters.

    Parameters
    ----------
    target
        Output target for the context (unused; retained for callsite compatibility).
    tmp_path
        Temporary directory for test isolation.
    config
        Optional configuration overrides.

    Returns
    -------
    BuildEnv
        Frozen execution environment for Hamilton and test helpers.
    """
    _ = target
    cfg = config or TargetContextConfig()
    effective_repo_root = cfg.repo_root or (tmp_path / "repo")
    effective_repo_root.mkdir(parents=True, exist_ok=True)

    gateway = cfg.gateway
    if gateway is None:
        factory = cfg.gateway_factory or GatewayFactory().with_macros()
        gateway = factory.open()

    snapshot = cfg.snapshot
    if isinstance(snapshot, tuple):
        snapshot_ref = make_snapshot(
            repo=snapshot[0],
            commit=snapshot[1],
            repo_root=effective_repo_root,
        )
    else:
        snapshot_ref = snapshot or make_snapshot(repo_root=effective_repo_root)

    build_paths = cfg.paths
    if build_paths is None:
        build_dir = effective_repo_root / "build"
        db_path = build_dir / "db" / "codeintel.duckdb"
        build_paths = BuildPaths(
            build_dir=build_dir,
            db_path=db_path,
            document_output_dir=build_dir / "document_output",
            scip_dir=build_dir / "scip",
            coverage_json=build_dir / "coverage" / "coverage.json",
            pytest_report=build_dir / "test-results" / "pytest-report.json",
            tool_cache=build_dir / ".tool_cache",
            log_db_path=build_dir / "logs" / "logs.duckdb",
        )

    providers = cfg.providers or create_default_providers(ToolsConfig.default())

    return build_test_env(
        BuildEnvSpec(
            gateway=gateway,
            snapshot=snapshot_ref,
            paths=build_paths,
            providers=providers,
            build_config=BuildConfig.empty(),
            settings=TEST_BUILD_SETTINGS,
            profile=cfg.profile,
        )
    )


def build_ingestion_adapters(
    ctx: BuildEnv,
) -> tuple[
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    ToolRunnerAdapter,
]:
    """Create ingestion adapters aligned with a target context.

    Returns
    -------
    tuple
        storage, discovery, change detection, and tool adapters.
    """
    storage = DuckDBStorageAdapter(ctx.gateway)
    discovery = FilesystemDiscoveryAdapter(ctx.snapshot.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    tools = ToolRunnerAdapter(ctx.providers.tool_service)
    return storage, discovery, change_detection, tools


def build_repo_with_variants(
    tmp_path: Path,
    *,
    include_invalid: bool = False,
    include_macros: bool = False,
    include_symlinks: bool = False,
    extra_structure: Mapping[str, str] | None = None,
) -> Path:
    """Construct a sample repository with optional invalid files, macros, and symlinks.

    Returns
    -------
    Path
        Path to the created repository root.
    """
    structure: dict[str, str] = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
        "README.md": "# Sample repo\n",
    }
    if include_invalid:
        structure["pkg/invalid.py"] = "this is not valid python"
    if include_macros:
        structure["macros/ingest.sql"] = "-- macros for ingestion\n"
    if extra_structure:
        structure.update(extra_structure)
    repo_root = tmp_path / "repo"
    write_tree(repo_root, structure)
    if include_symlinks:
        target = repo_root / "pkg" / "mod.py"
        symlink_path = repo_root / "pkg" / "mod_link.py"
        if not symlink_path.exists():
            symlink_path.symlink_to(target)
    return repo_root


def repo_variants(
    base_structure: Mapping[str, str] | None = None,
    *,
    invalid_structure: Mapping[str, str] | None = None,
    macro_structure: Mapping[str, str] | None = None,
) -> dict[str, RepoVariantOptions]:
    """Construct common repo variants with invalid files, macros, and symlinks.

    Returns
    -------
    dict[str, RepoVariantOptions]
        Variants keyed by label (base, with_invalid, with_macros,
        with_invalid_and_macros, with_symlink).
    """
    structure = base_structure or {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
    }
    invalid = invalid_structure or {"pkg/invalid.py": "this is not valid python"}
    macros = macro_structure or {"macros/ingest.sql": "-- macros for ingestion\n"}

    return {
        "base": RepoVariantOptions(repo_structure=structure),
        "with_invalid": RepoVariantOptions(repo_structure={**structure, **invalid}),
        "with_macros": RepoVariantOptions(repo_structure={**structure, **macros}),
        "with_invalid_and_macros": RepoVariantOptions(
            repo_structure={**structure, **invalid, **macros}
        ),
        "with_symlink": RepoVariantOptions(repo_structure=structure, include_symlinks=True),
    }


def build_ingestion_context_bundle(
    tmp_path: Path,
    *,
    variants: RepoVariantOptions | None = None,
    gateway_factory: GatewayFactory | None = None,
) -> IngestionContextBundle:
    """Build a fully wired ingestion context bundle from a repo spec.

    Returns
    -------
    IngestionContextBundle
        Bundle containing repo root, gateway, adapters, and seeded module paths.
    """
    opts = variants or RepoVariantOptions()
    if opts.repo_structure is not None:
        repo_root = tmp_path / "repo"
        write_tree(repo_root, opts.repo_structure)
    else:
        repo_root = build_repo_with_variants(
            tmp_path,
            include_invalid=opts.include_invalid,
            include_macros=opts.include_macros,
            include_symlinks=opts.include_symlinks,
        )
    gateway = (gateway_factory or GatewayFactory().with_macros()).open()
    target = _make_ingestion_target("repo_scan", "Repository scan target for testing")
    ctx = build_target_context_for_target(
        target,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, tools = build_ingestion_adapters(ctx)
    if opts.module_paths is not None:
        seeded_paths = tuple(opts.module_paths)
    else:
        seeded_paths = tuple(module_paths_expected_from_repo_tree(repo_root))
    if seeded_paths:
        seed_modules_and_repo_map(ctx, seeded_paths)
    return IngestionContextBundle(
        repo_root=repo_root,
        gateway=gateway,
        ctx=ctx,
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
        tools=tools,
        module_paths=seeded_paths,
    )


def materialize_rows_for_snapshot(
    gateway: StorageGateway,
    table_key: str,
    rows: Sequence[tuple[object, ...]],
    *,
    snapshot: SnapshotRef,
) -> None:
    """Materialize row tuples into a snapshot-scoped table."""
    columns = column_names_for_table_key(table_key)
    resolved_columns = columns if columns else None
    warehouse = Warehouse(gateway)
    warehouse.materialize_rows(
        table_key,
        rows,
        columns=resolved_columns,
        options=MaterializeOptions(
            mode="replace",
            snapshot=snapshot,
        ),
    )


def materialize_repo_scan_result(
    gateway: StorageGateway,
    scan_result: RepoScanResult,
    *,
    snapshot: SnapshotRef,
) -> None:
    """Persist repo scan rows for module inventory tables."""
    materialize_rows_for_snapshot(
        gateway,
        "core.modules",
        scan_result.module_rows,
        snapshot=snapshot,
    )
    materialize_rows_for_snapshot(
        gateway,
        "core.file_state",
        scan_result.file_state_rows,
        snapshot=snapshot,
    )
    materialize_rows_for_snapshot(
        gateway,
        "core.repo_map",
        scan_result.repo_map_rows,
        snapshot=snapshot,
    )


def write_coverage_file(
    build_dir: Path,
    *,
    filename: str = "coverage.json",
    content: str | Mapping[str, object] | None = None,
) -> Path:
    """Write a coverage artifact into the build directory.

    Returns
    -------
    Path
        Path to the written coverage file.
    """
    coverage_path = build_dir / filename
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(content) if isinstance(content, Mapping) else content or "{}"
    coverage_path.write_text(payload, encoding="utf-8")
    return coverage_path


def write_pytest_report(
    build_dir: Path,
    *,
    tests: list[dict[str, object]] | None = None,
    summary: Mapping[str, int] | None = None,
    filename: str = "pytest-report.json",
) -> Path:
    """Render a pytest JSON report under the build/test-results directory.

    Returns
    -------
    Path
        Path to the written report file.
    """
    report_dir = build_dir / "test-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = pytest_report_payload(
        tests=tests or [],
        summary=summary or {"passed": 0, "failed": 0, "skipped": 0},
    )
    report_path = report_dir / filename
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def write_scip_index(
    build_dir: Path,
    documents: Sequence[Mapping[str, object]],
    *,
    filename: str = "index.scip",
) -> Path:
    """
    Write a SCIP protobuf index under ``build/scip``.

    Parameters
    ----------
    build_dir
        Base build directory for the target context.
    documents
        Iterable of SCIP document payloads.
    filename
        Optional filename override (defaults to ``index.scip``).

    Returns
    -------
    Path
        Path to the written index file.
    """
    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    index_path = scip_dir / filename
    typed_documents = [dict(doc) for doc in documents]
    proto_module_path = ensure_proto_module()
    write_proto_index(
        index_path,
        proto_module_path=proto_module_path,
        documents=typed_documents,
    )
    return index_path


def build_repo_with_configs(
    tmp_path: Path,
    *,
    include_invalid: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    """Create a realistic repo layout with optional invalid config.

    Returns
    -------
    tuple[Path, tuple[str, ...]]
        Repo root and the Python module paths created.
    """
    structure: dict[str, str] = {
        "pkg/__init__.py": "",
        "pkg/service.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
        "config/app.yaml": "service:\n  retries: 3\n  hosts:\n    - api.local\n",
        "config/settings.toml": 'feature = true\n[db]\nurl = "duckdb:///tmp.db"\n',
        "config/app.ini": "[service]\ntimeout=30\n",
    }
    if include_invalid:
        structure["config/broken.yml"] = ":\n  - invalid\n"

    repo_root = tmp_path / "repo"
    write_tree(repo_root, structure)
    modules = tuple(path for path in structure if path.endswith(".py"))
    return repo_root, modules


def build_scan_profile(
    repo_root: Path,
    *,
    include_globs: Sequence[str] = ("*.py",),
    ignore_dirs: Sequence[str] = (),
    log_every: int | None = None,
    log_interval: float | None = None,
) -> ScanProfile:
    """Create a ScanProfile with shared defaults.

    Returns
    -------
    ScanProfile
        Profile covering the repo root with optional ignore patterns.
    """
    if log_every is None and log_interval is None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
        )
    if log_every is None and log_interval is not None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
            log_interval=log_interval,
        )
    if log_interval is None and log_every is not None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
            log_every=log_every,
        )
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=tuple(include_globs),
        ignore_dirs=tuple(ignore_dirs),
        log_every=log_every or 0,
        log_interval=log_interval or 0.0,
    )


def create_scan_step(
    gateway: StorageGateway,
    repo_root: Path,
    tmp_path: Path,
) -> tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]:
    """Create scan step and adapters for a repository.

    Returns
    -------
    tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]
        Repo scan step plus the storage and discovery adapters backing it.
    """
    target = _make_ingestion_target("repo_scan", "Repository scan step")
    ctx = build_target_context_for_target(
        target,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, _ = build_ingestion_adapters(ctx)
    scan_step = RepoScanStep(discovery=discovery, change_detection=change_detection)
    return scan_step, storage, discovery


def make_scan_setup(
    tmp_path: Path,
    *,
    options: ScanSetupOptions | None = None,
) -> ScanSetup:
    """Build a repo, profile, and scan step bundle for reuse across tests.

    Returns
    -------
    ScanSetup
        Bundle containing repo root, gateway, profile, and scan adapters.
    """
    opts = options or ScanSetupOptions()
    if opts.repo_structure is not None:
        repo_root = tmp_path / "repo"
        write_tree(repo_root, opts.repo_structure)
    else:
        repo_root = build_repo_with_variants(tmp_path, include_invalid=opts.include_invalid)
    gateway = (opts.gateway_factory or GatewayFactory().with_macros()).open()
    profile = build_scan_profile(
        repo_root,
        include_globs=opts.include_globs,
        ignore_dirs=opts.ignore_dirs,
        log_every=opts.log_every,
        log_interval=opts.log_interval,
    )
    scan_step, storage, discovery = create_scan_step(gateway, repo_root, tmp_path)
    return ScanSetup(
        repo_root=repo_root,
        gateway=gateway,
        profile=profile,
        scan_step=scan_step,
        storage=storage,
        discovery=discovery,
    )


def build_scan_setup(
    tmp_path: Path,
    *,
    options: ScanSetupOptions | None = None,
) -> ScanSetup:
    """Alias for make_scan_setup to keep older call sites working.

    Returns
    -------
    ScanSetup
        Bundle containing repo root, gateway, profile, and scan adapters.
    """
    return make_scan_setup(tmp_path, options=options)


def create_scan_and_docstring_steps(
    gateway: StorageGateway,
    repo_root: Path,
    tmp_path: Path,
) -> tuple[RepoScanStep, DocstringsExtractStep]:
    """Create scan and docstring steps from gateway and repo root.

    Returns
    -------
    tuple[RepoScanStep, DocstringsExtractStep]
        Configured repo scan and docstring extraction steps.
    """
    scan_step, _storage, discovery = create_scan_step(gateway, repo_root, tmp_path)
    doc_step = DocstringsExtractStep(discovery=discovery)
    return scan_step, doc_step


@dataclass(frozen=True)
class ScipIngestContext:
    """Bundle of SCIP ingest fixtures reused across tests."""

    repo_root: Path
    gateway: StorageGateway
    storage: DuckDBStorageAdapter
    tools: ToolRunnerAdapter
    build_dir: Path


def build_scip_ingest_context(tmp_path: Path) -> ScipIngestContext:
    """Create a repo, gateway, and adapters for SCIP ingest tests.

    Returns
    -------
    ScipIngestContext
        Context containing repo_root, gateway, adapters, and build_dir.
    """
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        {"pkg/__init__.py": "", "pkg/mod.py": "def foo(x: int) -> int:\n    return x + 1\n"},
    )
    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    gateway = GatewayFactory().file_backed(db_path).with_macros().open()
    target = _make_ingestion_target("tests_ingest", "Tests ingestion target")
    ctx = build_target_context_for_target(
        target,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, _ = build_ingestion_adapters(ctx)
    _ = discovery, change_detection
    tools_config = ToolsConfig.default()
    runner = ToolRunner(tools_config=tools_config, cache_dir=build_dir / ".tool_cache")
    service = ToolService(runner, tools_config)
    tool_adapter = ToolRunnerAdapter(service)
    return ScipIngestContext(
        repo_root=repo_root,
        gateway=gateway,
        storage=storage,
        tools=tool_adapter,
        build_dir=build_dir,
    )


def build_scip_repo_fixture(tmp_path: Path) -> ScipIngestContext:
    """Variant of build_scip_ingest_context that also seeds dummy SCIP artifacts.

    Returns
    -------
    ScipIngestContext
        Context pre-seeded with dummy SCIP artifacts.
    """
    context = build_scip_ingest_context(tmp_path)
    write_dummy_scip_files(context.build_dir)
    return context


def module_paths_from_context(ctx: BuildEnv) -> list[str]:
    """Fetch module paths using the shared helper.

    Returns
    -------
    list[str]
        Module paths for the target context.
    """
    return sorted(modules_expected_from_env(ctx).keys())


def module_records_for_paths(
    paths: Sequence[str],
    repo_root: Path,
) -> list[ModuleRecord]:
    """Convert relative paths to ModuleRecord objects with metadata.

    Returns
    -------
    list[ModuleRecord]
        Module records containing module names and file paths.
    """
    return paths_to_modules(paths, repo_root)


@contextmanager
def module_inventory_context(
    tmp_path: Path,
    *,
    repo_structure: Mapping[str, str] | None = None,
    gateway_factory: GatewayFactory | None = None,
) -> Generator[ModuleInventoryContext]:
    """Create a ready-to-use module inventory context.

    Yields
    ------
    ModuleInventoryContext
        Snapshot, gateway, profile, and adapters for inventory round-trips.
    """
    repo_root = tmp_path / "repo"
    write_tree(
        repo_root,
        repo_structure
        or {
            "src/pkg/a.py": "print('a')\n",
            "src/pkg/b.py": "print('b')\n",
        },
    )
    snapshot = make_snapshot(repo="demo", commit="abc123", repo_root=repo_root)
    gateway = (gateway_factory or GatewayFactory().with_macros()).open()
    profile = default_code_profile(repo_root)
    scan_step, storage, discovery = create_scan_step(gateway, repo_root, tmp_path)
    ctx = ModuleInventoryContext(
        snapshot=snapshot,
        gateway=gateway,
        profile=profile,
        scan_step=scan_step,
        storage=storage,
        discovery=discovery,
    )
    with closing_gateway(gateway):
        yield ctx


@contextmanager
def closing_gateway(gateway: StorageGateway) -> Generator[StorageGateway]:
    """Ensure storage gateways are closed after use.

    Yields
    ------
    StorageGateway
        The provided gateway, guaranteed to be closed on exit.
    """
    try:
        yield gateway
    finally:
        gateway.close()


def seed_modules_and_repo_map(
    ctx: BuildEnv,
    paths: Sequence[str],
) -> None:
    """
    Insert module rows and a repo_map entry for the provided paths.

    Parameters
    ----------
    ctx
        Build environment with gateway and snapshot metadata.
    paths
        Iterable of module-relative file paths to seed.
    """
    repo_root = ctx.snapshot.repo_root
    records = module_records_for_paths(paths, repo_root)
    con = ctx.gateway.con
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [ctx.snapshot.repo, ctx.snapshot.commit],
    )
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [ctx.snapshot.repo, ctx.snapshot.commit],
    )
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            (
                record.module_name,
                record.file_path.relative_to(repo_root).as_posix(),
                ctx.snapshot.repo,
                ctx.snapshot.commit,
            )
            for record in records
        ],
    )
    modules_json = {
        record.module_name: record.file_path.relative_to(repo_root).as_posix() for record in records
    }
    con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, '{}', CURRENT_TIMESTAMP)
        """,
        [ctx.snapshot.repo, ctx.snapshot.commit, json.dumps(modules_json)],
    )
    ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()


def seed_inventory_from_paths(
    repo_root: Path,
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    paths: Sequence[str],
) -> None:
    """Seed core.modules and repo_map tables from relative paths."""
    records = module_records_for_paths(paths, repo_root)
    con = gateway.con
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            (
                record.module_name,
                record.file_path.relative_to(repo_root).as_posix(),
                repo,
                commit,
            )
            for record in records
        ],
    )
    modules_json = {
        record.module_name: record.file_path.relative_to(repo_root).as_posix() for record in records
    }
    con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, '{}', CURRENT_TIMESTAMP)
        """,
        [repo, commit, json.dumps(modules_json)],
    )
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    ModulesAssertions(gateway, snapshot).inventory_consistent()


@dataclass(frozen=True)
class SeedIngestionConfig:
    """Configuration for seeding common ingestion tables."""

    module_paths: Sequence[str] | None = None
    numeric_tables: Mapping[str, Sequence[float]] | None = None
    varchar_tables: Mapping[str, Sequence[tuple[int, str | None]]] | None = None
    foreign_keys: (
        Sequence[tuple[str, str, Sequence[tuple[int, str]], Sequence[tuple[int, int | None]]]]
        | None
    ) = None
    include_defaults: bool = True
    include_orphans: bool = False
    include_duplicates: bool = False


def seed_ingestion_tables(
    ctx: BuildEnv,
    config: SeedIngestionConfig | None = None,
) -> None:
    """Seed common ingestion tables in a single call."""
    cfg = config or SeedIngestionConfig()
    if cfg.module_paths:
        seed_modules_and_repo_map(ctx, cfg.module_paths)

    tables_seeded = False
    if cfg.numeric_tables:
        tables_seeded = True
        for table, values in cfg.numeric_tables.items():
            seed_numeric_table(ctx.gateway, table, values)
    if cfg.varchar_tables:
        tables_seeded = True
        for table, values in cfg.varchar_tables.items():
            seed_varchar_table(ctx.gateway, table, values)
    if cfg.foreign_keys:
        tables_seeded = True
        for parent_table, child_table, parent_rows, child_rows in cfg.foreign_keys:
            seed_foreign_key_tables(
                ctx.gateway,
                parent_table=parent_table,
                child_table=child_table,
                parent_rows=parent_rows,
                child_rows=child_rows,
            )

    if cfg.include_defaults and not tables_seeded:
        seed_numeric_table(
            ctx.gateway,
            "core.test_numeric",
            [10.5, 5.0, 20.0, 5.0] if cfg.include_duplicates else [10.5, 5.0, 20.0],
        )
        seed_varchar_table(
            ctx.gateway,
            "core.test_varchar",
            [
                (1, "alpha"),
                (2, "beta"),
                (3, "gamma"),
                (4, "beta") if cfg.include_duplicates else (4, "delta"),
            ],
        )
        default_parent_rows: list[tuple[int, str]] = [(1, "Parent 1"), (2, "Parent 2")]
        default_child_rows: list[tuple[int, int | None]] = [(1, 1), (2, 1)]
        if cfg.include_orphans:
            default_child_rows.append((3, None))
        seed_foreign_key_tables(
            ctx.gateway,
            parent_table="core.test_parent",
            child_table="core.test_child",
            parent_rows=default_parent_rows,
            child_rows=default_child_rows,
        )


def seed_foreign_key_tables(
    gateway: StorageGateway,
    *,
    parent_table: str,
    child_table: str,
    parent_rows: Sequence[tuple[int, str]],
    child_rows: Sequence[tuple[int, int | None]],
) -> None:
    """
    Create parent/child tables and populate rows for orphan-ref tests.

    Raises
    ------
    ValueError
        If the provided table pair is not supported.
    """
    sql_lookup: dict[tuple[str, str], tuple[str, str, str, str, str, str]] = {
        (
            "core.test_parent",
            "core.test_child",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent",
            "DELETE FROM core.test_child",
            "INSERT INTO core.test_parent (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child (id, parent_id) VALUES (?, ?)",
        ),
        (
            "core.test_parent2",
            "core.test_child2",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent2 (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child2 (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent2",
            "DELETE FROM core.test_child2",
            "INSERT INTO core.test_parent2 (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child2 (id, parent_id) VALUES (?, ?)",
        ),
        (
            "core.test_parent3",
            "core.test_child3",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent3 (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child3 (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent3",
            "DELETE FROM core.test_child3",
            "INSERT INTO core.test_parent3 (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child3 (id, parent_id) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get((parent_table, child_table))
    if queries is None:
        message = f"Unsupported foreign key test tables: {parent_table}, {child_table}"
        raise ValueError(message)

    (
        create_parent_sql,
        create_child_sql,
        delete_parent_sql,
        delete_child_sql,
        insert_parent_sql,
        insert_child_sql,
    ) = queries
    gateway.con.execute(create_parent_sql)
    gateway.con.execute(create_child_sql)
    gateway.con.execute(delete_parent_sql)
    gateway.con.execute(delete_child_sql)
    if parent_rows:
        gateway.con.executemany(
            insert_parent_sql,
            list(parent_rows),
        )
    if child_rows:
        gateway.con.executemany(
            insert_child_sql,
            list(child_rows),
        )


def seed_numeric_table(gateway: StorageGateway, table: str, values: Sequence[float]) -> None:
    """
    Create or truncate a numeric table and insert provided values.

    Raises
    ------
    ValueError
        If an unsupported table name is provided.
    """
    sql_lookup: dict[str, tuple[str, str, str]] = {
        "core.test_numeric": (
            "CREATE TABLE IF NOT EXISTS core.test_numeric (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_numeric",
            "INSERT INTO core.test_numeric (id, value) VALUES (?, ?)",
        ),
        "core.test_numeric2": (
            "CREATE TABLE IF NOT EXISTS core.test_numeric2 (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_numeric2",
            "INSERT INTO core.test_numeric2 (id, value) VALUES (?, ?)",
        ),
        "core.test_empty_num": (
            "CREATE TABLE IF NOT EXISTS core.test_empty_num (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_empty_num",
            "INSERT INTO core.test_empty_num (id, value) VALUES (?, ?)",
        ),
        "core.test_empty_num2": (
            "CREATE TABLE IF NOT EXISTS core.test_empty_num2 (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_empty_num2",
            "INSERT INTO core.test_empty_num2 (id, value) VALUES (?, ?)",
        ),
        "core.test_pos": (
            "CREATE TABLE IF NOT EXISTS core.test_pos (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_pos",
            "INSERT INTO core.test_pos (id, value) VALUES (?, ?)",
        ),
        "core.test_all_pos": (
            "CREATE TABLE IF NOT EXISTS core.test_all_pos (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_all_pos",
            "INSERT INTO core.test_all_pos (id, value) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get(table)
    if queries is None:
        message = f"Unsupported numeric table for tests: {table}"
        raise ValueError(message)

    create_sql, delete_sql, insert_sql = queries
    gateway.con.execute(create_sql)
    gateway.con.execute(delete_sql)
    params = [(idx, value) for idx, value in enumerate(values, start=1)]
    if params:
        gateway.con.executemany(insert_sql, params)


def seed_varchar_table(
    gateway: StorageGateway,
    table: str,
    values: Sequence[tuple[int, str | None]],
) -> None:
    """
    Create or truncate a VARCHAR table and insert provided rows.

    Raises
    ------
    ValueError
        If an unsupported table name is provided.
    """
    sql_lookup: dict[str, tuple[str, str, str]] = {
        "core.test_nulls": (
            "CREATE TABLE IF NOT EXISTS core.test_nulls (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_nulls",
            "INSERT INTO core.test_nulls (id, value) VALUES (?, ?)",
        ),
        "core.test_dupes": (
            "CREATE TABLE IF NOT EXISTS core.test_dupes (id INTEGER, name VARCHAR)",
            "DELETE FROM core.test_dupes",
            "INSERT INTO core.test_dupes (id, name) VALUES (?, ?)",
        ),
        "core.test_unique": (
            "CREATE TABLE IF NOT EXISTS core.test_unique (id INTEGER, name VARCHAR)",
            "DELETE FROM core.test_unique",
            "INSERT INTO core.test_unique (id, name) VALUES (?, ?)",
        ),
        "core.test_frac1": (
            "CREATE TABLE IF NOT EXISTS core.test_frac1 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac1",
            "INSERT INTO core.test_frac1 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac2": (
            "CREATE TABLE IF NOT EXISTS core.test_frac2 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac2",
            "INSERT INTO core.test_frac2 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac3": (
            "CREATE TABLE IF NOT EXISTS core.test_frac3 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac3",
            "INSERT INTO core.test_frac3 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac_empty": (
            "CREATE TABLE IF NOT EXISTS core.test_frac_empty (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac_empty",
            "INSERT INTO core.test_frac_empty (id, value) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get(table)
    if queries is None:
        message = f"Unsupported varchar table for tests: {table}"
        raise ValueError(message)

    create_sql, delete_sql, insert_sql = queries
    gateway.con.execute(create_sql)
    gateway.con.execute(delete_sql)
    if values:
        gateway.con.executemany(insert_sql, list(values))


def make_resource_case_params() -> tuple[tuple[str, dict[str, bool]], ...]:
    """Provide standard resource/gateway failure cases for parametrized tests.

    Returns
    -------
    tuple[tuple[str, dict[str, bool]], ...]
        Parametrization tuples describing resource and gateway failure modes.
    """
    return (
        (
            "resources",
            {
                "simulate_resources": True,
                "simulate_db_fallback": False,
                "simulate_gateway_failure": False,
            },
        ),
        (
            "db_fallback",
            {
                "simulate_resources": False,
                "simulate_db_fallback": True,
                "simulate_gateway_failure": False,
            },
        ),
        (
            "gateway_failure",
            {
                "simulate_resources": False,
                "simulate_db_fallback": False,
                "simulate_gateway_failure": True,
            },
        ),
    )
