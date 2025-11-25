"""Provisioning helpers for production-parity gateway-backed tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Self

from codeintel.analytics.cfg_dfg_metrics import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.graph_metrics import compute_graph_metrics
from codeintel.config.models import (
    CallGraphConfig,
    CoverageIngestConfig,
    GraphMetricsConfig,
    RepoScanConfig,
    TypingIngestConfig,
)
from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.ingestion.coverage_ingest import ingest_coverage_lines
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.ingestion.typing_ingest import ingest_typing_signals
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    open_gateway,
    open_memory_gateway,
)
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.builders import (
    AstMetricsRow,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    CoverageFunctionRow,
    DFGEdgeRow,
    DocstringRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    FunctionValidationRow,
    GoidCrosswalkRow,
    GoidRow,
    HotspotRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SymbolUseEdgeRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    TypednessRow,
    insert_ast_metrics,
    insert_call_graph_edges,
    insert_call_graph_nodes,
    insert_cfg_blocks,
    insert_cfg_edges,
    insert_coverage_functions,
    insert_dfg_edges,
    insert_docstrings,
    insert_function_metrics,
    insert_function_types,
    insert_function_validation,
    insert_goid_crosswalk,
    insert_goids,
    insert_hotspots,
    insert_import_graph_edges,
    insert_modules,
    insert_repo_map,
    insert_risk_factors,
    insert_static_diagnostics,
    insert_symbol_use_edges,
    insert_test_catalog,
    insert_test_coverage_edges,
    insert_typedness,
)
from tests._helpers.fakes import FakeToolRunner, utcnow

DEFAULT_REPO: Final = "demo/repo"
DEFAULT_COMMIT: Final = "deadbeef"


@dataclass(frozen=True)
class ProvisionedGateway:
    """Container for an ingested gateway and associated filesystem context."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    db_path: Path
    document_output_dir: Path
    coverage_file: Path
    gateway: StorageGateway
    runner: FakeToolRunner

    def close(self) -> None:
        """Close the underlying gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """
        Enter context manager scope.

        Returns
        -------
        ProvisionedGateway
            Self reference for use within a context block.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close gateway on context exit."""
        self.close()


@dataclass(frozen=True)
class RepoContext:
    """Paths and identifiers for a test repository."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    db_path: Path
    document_output_dir: Path


@dataclass(frozen=True)
class ProvisionOptions:
    """Options controlling ingestion seeds for provisioned gateways."""

    include_typing: bool = True
    include_coverage: bool = True
    build_graph_metrics: bool = False
    file_backed: bool = False
    db_path: Path | None = None
    include_seed_goid: bool = True


@dataclass(frozen=True)
class GatewayOptions:
    """Options controlling gateway setup without ingestion."""

    db_path: Path | None = None
    apply_schema: bool = True
    ensure_views: bool = True
    validate_schema: bool = True
    file_backed: bool = True
    strict_schema: bool = True


@dataclass(frozen=True)
class ProvisioningConfig:
    """Configuration for context-managed gateway provisioning."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    provision_options: ProvisionOptions | None = None
    gateway_options: GatewayOptions | None = None
    run_ingestion: bool = True


@contextmanager
def provisioned_gateway(
    repo_root: Path,
    config: ProvisioningConfig | None = None,
) -> Iterator[ProvisionedGateway]:
    """
    Context manager wrapping gateway provisioning and cleanup.

    Parameters
    ----------
    repo_root
        Root directory for the repository under test.
    config
        Provisioning configuration; defaults mirror ProvisioningConfig.

    Yields
    ------
    ProvisionedGateway
        Provisioned gateway scoped to the repo root.
    """
    cfg = config or ProvisioningConfig()
    if cfg.run_ingestion:
        ctx = provision_ingested_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.provision_options,
        )
    else:
        ctx = provision_gateway_with_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.gateway_options,
        )
    try:
        yield ctx
    finally:
        ctx.close()


def _write_sample_repo(repo_root: Path) -> list[Path]:
    """Create a minimal but realistic Python package for ingestion.

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


def _graph_metrics_repo(repo_root: Path) -> list[Path]:
    """Write a simple repo suitable for graph metrics computation.

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


def make_repo_context(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    db_path: Path | None = None,
) -> RepoContext:
    """Build a RepoContext with derived build/document paths.

    Returns
    -------
    RepoContext
        Derived paths and identifiers for the repo.
    """
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    document_output_dir = repo_root / "Document Output"
    db = db_path or build_dir / "db" / "codeintel.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    return RepoContext(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db,
        document_output_dir=document_output_dir,
    )


def _make_runner(repo_root: Path, files: list[Path]) -> FakeToolRunner:
    """Build a FakeToolRunner configured with deterministic payloads.

    Returns
    -------
    FakeToolRunner
        Runner seeded with canned tool outputs for the sample repo.
    """
    coverage_payload = {
        "files": {
            str(path.resolve()): {
                "executed_lines": [1],
                "missing_lines": [],
            }
            for path in files
        }
    }
    payloads = {
        "pyright": '{"generalDiagnostics": []}',
        "pyrefly": "",
        "ruff": "[]",
        "json": coverage_payload,
    }
    cache_dir = repo_root / "build" / ".tool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return FakeToolRunner(cache_dir=cache_dir, payloads=payloads)


def _open_gateway_from_context(ctx: RepoContext, opts: GatewayOptions) -> StorageGateway:
    effective_ensure_views = opts.ensure_views or opts.strict_schema
    effective_validate_schema = opts.validate_schema or opts.strict_schema
    if opts.file_backed:
        cfg = StorageConfig(
            db_path=ctx.db_path,
            read_only=False,
            apply_schema=opts.apply_schema,
            ensure_views=effective_ensure_views,
            validate_schema=effective_validate_schema,
        )
        return open_gateway(cfg)
    return open_memory_gateway(
        apply_schema=opts.apply_schema,
        ensure_views=effective_ensure_views,
        validate_schema=effective_validate_schema,
    )


def provision_ingested_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """
    Build a sample repo, run ingestion steps, and return a provisioned gateway.

    The gateway uses real schemas/views and populates:
    - core.modules/core.repo_map via ingest_repo
    - analytics.typedness/static_diagnostics via ingest_typing_signals
    - analytics.coverage_lines via ingest_coverage_lines

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway plus filesystem context for tests.
    """
    opts = options or ProvisionOptions()
    repo_root.mkdir(parents=True, exist_ok=True)
    ctx = make_repo_context(repo_root, repo=repo, commit=commit, db_path=opts.db_path)
    coverage_file = repo_root / ".coverage"
    coverage_file.write_text("", encoding="utf8")

    files = _write_sample_repo(repo_root)
    runner = _make_runner(repo_root, files)

    gateway_opts = GatewayOptions(file_backed=opts.file_backed)
    gateway = _open_gateway_from_context(ctx, gateway_opts)
    ingest_repo(
        gateway,
        cfg=RepoScanConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit),
    )
    if opts.include_typing:
        ingest_typing_signals(
            gateway,
            cfg=TypingIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit),
            runner=runner,
        )
    if opts.include_coverage:
        ingest_coverage_lines(
            gateway,
            cfg=CoverageIngestConfig.from_paths(
                repo_root=repo_root,
                repo=repo,
                commit=commit,
                coverage_file=coverage_file,
            ),
            runner=runner,
        )
    if opts.build_graph_metrics:
        _seed_cfg_dfg_for_metrics(gateway, rel_path="pkg/mod.py")
        compute_cfg_metrics(gateway, repo=repo, commit=commit)
        compute_dfg_metrics(gateway, repo=repo, commit=commit)

    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=ctx.build_dir,
        db_path=ctx.db_path,
        document_output_dir=ctx.document_output_dir,
        coverage_file=coverage_file,
        gateway=gateway,
        runner=runner,
    )


def provision_existing_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """
    Run ingestion over an existing repo tree using production entry points.

    Mirrors `provision_ingested_repo` but assumes callers have already written the
    desired repo contents to disk.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway plus repo context.
    """
    opts = options or ProvisionOptions()
    repo_root.mkdir(parents=True, exist_ok=True)
    ctx = make_repo_context(repo_root, repo=repo, commit=commit, db_path=opts.db_path)
    coverage_file = repo_root / ".coverage"
    coverage_file.write_text("", encoding="utf8")

    files = sorted(path for path in repo_root.rglob("*.py") if path.is_file())
    runner = _make_runner(repo_root, files)

    gateway_opts = GatewayOptions(file_backed=opts.file_backed)
    gateway = _open_gateway_from_context(ctx, gateway_opts)
    ingest_repo(
        gateway,
        cfg=RepoScanConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit),
    )
    if opts.include_typing:
        ingest_typing_signals(
            gateway,
            cfg=TypingIngestConfig.from_paths(repo_root=repo_root, repo=repo, commit=commit),
            runner=runner,
        )
    if opts.include_coverage:
        ingest_coverage_lines(
            gateway,
            cfg=CoverageIngestConfig.from_paths(
                repo_root=repo_root,
                repo=repo,
                commit=commit,
                coverage_file=coverage_file,
            ),
            runner=runner,
        )
    if opts.build_graph_metrics:
        _seed_cfg_dfg_for_metrics(gateway, rel_path="pkg/mod.py")
        compute_cfg_metrics(gateway, repo=repo, commit=commit)
        compute_dfg_metrics(gateway, repo=repo, commit=commit)

    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=ctx.build_dir,
        db_path=ctx.db_path,
        document_output_dir=ctx.document_output_dir,
        coverage_file=coverage_file,
        gateway=gateway,
        runner=runner,
    )


def provision_gateway_with_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: GatewayOptions | None = None,
) -> ProvisionedGateway:
    """
    Open a gateway anchored to repo paths without running ingestion.

    Useful when tests need to seed custom rows (including invalid ones) but want
    the canonical schemas applied.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with filesystem context.
    """
    opts = options or GatewayOptions()
    repo_root.mkdir(parents=True, exist_ok=True)
    ctx = make_repo_context(repo_root, repo=repo, commit=commit, db_path=opts.db_path)
    coverage_file = repo_root / ".coverage"
    coverage_file.touch()
    runner = _make_runner(repo_root, [])
    gateway = _open_gateway_from_context(ctx, opts)
    if opts.apply_schema and (opts.ensure_views or opts.strict_schema):
        apply_all_schemas(gateway.con)
    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=ctx.build_dir,
        db_path=ctx.db_path,
        document_output_dir=ctx.document_output_dir,
        coverage_file=coverage_file,
        gateway=gateway,
        runner=runner,
    )


def provision_docs_export_ready(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    db_path: Path | None = None,
    file_backed: bool = True,
) -> ProvisionedGateway:
    """Provision a gateway with minimal data for docs export smoke/validation tests.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway populated with docs export seeds.
    """
    ctx = provision_gateway_with_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=GatewayOptions(
            db_path=db_path,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            file_backed=file_backed,
        ),
    )
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit)
    return ctx


def seed_docs_export_invalid_profile(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    null_commit: bool = True,
    drop_commit_column: bool = False,
) -> None:
    """
    Seed minimal docs export data and flip required fields to trigger validation failures.

    Parameters
    ----------
    gateway
        Gateway to mutate.
    repo
        Repository identifier.
    commit
        Commit hash.
    null_commit
        When True, sets commit column in function_profile to NULL.
    drop_commit_column
        When True, removes the commit column from function_profile to induce
        schema validation failures.

    Raises
    ------
    ValueError
        If invoked against a strict gateway; use ``loose_gateway`` instead.
    """
    if getattr(gateway, "config", None) is not None and gateway.config.validate_schema:
        message = (
            "seed_docs_export_invalid_profile requires a non-strict gateway (use loose_gateway)."
        )
        raise ValueError(message)
    seed_docs_export_minimal(gateway, repo=repo, commit=commit)
    con = gateway.con
    con.execute("DROP TABLE IF EXISTS analytics.function_profile")
    commit_value = None if null_commit else commit
    if drop_commit_column:
        con.execute(
            """
            CREATE TABLE analytics.function_profile (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                rel_path TEXT,
                module TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_profile (function_goid_h128, urn, repo, rel_path, module)
            VALUES (1, 'urn:foo', ?, 'foo.py', 'pkg.foo')
            """,
            [repo],
        )
    else:
        con.execute(
            """
            CREATE TABLE analytics.function_profile (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                commit TEXT,
                rel_path TEXT,
                module TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_profile (
                function_goid_h128, urn, repo, commit, rel_path, module
            )
            VALUES (1, 'urn:foo', ?, ?, 'foo.py', 'pkg.foo')
            """,
            [repo, commit_value],
        )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.graph_validation (
            repo TEXT,
            commit TEXT,
            issue TEXT,
            severity TEXT,
            rel_path TEXT,
            detail TEXT,
            metadata JSON,
            created_at TIMESTAMP
        )
        """
    )
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, '{}', '{}', CURRENT_TIMESTAMP)
        """,
        [repo, commit],
    )


def seed_profile_data(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_path: str,
    module: str,
) -> None:
    """Seed profile-related tables with realistic rows for analytics tests."""
    con = gateway.con
    now = utcnow()

    con.execute(
        "DELETE FROM analytics.typedness WHERE path = ? AND repo = ? AND commit = ?",
        [rel_path, repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.static_diagnostics WHERE rel_path = ? AND repo = ? AND commit = ?",
        [rel_path, repo, commit],
    )
    con.execute("DELETE FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit])
    insert_modules(
        gateway,
        [
            ModuleRow(
                module=module,
                path=rel_path,
                repo=repo,
                commit=commit,
                tags='["server"]',
                owners='["team@example.com"]',
            )
        ],
    )

    insert_ast_metrics(
        gateway,
        [
            AstMetricsRow(
                rel_path=rel_path,
                node_count=10,
                function_count=1,
                class_count=0,
                avg_depth=1.0,
                max_depth=1,
                complexity=2.0,
                generated_at=now,
            )
        ],
    )
    insert_hotspots(
        gateway,
        [
            HotspotRow(
                rel_path=rel_path,
                commit_count=1,
                author_count=1,
                lines_added=5,
                lines_deleted=1,
                complexity=2.0,
                score=0.5,
            )
        ],
    )
    insert_typedness(
        gateway,
        [
            TypednessRow(
                repo=repo,
                commit=commit,
                path=rel_path,
                type_error_count=1,
                annotation_ratio='{"params": 0.5}',
                untyped_defs=0,
                overlay_needed=False,
            )
        ],
    )
    insert_static_diagnostics(
        gateway,
        [
            StaticDiagnosticsRow(
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                pyrefly_errors=1,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=1,
                has_errors=True,
            )
        ],
    )
    insert_docstrings(
        gateway,
        [
            DocstringRow(
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                module=module,
                qualname="pkg.mod.func",
                kind="function",
                lineno=1,
                end_lineno=2,
                raw_docstring="Doc",
                style="auto",
                short_desc="Short doc",
                long_desc="Longer doc",
                params_json="[]",
                returns_json='{"return": "int"}',
                raises_json="[]",
                examples_json="[]",
                created_at=now,
            )
        ],
    )
    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
                loc=4,
                logical_loc=3,
                cyclomatic_complexity=2,
                complexity_bucket="medium",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.5,
                file_typed_ratio=0.5,
                static_error_count=1,
                has_static_errors=True,
                executable_lines=4,
                covered_lines=2,
                coverage_ratio=0.5,
                tested=True,
                test_count=1,
                failing_test_count=1,
                last_test_status="some_failing",
                risk_score=0.9,
                risk_level="high",
                tags='["server"]',
                owners='["team@example.com"]',
                created_at=now,
            )
        ],
    )
    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
                start_line=1,
                end_line=2,
                loc=4,
                logical_loc=3,
                param_count=2,
                positional_params=1,
                keyword_only_params=1,
                has_varargs=True,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=2,
                max_nesting_depth=1,
                stmt_count=2,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="medium",
                created_at=now,
            )
        ],
    )
    insert_function_types(
        gateway,
        [
            FunctionTypesRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
                start_line=1,
                end_line=2,
                total_params=2,
                annotated_params=2,
                unannotated_params=0,
                param_typed_ratio=1.0,
                has_return_annotation=True,
                return_type="int",
                return_type_source="annotation",
                type_comment=None,
                param_types_json="[]",
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="analysis",
                created_at=now,
            )
        ],
    )
    insert_coverage_functions(
        gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
                start_line=1,
                end_line=2,
                executable_lines=4,
                covered_lines=2,
                coverage_ratio=0.5,
                tested=True,
                untested_reason="",
                created_at=now,
            )
        ],
    )
    insert_test_catalog(
        gateway,
        [
            TestCatalogRow(
                test_id="pkg/mod.py::test_func",
                test_goid_h128=2,
                urn="goid:demo/repo#python:function:pkg.mod.test_func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                qualname="pkg.mod.test_func",
                kind="function",
                status="failed",
                duration_ms=1500,
                markers="[]",
                parametrized=False,
                flaky=True,
                created_at=now,
            )
        ],
    )
    insert_test_coverage_edges(
        gateway,
        [
            TestCoverageEdgeRow(
                test_id="pkg/mod.py::test_func",
                test_goid_h128=2,
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                qualname="pkg.mod.func",
                covered_lines=2,
                executable_lines=4,
                coverage_ratio=0.5,
                last_status="failed",
                created_at=now,
            )
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                rel_path,
                1,
                1,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                3,
                1,
                rel_path,
                2,
                2,
                "python",
                "direct",
                "global_name",
                1.0,
            ),
        ],
    )
    insert_call_graph_nodes(
        gateway,
        [
            CallGraphNodeRow(
                1,
                "python",
                "function",
                0,
                is_public=True,
                rel_path=rel_path,
            ),
            CallGraphNodeRow(
                2,
                "python",
                "function",
                0,
                is_public=False,
                rel_path=rel_path,
            ),
            CallGraphNodeRow(
                3,
                "python",
                "function",
                0,
                is_public=False,
                rel_path=rel_path,
            ),
        ],
    )
    insert_import_graph_edges(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=module,
                dst_module=module,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
            )
        ],
    )


def _seed_cfg_dfg_for_metrics(
    gateway: StorageGateway,
    *,
    rel_path: str,
) -> None:
    """Seed minimal CFG/DFG rows so compute_cfg/dfg_metrics can run."""
    cfg_blocks = [
        CFGBlockRow(1, 0, "1:block0", "entry", rel_path, 1, 1, "entry", "[]", 0, 1),
        CFGBlockRow(1, 1, "1:block1", "body", rel_path, 2, 3, "body", "[]", 1, 1),
        CFGBlockRow(1, 2, "1:block2", "loop_head", rel_path, 4, 4, "loop_head", "[]", 1, 2),
        CFGBlockRow(1, 3, "1:block3", "unreachable", rel_path, 10, 10, "body", "[]", 0, 0),
        CFGBlockRow(1, 4, "1:block4", "exit", rel_path, 11, 11, "exit", "[]", 1, 0),
    ]
    insert_cfg_blocks(gateway, cfg_blocks)
    cfg_edges = [
        CFGEdgeRow(1, "1:block0", "1:block1", "fallthrough"),
        CFGEdgeRow(1, "1:block1", "1:block2", "loop"),
        CFGEdgeRow(1, "1:block2", "1:block1", "back"),
        CFGEdgeRow(1, "1:block2", "1:block4", "fallthrough"),
    ]
    insert_cfg_edges(gateway, cfg_edges)

    dfg_edges = [
        DFGEdgeRow(
            1,
            "1:block0",
            "1:block1",
            "a",
            "a",
            "data-flow",
            via_phi=False,
            use_kind="data-flow",
        ),
        DFGEdgeRow(
            1,
            "1:block1",
            "1:block2",
            "a",
            "a",
            "phi",
            via_phi=True,
            use_kind="phi",
        ),
        DFGEdgeRow(
            1,
            "1:block1",
            "1:block1",
            "a",
            "a",
            "intra-block",
            via_phi=False,
            use_kind="intra-block",
        ),
    ]
    insert_dfg_edges(gateway, dfg_edges)


def provision_graph_ready_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """
    Provision a repo with graph metrics ready (modules + CFG/DFG metrics seeded).

    Inserts a minimal GOID row for pkg.mod.func so graph tests can bind to it.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with seeded CFG/DFG state.
    """
    opts = options or ProvisionOptions(build_graph_metrics=True)
    ctx = provision_ingested_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=opts.include_typing,
            include_coverage=opts.include_coverage,
            build_graph_metrics=True,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=opts.include_seed_goid,
        ),
    )
    if opts.include_seed_goid:
        con = ctx.gateway.con
        con.execute(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, created_at
            )
            VALUES (
                1, 'urn:pkg.mod:func', ?, ?, 'pkg/mod.py', 'python', 'function',
                'pkg.mod.func', 1, 2, CURRENT_TIMESTAMP
            )
            """,
            [repo, commit],
        )
    return ctx


def seed_callgraph_goids(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    entries: list[tuple[int, str, str, int, int, str]],
) -> None:
    """Insert GOIDs for callgraph tests using gateway helpers."""
    now = utcnow()
    rows = [
        GoidRow(
            goid_h128=goid,
            urn=urn,
            repo=repo,
            commit=commit,
            rel_path=rel_path,
            kind=kind_value,
            qualname=urn.split(":", maxsplit=1)[-1],
            start_line=start_line,
            end_line=end_line,
            created_at=now,
        )
        for goid, urn, rel_path, start_line, end_line, kind_value in entries
    ]
    insert_goids(gateway, rows)


def seed_function_graph_cycle(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_path: str,
) -> None:
    """Seed minimal callgraph nodes/edges to exercise cycle detection."""
    gateway.con.execute(
        """
        DELETE FROM graph.call_graph_edges
        WHERE repo = ? AND commit = ? AND caller_goid_h128 IN (1,2)
        """,
        [repo, commit],
    )
    gateway.con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1, 2)")
    insert_call_graph_nodes(
        gateway,
        [
            CallGraphNodeRow(
                1,
                "python",
                "function",
                0,
                is_public=True,
                rel_path=rel_path,
            ),
            CallGraphNodeRow(
                2,
                "python",
                "function",
                0,
                is_public=False,
                rel_path=rel_path,
            ),
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                rel_path,
                1,
                1,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                rel_path,
                2,
                2,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                2,
                1,
                rel_path,
                3,
                1,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
        ],
    )


def seed_module_graph_inputs(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    module_a: str,
    module_b: str,
) -> None:
    """Seed import/symbol edges for module graph metrics calculations."""
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND module IN (?, ?)",
        [repo, commit, module_a, module_b],
    )
    insert_modules(
        gateway,
        [
            ModuleRow(module=module_a, path="pkg/mod_a.py", repo=repo, commit=commit),
            ModuleRow(module=module_b, path="pkg/mod_b.py", repo=repo, commit=commit),
        ],
    )
    gateway.con.execute(
        """
        DELETE FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ? AND src_module = ? AND dst_module = ?
        """,
        [repo, commit, module_a, module_b],
    )
    insert_import_graph_edges(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=module_a,
                dst_module=module_b,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            )
        ],
    )
    gateway.con.execute(
        """
        DELETE FROM graph.symbol_use_edges
        WHERE symbol = 'sym' AND def_path = ? AND use_path = ?
        """,
        ["pkg/mod_b.py", "pkg/mod_a.py"],
    )
    insert_symbol_use_edges(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path="pkg/mod_b.py",
                use_path="pkg/mod_a.py",
                same_file=False,
                same_module=False,
            )
        ],
    )


def seed_docs_export_minimal(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Seed the minimal rows needed for docs export smoke tests."""
    con = gateway.con
    now = utcnow()
    goid = 1
    apply_all_schemas(con)

    con.execute("DELETE FROM core.repo_map WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.goid_crosswalk WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 = ?", [goid])
    con.execute("DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM graph.cfg_blocks WHERE function_goid_h128 = ?", [goid])
    con.execute(
        "DELETE FROM graph.import_graph_edges WHERE repo = ? AND commit = ?", [repo, commit]
    )
    con.execute("DELETE FROM graph.symbol_use_edges WHERE symbol = 'sym'")
    con.execute("DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute(
        "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    insert_repo_map(
        gateway, [RepoMapRow(repo=repo, commit=commit, modules={"pkg.foo": "foo.py"}, overlays={})]
    )
    insert_modules(
        gateway,
        [
            ModuleRow(
                module="pkg.foo",
                path="foo.py",
                repo=repo,
                commit=commit,
            )
        ],
    )
    insert_goids(
        gateway,
        [
            GoidRow(
                goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                kind="function",
                qualname="pkg.foo:func",
                start_line=1,
                end_line=10,
                created_at=now,
            )
        ],
    )
    insert_goid_crosswalk(
        gateway,
        [
            GoidCrosswalkRow(
                repo=repo,
                commit=commit,
                goid="urn:foo",
                lang="python",
                module_path="pkg.foo",
                file_path="foo.py",
                start_line=1,
                end_line=10,
                scip_symbol="scip-python foo",
                ast_qualname="pkg.foo:func",
                cst_node_id=None,
                chunk_id=None,
                symbol_id=None,
                updated_at=now,
            )
        ],
    )
    insert_call_graph_nodes(
        gateway,
        [
            CallGraphNodeRow(
                goid,
                "python",
                "function",
                0,
                is_public=True,
                rel_path="foo.py",
            )
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                goid,
                goid,
                "foo.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            )
        ],
    )
    insert_cfg_blocks(
        gateway,
        [CFGBlockRow(goid, 0, f"{goid}:block0", "entry", "foo.py", 1, 1, "entry", "[]", 0, 0)],
    )
    insert_import_graph_edges(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module="pkg.foo",
                dst_module="pkg.bar",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
                module_layer=0,
            )
        ],
    )
    insert_symbol_use_edges(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path="foo.py",
                use_path="foo.py",
                same_file=True,
                same_module=True,
                def_goid_h128=goid,
                use_goid_h128=goid,
            )
        ],
    )
    insert_docstrings(
        gateway,
        [
            DocstringRow(
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                module="pkg.foo",
                qualname="pkg.foo:func",
                kind="function",
                lineno=1,
                end_lineno=1,
                raw_docstring="demo",
                style="auto",
                short_desc="demo",
                long_desc="",
                params_json="[]",
                returns_json='{"type": "str"}',
                raises_json="[]",
                examples_json="[]",
                created_at=now,
            )
        ],
    )
    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
                start_line=1,
                end_line=10,
                loc=10,
                logical_loc=10,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=1,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            )
        ],
    )
    insert_function_types(
        gateway,
        [
            FunctionTypesRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
                start_line=1,
                end_line=10,
                total_params=1,
                annotated_params=1,
                unannotated_params=0,
                param_typed_ratio=1.0,
                has_return_annotation=True,
                return_type="str",
                return_type_source="annotation",
                type_comment=None,
                param_types_json="{}",
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="pyright",
                created_at=now,
            )
        ],
    )
    insert_coverage_functions(
        gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
                start_line=1,
                end_line=10,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                untested_reason=None,
                created_at=now,
            )
        ],
    )
    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
                loc=10,
                logical_loc=10,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="pyright",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="passed",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            )
        ],
    )
    insert_test_catalog(
        gateway,
        [
            TestCatalogRow(
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="pkg.foo::test_func",
                status="passed",
                created_at=now,
            )
        ],
    )
    insert_test_coverage_edges(
        gateway,
        [
            TestCoverageEdgeRow(
                test_id="t1",
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="pkg.foo:func",
                covered_lines=1,
                executable_lines=1,
                coverage_ratio=1.0,
                last_status="passed",
                created_at=now,
                test_goid_h128=None,
            )
        ],
    )


def seed_graph_validation_gaps(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Seed rows that trigger graph validation warnings."""
    con = gateway.con
    now = utcnow()
    con.execute(
        """
        INSERT INTO core.ast_nodes (
            path, node_type, name, qualname, lineno, end_lineno, col_offset, end_col_offset,
            parent_qualname, decorators, docstring, hash
        ) VALUES ('pkg/a.py', 'FunctionDef', 'foo', 'pkg.a.foo', 1, 2, 0, 0, 'pkg.a',
                  '[]', NULL, 'h1')
        """
    )
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('pkg.a', 'pkg/a.py', ?, ?, 'python', '[]', '[]')
        """,
        [repo, commit],
    )
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (1, 'urn:pkg.b.caller', ?, ?, 'pkg/b.py', 'python', 'function',
                  'pkg.b.caller', 1, 5, ?)
        """,
        [repo, commit, now],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo, commit, caller_goid_h128, callee_goid_h128, callsite_path, callsite_line,
            callsite_col, language, kind, resolved_via, confidence, evidence_json
        ) VALUES (?, ?, 1, NULL, 'pkg/b.py', 50, 0, 'python', 'unresolved', 'unresolved',
                  0.0, '{}')
        """,
        [repo, commit],
    )


def seed_call_graph_scoping(
    gateway: StorageGateway,
    *,
    now_iso: str,
) -> None:
    """Seed call graph edges across repos/commits for scoping tests."""
    now = datetime.fromisoformat(now_iso)
    con = gateway.con
    con.execute("DELETE FROM graph.call_graph_edges WHERE repo IN ('r1', 'r2')")
    con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1, 2)")
    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE (repo, commit) IN (('r1','c1'),('r2','c2'))"
    )
    con.execute("DELETE FROM core.goids WHERE goid_h128 IN (1, 2)")
    insert_goids(
        gateway,
        [
            GoidRow(
                goid_h128=1,
                urn="urn:1",
                repo="r1",
                commit="c1",
                rel_path="a.py",
                kind="function",
                qualname="a.f",
                start_line=1,
                end_line=2,
                created_at=now,
            ),
            GoidRow(
                goid_h128=2,
                urn="urn:2",
                repo="r2",
                commit="c2",
                rel_path="b.py",
                kind="function",
                qualname="b.f",
                start_line=1,
                end_line=2,
                created_at=now,
            ),
        ],
    )
    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                urn="urn:1",
                repo="r1",
                commit="c1",
                rel_path="a.py",
                language="python",
                kind="function",
                qualname="a.f",
                loc=0,
                logical_loc=0,
                cyclomatic_complexity=0,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=0.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=0,
                covered_lines=0,
                coverage_ratio=0.0,
                tested=False,
                test_count=0,
                failing_test_count=0,
                last_test_status="",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
            RiskFactorRow(
                function_goid_h128=2,
                urn="urn:2",
                repo="r2",
                commit="c2",
                rel_path="b.py",
                language="python",
                kind="function",
                qualname="b.f",
                loc=0,
                logical_loc=0,
                cyclomatic_complexity=0,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=0.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=0,
                covered_lines=0,
                coverage_ratio=0.0,
                tested=False,
                test_count=0,
                failing_test_count=0,
                last_test_status="",
                risk_score=0.9,
                risk_level="high",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                "r1",
                "c1",
                1,
                None,
                "a.py",
                1,
                0,
                "python",
                "direct",
                "local",
                1.0,
            ),
            CallGraphEdgeRow(
                "r2",
                "c2",
                2,
                None,
                "b.py",
                2,
                0,
                "python",
                "direct",
                "local",
                1.0,
            ),
        ],
    )


def graph_metrics_ready_gateway(  # noqa: PLR0913
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    graph_cfg: GraphMetricsConfig | None = None,
    include_symbol_edges: bool = True,
    file_backed: bool = False,
    db_path: Path | None = None,
    run_metrics: bool = True,
    build_callgraph_enabled: bool = True,
) -> ProvisionedGateway:
    """
    Provision a gateway with callgraph/import data and run graph metrics end-to-end.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with graph metrics populated.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    _graph_metrics_repo(repo_root)
    ctx = provision_existing_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=False,
            include_coverage=False,
            build_graph_metrics=False,
            file_backed=file_backed,
            db_path=db_path,
            include_seed_goid=False,
        ),
    )
    gateway = ctx.gateway
    if run_metrics:
        gateway.con.execute(
            "DELETE FROM core.goids WHERE repo = ? AND commit = ? AND goid_h128 IN (1, 2)",
            [repo, commit],
        )
        insert_goids(
            gateway,
            [
                GoidRow(
                    goid_h128=1,
                    urn="urn:pkg.mod_a.a",
                    repo=repo,
                    commit=commit,
                    rel_path="pkg/mod_a.py",
                    kind="function",
                    qualname="pkg.mod_a.a",
                    start_line=1,
                    end_line=4,
                    created_at=utcnow(),
                ),
                GoidRow(
                    goid_h128=2,
                    urn="urn:pkg.mod_b.b",
                    repo=repo,
                    commit=commit,
                    rel_path="pkg/mod_b.py",
                    kind="function",
                    qualname="pkg.mod_b.b",
                    start_line=1,
                    end_line=3,
                    created_at=utcnow(),
                ),
            ],
        )
        insert_call_graph_nodes(
            gateway,
            [
                CallGraphNodeRow(
                    1,
                    "python",
                    "function",
                    0,
                    is_public=True,
                    rel_path="pkg/mod_a.py",
                ),
                CallGraphNodeRow(
                    2,
                    "python",
                    "function",
                    0,
                    is_public=True,
                    rel_path="pkg/mod_b.py",
                ),
            ],
        )
        insert_call_graph_edges(
            gateway,
            [
                CallGraphEdgeRow(
                    repo,
                    commit,
                    1,
                    2,
                    "pkg/mod_a.py",
                    3,
                    0,
                    "python",
                    "direct",
                    "local_name",
                    1.0,
                )
            ],
        )
    if build_callgraph_enabled:
        cfg = CallGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
        build_call_graph(gateway, cfg)
    if include_symbol_edges:
        insert_symbol_use_edges(
            gateway,
            [
                SymbolUseEdgeRow(
                    symbol="sym",
                    def_path="pkg/mod_b.py",
                    use_path="pkg/mod_a.py",
                    same_file=False,
                    same_module=False,
                )
            ],
        )
    if run_metrics:
        cfg = graph_cfg or GraphMetricsConfig.from_paths(repo=repo, commit=commit)
        compute_graph_metrics(gateway, cfg)
    return ctx


def docs_views_ready_gateway(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    file_backed: bool = False,
    db_path: Path | None = None,
) -> ProvisionedGateway:
    """
    Provision a gateway ready for docs views/tests with realistic seeds.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with repo_map/modules/goids, coverage, and risk factors.
    """
    ctx = provision_ingested_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=True,
            include_coverage=True,
            build_graph_metrics=True,
            file_backed=file_backed,
            db_path=db_path,
            include_seed_goid=True,
        ),
    )
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit)
    return ctx


def build_callgraph_fixture_repo(  # noqa: PLR0913
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    file_backed: bool = False,
    db_path: Path | None = None,
    goid_entries: list[tuple[int, str, str, int, int, str]] | None = None,
) -> ProvisionedGateway:
    """
    Create the alias/relative-import callgraph repo and build callgraph via production APIs.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway after callgraph build.
    """
    write_callgraph_alias_repo(repo_root)
    ctx = provision_existing_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=False,
            include_coverage=False,
            build_graph_metrics=False,
            file_backed=file_backed,
            db_path=db_path,
            include_seed_goid=False,
        ),
    )
    gateway = ctx.gateway
    if goid_entries:
        seed_callgraph_goids(gateway, repo=repo, commit=commit, entries=goid_entries)
    cfg = CallGraphConfig.from_paths(repo=repo, commit=commit, repo_root=repo_root)
    build_call_graph(gateway, cfg)
    return ctx


def seed_mcp_backend(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Seed minimal data for MCP backend tests."""
    con = gateway.con
    now = utcnow()
    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.function_validation WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    insert_risk_factors(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="foo",
                loc=1,
                logical_loc=1,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="passed",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            )
        ],
    )
    insert_function_metrics(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=1,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="foo",
                start_line=1,
                end_line=1,
                loc=1,
                logical_loc=1,
                param_count=0,
                positional_params=0,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=1,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            )
        ],
    )
    insert_function_validation(
        gateway,
        [
            FunctionValidationRow(
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="foo",
                issue="span_not_found",
                detail="Span 1-2",
                created_at=now,
            )
        ],
    )
    insert_call_graph_edges(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                "foo.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                3,
                1,
                "bar.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
        ],
    )
    insert_test_catalog(
        gateway,
        [
            TestCatalogRow(
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path="tests/t.py",
                qualname="tests.t",
                status="passed",
                created_at=now,
            )
        ],
    )
    insert_test_coverage_edges(
        gateway,
        [
            TestCoverageEdgeRow(
                test_id="t1",
                function_goid_h128=1,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="foo",
                covered_lines=1,
                executable_lines=1,
                coverage_ratio=1.0,
                last_status="passed",
                created_at=now,
            )
        ],
    )
