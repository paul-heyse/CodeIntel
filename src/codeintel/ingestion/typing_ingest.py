"""Ingest typedness ratios and static diagnostics for Python files."""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from codeintel.config import TypingIngestStepConfig
from codeintel.config.dataset_contract import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord, iter_modules, run_batch
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    default_code_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map

log = logging.getLogger(__name__)


@dataclass
class AnnotationInfo:
    """
    Ratio and count statistics summarizing annotations in a file.

    Parameters
    ----------
    params_ratio : float
        Fraction of parameters (excluding self/cls) with annotations.
    returns_ratio : float
        Fraction of functions with return annotations.
    untyped_defs : int
        Count of function definitions missing full annotations.
    """

    params_ratio: float
    returns_ratio: float
    untyped_defs: int


def _compute_annotation_info_for_file(path: Path) -> AnnotationInfo | None:
    try:
        source = path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    total_params = 0
    annotated_params = 0
    func_count = 0
    return_annotated = 0
    untyped_defs = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1

            params = []
            # Python 3.8+ supports posonlyargs; older code will ignore.
            posonly = getattr(node.args, "posonlyargs", [])
            params.extend(posonly)
            params.extend(node.args.args)
            params.extend(node.args.kwonlyargs)

            for arg in params:
                if arg.arg in {"self", "cls"}:
                    continue
                total_params += 1
                if arg.annotation is not None:
                    annotated_params += 1

            has_return = node.returns is not None
            if has_return:
                return_annotated += 1

            fully_typed = (
                all(arg.annotation is not None for arg in params if arg.arg not in {"self", "cls"})
                and has_return
            )
            if not fully_typed:
                untyped_defs += 1

    params_ratio = annotated_params / total_params if total_params else 1.0
    returns_ratio = return_annotated / func_count if func_count else 1.0

    return AnnotationInfo(
        params_ratio=params_ratio,
        returns_ratio=returns_ratio,
        untyped_defs=untyped_defs,
    )


async def _collect_error_maps(
    repo_root: Path,
    service: ToolService,
) -> dict[str, dict[str, int]]:
    pyrefly_map, pyright_map, ruff_map = await asyncio.gather(
        service.run_pyrefly(repo_root),
        service.run_pyright(repo_root),
        service.run_ruff(repo_root),
    )
    return {
        "pyrefly": dict(pyrefly_map),
        "pyright": dict(pyright_map),
        "ruff": dict(ruff_map),
    }


@dataclass
class TypingIngestResult:
    """Bundle typedness and diagnostic rows for a module."""

    typedness: TypednessRow | None
    diagnostics: StaticDiagnosticRow | None


def _delete_existing_typing_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
) -> None:
    """
    Delete existing typedness and diagnostic rows for selected paths.

    When rel_paths is empty, delete all rows for repo@commit.
    """
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM analytics.typedness
            WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM analytics.static_diagnostics
            WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        return

    run_batch(
        gateway,
        "analytics.typedness",
        [],
        delete_params=[repo, commit],
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [],
        delete_params=[repo, commit],
    )


@dataclass
class TypingIngestOps(IncrementalIngestOps[TypingIngestResult]):
    """
    Implement incremental ingest operations for typedness and diagnostics datasets.

    Tool outputs are computed once per repo, while AST parsing is limited to changed files.
    """

    cfg: TypingIngestStepConfig
    repo_root: Path
    error_maps: dict[str, dict[str, int]]
    dataset_name: str = field(init=False, default="analytics.typedness")

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Process only Python source files.

        Returns
        -------
        bool
            True when the module is a Python source file.
        """
        return module.rel_path.endswith(".py")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Delete rows for modules scheduled for removal."""
        _delete_existing_typing_rows(
            gateway,
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_paths=list(rel_paths),
        )

    def process_module(self, module: ModuleRecord) -> Iterable[TypingIngestResult]:
        """
        Compute typedness and diagnostic rows for a single module.

        Returns
        -------
        Iterable[TypingIngestResult]
            Rows summarizing annotations and static diagnostics.
        """
        path = module.file_path
        info = _compute_annotation_info_for_file(path) or AnnotationInfo(
            params_ratio=0.0,
            returns_ratio=0.0,
            untyped_defs=0,
        )

        pf_errors = self.error_maps["pyrefly"].get(module.rel_path, 0)
        py_errors = self.error_maps["pyright"].get(module.rel_path, 0)
        ruff_errors = self.error_maps["ruff"].get(module.rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness = TypednessRow(
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            path=module.rel_path,
            type_error_count=total_errors,
            annotation_ratio={
                "params": info.params_ratio,
                "returns": info.returns_ratio,
            },
            untyped_defs=info.untyped_defs,
            overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
        )

        diagnostics = StaticDiagnosticRow(
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_path=module.rel_path,
            pyrefly_errors=pf_errors,
            pyright_errors=py_errors,
            ruff_errors=ruff_errors,
            total_errors=total_errors,
            has_errors=total_errors > 0,
        )

        return [
            TypingIngestResult(
                typedness=typedness,
                diagnostics=diagnostics,
            )
        ]

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[TypingIngestResult]) -> None:
        """Insert typedness and diagnostic rows."""
        if not rows:
            return

        rel_paths: set[str] = set()
        typedness_rows: list[TypednessRow] = []
        diag_rows: list[StaticDiagnosticRow] = []

        for result in rows:
            if result.typedness is not None:
                typedness_rows.append(result.typedness)
                rel_paths.add(result.typedness["path"])
            if result.diagnostics is not None:
                diag_rows.append(result.diagnostics)
                rel_paths.add(result.diagnostics["rel_path"])

        if rel_paths:
            _delete_existing_typing_rows(
                gateway,
                repo=self.cfg.repo,
                commit=self.cfg.commit,
                rel_paths=sorted(rel_paths),
            )

        if typedness_rows:
            run_batch(
                gateway,
                "analytics.typedness",
                [typedness_row_to_tuple(row) for row in typedness_rows],
                delete_params=None,
                scope=f"{self.cfg.repo}@{self.cfg.commit}",
            )

        if diag_rows:
            run_batch(
                gateway,
                "analytics.static_diagnostics",
                [static_diagnostic_to_tuple(row) for row in diag_rows],
                delete_params=None,
                scope=f"{self.cfg.repo}@{self.cfg.commit}",
            )


def _ingest_typing_full(
    *,
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    profile: ScanProfile,
    error_maps: dict[str, dict[str, int]],
) -> None:
    """
    Execute the legacy full-scan typing ingest path.

    Parameters
    ----------
    gateway :
        Storage gateway used for persistence.
    cfg :
        Step configuration for the current snapshot.
    profile :
        Scan profile controlling file discovery.
    error_maps :
        Static diagnostic counts keyed by relative path.
    """
    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=log,
    )
    annotation_info: dict[str, AnnotationInfo] = {}
    for record in iter_modules(
        module_map,
        cfg.repo_root,
        logger=log,
        scan_profile=profile,
    ):
        info = _compute_annotation_info_for_file(record.file_path)
        if info is not None:
            annotation_info[record.rel_path] = info

    path_set = (
        set(annotation_info)
        | set(error_maps["pyrefly"])
        | set(error_maps["pyright"])
        | set(error_maps["ruff"])
    )

    typedness_rows: list[TypednessRow] = []
    diag_rows: list[StaticDiagnosticRow] = []
    default_info = AnnotationInfo(params_ratio=0.0, returns_ratio=0.0, untyped_defs=0)
    for rel_path in sorted(path_set):
        info = annotation_info.get(rel_path, default_info)
        pf_errors = error_maps["pyrefly"].get(rel_path, 0)
        py_errors = error_maps["pyright"].get(rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness_rows.append(
            TypednessRow(
                repo=cfg.repo,
                commit=cfg.commit,
                path=rel_path,
                type_error_count=total_errors,
                annotation_ratio={
                    "params": info.params_ratio,
                    "returns": info.returns_ratio,
                },
                untyped_defs=info.untyped_defs,
                overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
            )
        )

        diag_rows.append(
            StaticDiagnosticRow(
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path,
                pyrefly_errors=pf_errors,
                pyright_errors=py_errors,
                ruff_errors=error_maps["ruff"].get(rel_path, 0),
                total_errors=total_errors,
                has_errors=total_errors > 0,
            )
        )

    run_batch(
        gateway,
        "analytics.typedness",
        [typedness_row_to_tuple(row) for row in typedness_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [static_diagnostic_to_tuple(row) for row in diag_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "Typedness & static diagnostics ingested for %d files in %s@%s",
        len(path_set),
        cfg.repo,
        cfg.commit,
    )


def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Populate per-file typedness and static diagnostics.

      - analytics.typedness
      - analytics.static_diagnostics

    Notes
    -----
      * Pyrefly drives static error counts; annotation_ratio is computed from Python AST
        (params & returns).
    tracker :
        Optional change tracker enabling incremental ingestion.
    """
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))
    service = tool_service
    if service is None:
        tools_config: ToolsConfig
        if isinstance(cfg.tool_runner, ToolRunner):
            tools_config = cfg.tool_runner.tools_config
            shared_runner = cfg.tool_runner
        else:
            tools_config = ToolsConfig.model_validate({})
            shared_runner = ToolRunner(
                tools_config=tools_config, cache_dir=repo_root / "build" / ".tool_cache"
            )
        service = ToolService(shared_runner, tools_config)

    error_maps = asyncio.run(_collect_error_maps(repo_root, service))

    if tracker is not None:
        ops = TypingIngestOps(
            cfg=cfg,
            repo_root=repo_root,
            error_maps=error_maps,
        )
        run_incremental_ingest(tracker, ops)
        log.info(
            "Typedness & static diagnostics ingested incrementally for %s@%s",
            cfg.repo,
            cfg.commit,
        )
        return

    _ingest_typing_full(
        gateway=gateway,
        cfg=cfg,
        profile=profile,
        error_maps=error_maps,
    )

    return
