"""Build test_coverage_edges from coverage contexts and test catalog."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import polars as pl
from coverage import Coverage
from coverage.exceptions import CoverageException

from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsTestCoverageEdgesRow as TestCoverageEdgeRow,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from coverage import CoverageData

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestCoverageOptions:
    """Configuration options for test coverage edge computation.

    Parameters
    ----------
    coverage_file
        Optional explicit path to coverage file; defaults to repo_root/.coverage.
    coverage_loader
        Optional callable to load coverage data; signature is
        ``(snapshot: SnapshotRef, coverage_file: Path | None) -> Coverage | None``.
    """

    coverage_file: Path | None = None
    coverage_loader: Callable[[SnapshotRef, Path | None], Coverage | None] | None = None


@dataclass
class EdgeContext:
    """Shared context for building test coverage edges."""

    status_by_test: dict[str, str]
    repo: str
    commit: str
    now: datetime
    test_meta_by_id: dict[str, tuple[int | None, str | None]]


class FunctionRow(TypedDict):
    """Minimal function metadata used for coverage edge construction."""

    goid_h128: int
    urn: str
    rel_path: str
    qualname: str
    start_line: int
    end_line: int | None


def _load_coverage_data(
    snapshot: SnapshotRef,
    coverage_file: Path | None = None,
) -> Coverage | None:
    """Load coverage data from file.

    Parameters
    ----------
    snapshot
        Snapshot reference for repository context.
    coverage_file
        Optional explicit coverage file path.

    Returns
    -------
    Coverage | None
        Coverage object when file exists, otherwise None.
    """
    coverage_path = coverage_file or (snapshot.repo_root / ".coverage")
    if not coverage_path.is_file():
        log.warning("Coverage file %s not found; skipping test coverage edges", coverage_path)
        return None

    cov = Coverage(data_file=str(coverage_path))
    cov.load()
    return cov


def load_coverage_data(
    snapshot: SnapshotRef,
    coverage_file: Path | None = None,
) -> Coverage | None:
    """
    Public wrapper for loading coverage data with logging on missing files.

    Parameters
    ----------
    snapshot
        Snapshot reference for repository context.
    coverage_file
        Optional explicit coverage file path.

    Returns
    -------
    Coverage | None
        Coverage object when file exists, otherwise None.
    """
    return _load_coverage_data(snapshot, coverage_file)


__all__ = [
    "EdgeContext",
    "FunctionRow",
    "TestCoverageOptions",
    "backfill_test_goids_for_catalog",
    "build_edges_for_file_for_tests",
    "build_test_coverage_edges_rows",
    "load_coverage_data",
]


def _functions_by_path(
    _snapshot: SnapshotRef,
    *,
    catalog_provider: FunctionCatalogProvider,
) -> dict[str, list[FunctionRow]]:
    """Build mapping of file paths to function metadata.

    Parameters
    ----------
    _snapshot
        Snapshot reference (unused).
    catalog_provider
        Pre-loaded function catalog provider.

    Returns
    -------
    dict[str, list[FunctionRow]]
        Functions keyed by relative file path.
    """
    catalog = catalog_provider.catalog()
    if not catalog.function_spans:
        return {}

    funcs_by_path: dict[str, list[FunctionRow]] = {}
    for span in catalog.function_spans:
        funcs_by_path.setdefault(span.rel_path, []).append(
            FunctionRow(
                goid_h128=span.goid,
                urn=catalog.urn_for_goid(span.goid) or "",
                rel_path=span.rel_path,
                qualname=span.qualname,
                start_line=span.start_line,
                end_line=span.end_line,
            )
        )
    return funcs_by_path


def _test_catalog_rows_from_frame(
    frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> list[tuple[object, ...]]:
    if frame is None or frame.is_empty():
        return []
    filtered = _filter_frame_by_snapshot(frame, repo=repo, commit=commit)
    return [
        (row.get("test_id"), row.get("rel_path"), row.get("qualname"))
        for row in filtered.iter_rows(named=True)
    ]


def _goid_rows_from_frame(
    frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> list[tuple[object, ...]]:
    if frame is None or frame.is_empty():
        return []
    filtered = _filter_frame_by_snapshot(frame, repo=repo, commit=commit)
    return [
        (
            row.get("goid_h128"),
            row.get("urn"),
            row.get("rel_path"),
            row.get("qualname"),
        )
        for row in filtered.iter_rows(named=True)
    ]


def _goid_index(
    rows: Iterable[tuple[object, ...]],
) -> dict[tuple[str, str], tuple[int, str]]:
    index: dict[tuple[str, str], tuple[int, str]] = {}
    for goid_h128, urn, rel_path, qualname in rows:
        goid = normalize_decimal_id(goid_h128)
        if goid is None:
            continue
        rel_path_value = str(rel_path).replace("\\", "/")
        qualname_value = str(qualname)
        index[rel_path_value, qualname_value] = (goid, str(urn))
    return index


def _collect_test_goid_updates(
    rows: Iterable[tuple[object, ...]],
    *,
    index: Mapping[tuple[str, str], tuple[int, str]],
) -> tuple[dict[str, int], dict[str, str], list[tuple[int, str, str, str]]]:
    goid_by_id: dict[str, int] = {}
    urn_by_id: dict[str, str] = {}
    updates: list[tuple[int, str, str, str]] = []
    for test_id_raw, rel_path_raw, qualname in rows:
        normalized = None if qualname is None else str(qualname).replace("::", ".")
        if normalized is None:
            continue
        test_id = str(test_id_raw)
        rel_path = str(rel_path_raw).replace("\\", "/")
        hit = index.get((rel_path, normalized))
        if hit:
            goid_by_id[test_id] = hit[0]
            urn_by_id[test_id] = hit[1]
            updates.append((hit[0], hit[1], test_id, rel_path))
    return goid_by_id, urn_by_id, updates


def _backfill_test_goids_from_frames(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, int], dict[str, str]]:
    """Try to map test_catalog entries to GOIDs and update catalog rows.

    Parameters
    ----------
    test_catalog_frame
        Test catalog rows for the snapshot.
    goids_frame
        GOID rows for the snapshot.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    tests_rows = _test_catalog_rows_from_frame(
        test_catalog_frame,
        repo=repo,
        commit=commit,
    )
    if not tests_rows:
        return {}, {}

    goid_rows = _goid_rows_from_frame(goids_frame, repo=repo, commit=commit)
    if not goid_rows:
        return {}, {}

    index = _goid_index(goid_rows)
    goid_by_id, urn_by_id, updates = _collect_test_goid_updates(tests_rows, index=index)
    _ = updates
    return goid_by_id, urn_by_id


def backfill_test_goids_for_catalog(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, int], dict[str, str]]:
    """Backfill GOIDs and URNs for tests in test_catalog.

    Parameters
    ----------
    test_catalog_frame
        Test catalog rows for the snapshot.
    goids_frame
        GOID rows for the snapshot.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    return _backfill_test_goids_from_frames(
        test_catalog_frame,
        goids_frame,
        repo=repo,
        commit=commit,
    )


def build_edges_for_file_for_tests(
    file_funcs: list[FunctionRow],
    statements_set: set[int],
    contexts_by_lineno: dict[int, set[str]],
    rel_path: str,
    ctx: EdgeContext,
) -> list[TestCoverageEdgeRow]:
    """Exposed wrapper around `_edges_for_file` for unit testing.

    Parameters
    ----------
    file_funcs
        Functions within the file (dicts with GOID metadata).
    statements_set
        Executable statement lines for the file.
    contexts_by_lineno
        Mapping of line numbers to coverage contexts (pytest nodeids).
    rel_path
        Repo-relative file path.
    ctx
        EdgeContext holding repo metadata and test mappings.

    Returns
    -------
    list[TestCoverageEdgeRow]
        Edge records mirroring analytics.test_coverage_edges schema.
    """
    return _edges_for_file(
        file_funcs=file_funcs,
        statements_set=statements_set,
        contexts_by_lineno=contexts_by_lineno,
        rel_path=rel_path,
        ctx=ctx,
    )


def _file_coverage(
    cov: Coverage,
    data: CoverageData,
    abs_file: Path,
) -> tuple[set[int], dict[int, set[str]]]:
    """Extract executable statements and contexts for a single file.

    Returns
    -------
    tuple[set[int], dict[int, set[str]]]
        Executable statement line numbers and context mappings.
    """
    try:
        _, statements, _, _missing, _executed = cov.analysis2(str(abs_file))
    except CoverageException as exc:
        log.warning("coverage.analysis2 failed for %s: %s", abs_file, exc)
        return set(), {}

    statements_set = set(statements)
    try:
        raw_contexts = data.contexts_by_lineno(str(abs_file)) or {}
        contexts_by_lineno: dict[int, set[str]] = {
            ln: set(ctxs) for ln, ctxs in raw_contexts.items()
        }
    except CoverageException:
        contexts_by_lineno = {}

    return statements_set, contexts_by_lineno


def _edges_for_file(
    file_funcs: list[FunctionRow],
    statements_set: set[int],
    contexts_by_lineno: dict[int, set[str]],
    rel_path: str,
    ctx: EdgeContext,
) -> list[TestCoverageEdgeRow]:
    """Build coverage edges for functions in a file.

    Parameters
    ----------
    file_funcs
        List of functions in the file.
    statements_set
        Set of executable statement line numbers.
    contexts_by_lineno
        Mapping of line numbers to coverage contexts.
    rel_path
        Relative file path.
    ctx
        Edge context with repository and test metadata.

    Returns
    -------
    list[TestCoverageEdgeRow]
        Coverage edge rows for this file.
    """
    edges: list[TestCoverageEdgeRow] = []
    for info in file_funcs:
        start_line = int(info["start_line"])
        end_line = int(info["end_line"]) if info["end_line"] is not None else start_line
        executable_lines = len([ln for ln in statements_set if start_line <= ln <= end_line])
        if executable_lines == 0:
            continue

        covered_by_test: dict[str, int] = defaultdict(int)
        for ln in range(start_line, end_line + 1):
            if ln not in statements_set:
                continue
            contexts = contexts_by_lineno.get(ln) or set()
            for ctx_name in contexts:
                covered_by_test[ctx_name] += 1

        for test_id, covered_lines in covered_by_test.items():
            last_status = ctx.status_by_test.get(test_id, "unknown")
            coverage_ratio = covered_lines / float(executable_lines) if executable_lines else None
            test_goid, test_urn = ctx.test_meta_by_id.get(test_id, (None, None))
            edges.append(
                TestCoverageEdgeRow(
                    test_id=test_id,
                    test_goid_h128=test_goid,
                    function_goid_h128=int(info["goid_h128"]),
                    urn=test_urn or info["urn"],
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=rel_path,
                    qualname=info["qualname"],
                    covered_lines=covered_lines,
                    executable_lines=executable_lines,
                    coverage_ratio=coverage_ratio if coverage_ratio is not None else 0.0,
                    last_status=last_status,
                    created_at=ctx.now,
                )
            )
    return edges


def _test_status_and_meta_from_frames(
    test_catalog_frame: pl.DataFrame | None,
    goids_frame: pl.DataFrame | None,
    *,
    repo: str,
    commit: str,
) -> tuple[dict[str, str], dict[str, tuple[int | None, str | None]]]:
    """Load test status and metadata for the snapshot.

    Parameters
    ----------
    test_catalog_frame
        Test catalog rows for the snapshot.
    goids_frame
        GOID rows for the snapshot.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    tuple[dict[str, str], dict[str, tuple[int | None, str | None]]]
        Status by test ID and metadata (goid, urn) by test ID.
    """
    status_by_test: dict[str, str] = {}
    if test_catalog_frame is not None and not test_catalog_frame.is_empty():
        filtered = _filter_frame_by_snapshot(test_catalog_frame, repo=repo, commit=commit)
        for row in filtered.iter_rows(named=True):
            test_id_raw = row.get("test_id")
            status = row.get("status")
            if test_id_raw is None or status is None:
                continue
            status_by_test[str(test_id_raw)] = str(status)
    test_goid_by_id, test_urn_by_id = _backfill_test_goids_from_frames(
        test_catalog_frame,
        goids_frame,
        repo=repo,
        commit=commit,
    )
    test_meta_by_id = {
        test_id: (test_goid_by_id.get(test_id), test_urn_by_id.get(test_id))
        for test_id in set(status_by_test.keys())
        | set(test_goid_by_id.keys())
        | set(test_urn_by_id.keys())
    }
    return status_by_test, test_meta_by_id


def build_test_coverage_edges_rows(
    snapshot: SnapshotRef,
    *,
    options: TestCoverageOptions | None = None,
    catalog_provider: FunctionCatalogProvider,
    test_catalog_frame: pl.DataFrame | None = None,
    goids_frame: pl.DataFrame | None = None,
) -> list[TestCoverageEdgeRow]:
    """Populate analytics.test_coverage_edges by combining coverage contexts with GOIDs.

    This expects coverage.py to have been run with dynamic contexts enabled
    (e.g., dynamic_context = test_function) so contexts_by_lineno returns
    pytest nodeids.

    Parameters
    ----------
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    options
        Optional coverage configuration options.
    catalog_provider
        Pre-loaded catalog provider.
    test_catalog_frame
        Snapshot test catalog rows.
    goids_frame
        Snapshot GOID rows.

    Returns
    -------
    list[TestCoverageEdgeRow]
        Rows ready for insertion into analytics.test_coverage_edges.
    """
    log.info(
        "Building test_coverage_edges rows for repo=%s commit=%s",
        snapshot.repo,
        snapshot.commit,
    )

    opts = options or TestCoverageOptions()
    if opts.coverage_loader is not None:
        cov = opts.coverage_loader(snapshot, opts.coverage_file)
    else:
        cov = _load_coverage_data(snapshot, opts.coverage_file)
    if cov is None:
        return []

    funcs_by_path = _functions_by_path(snapshot, catalog_provider=catalog_provider)
    if not funcs_by_path:
        log.info("No functions found; skipping test coverage edges")
        return []

    status_by_test, test_meta_by_id = _test_status_and_meta_from_frames(
        test_catalog_frame,
        goids_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )

    edge_ctx = EdgeContext(
        status_by_test=status_by_test,
        repo=snapshot.repo,
        commit=snapshot.commit,
        now=datetime.now(UTC),
        test_meta_by_id=test_meta_by_id,
    )
    data = cov.get_data()
    insert_rows: list[TestCoverageEdgeRow] = []

    for measured in data.measured_files():
        abs_file = Path(measured).resolve()
        try:
            rel_path = normalize_path(abs_file.relative_to(snapshot.repo_root))
        except ValueError:
            continue

        file_funcs = funcs_by_path.get(rel_path)
        if not file_funcs:
            continue

        statements_set, contexts_by_lineno = _file_coverage(cov, data, abs_file)
        if not statements_set:
            continue

        insert_rows.extend(
            _edges_for_file(
                file_funcs=file_funcs,
                statements_set=statements_set,
                contexts_by_lineno=contexts_by_lineno,
                rel_path=rel_path,
                ctx=edge_ctx,
            )
        )

    log.info(
        "test_coverage_edges rows built: %d rows for %s@%s",
        len(insert_rows),
        snapshot.repo,
        snapshot.commit,
    )
    return insert_rows


def _filter_frame_by_snapshot(
    frame: pl.DataFrame,
    *,
    repo: str,
    commit: str,
) -> pl.DataFrame:
    filtered = frame
    if "repo" in filtered.columns:
        filtered = filtered.filter(pl.col("repo") == repo)
    if "commit" in filtered.columns:
        filtered = filtered.filter(pl.col("commit") == commit)
    return filtered
