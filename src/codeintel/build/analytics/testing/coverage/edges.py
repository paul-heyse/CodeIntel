"""Build test_coverage_edges from coverage contexts and test catalog."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from coverage import Coverage
from coverage.exceptions import CoverageException

from codeintel.core.catalog import CatalogService
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsTestCoverageEdgesRow as TestCoverageEdgeRow,
)
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.query_results import iter_tuples_from_arrow_reader
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from coverage import CoverageData

    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

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
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> dict[str, list[FunctionRow]]:
    """Build mapping of file paths to function metadata.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    catalog_provider
        Optional pre-loaded catalog provider.

    Returns
    -------
    dict[str, list[FunctionRow]]
        Functions keyed by relative file path.
    """
    provider = catalog_provider or CatalogService.from_db(
        gateway, repo=snapshot.repo, commit=snapshot.commit
    )
    catalog = provider.catalog()
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


def _backfill_test_goids(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> tuple[dict[str, int], dict[str, str]]:
    """Try to map test_catalog entries to GOIDs and update catalog rows.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    con = gateway.con
    tests_reader = con.execute(
        """
        SELECT test_id, rel_path, qualname
        FROM analytics.test_catalog
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    tests_iter = iter_tuples_from_arrow_reader(tests_reader)
    first_test = next(tests_iter, None)
    if first_test is None:
        return {}, {}

    goid_reader = gateway.execute(
        """
        SELECT goid_h128, urn, rel_path, qualname
        FROM core.goids
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    goid_iter = iter_tuples_from_arrow_reader(goid_reader)
    first_goid = next(goid_iter, None)
    if first_goid is None:
        return {}, {}

    goid_index: dict[tuple[str, str], tuple[int, str]] = {
        (str(rel_path).replace("\\", "/"), str(qualname)): (int(goid_h128), str(urn))
        for goid_h128, urn, rel_path, qualname in chain([first_goid], goid_iter)
    }

    goid_by_id: dict[str, int] = {}
    urn_by_id: dict[str, str] = {}
    updates: list[tuple[int, str, str, str]] = []

    for test_id_raw, rel_path_raw, qualname in chain([first_test], tests_iter):
        normalized = None if qualname is None else str(qualname).replace("::", ".")
        if normalized is None:
            continue
        test_id = str(test_id_raw)
        rel_path = str(rel_path_raw).replace("\\", "/")
        hit = goid_index.get((rel_path, normalized))
        if hit:
            goid, urn = hit
            goid_by_id[test_id] = goid
            urn_by_id[test_id] = urn
            updates.append((goid, urn, test_id, rel_path))

    if updates:
        backend = gateway.policy
        backend.ensure_table("analytics.test_catalog")
        rows = [
            (test_id, goid, urn, snapshot.repo, snapshot.commit, rel_path)
            for goid, urn, test_id, rel_path in updates
        ]
        backend.upsert(
            "analytics.test_catalog",
            rows,
            columns=[
                "test_id",
                "test_goid_h128",
                "urn",
                "repo",
                "commit",
                "rel_path",
            ],
            upsert=UpsertSpec(
                conflict_columns=("test_id",),
                update_columns=("test_goid_h128", "urn"),
            ),
        )

    return goid_by_id, urn_by_id


def backfill_test_goids_for_catalog(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> tuple[dict[str, int], dict[str, str]]:
    """Backfill GOIDs and URNs for tests in test_catalog.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.

    Returns
    -------
    tuple[dict[str, int], dict[str, str]]
        Mappings from test_id to GOID h128 and URN.
    """
    return _backfill_test_goids(gateway, snapshot)


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


def _test_status_and_meta(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> tuple[dict[str, str], dict[str, tuple[int | None, str | None]]]:
    """Load test status and metadata for the snapshot.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.

    Returns
    -------
    tuple[dict[str, str], dict[str, tuple[int | None, str | None]]]
        Status by test ID and metadata (goid, urn) by test ID.
    """
    status_reader = gateway.execute(
        """
        SELECT test_id, status
        FROM analytics.test_catalog
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)
    status_by_test = {
        row[0]: row[1] for row in iter_tuples_from_arrow_reader(status_reader)
    }
    test_goid_by_id, test_urn_by_id = _backfill_test_goids(gateway, snapshot)
    test_meta_by_id = {
        test_id: (test_goid_by_id.get(test_id), test_urn_by_id.get(test_id))
        for test_id in set(status_by_test.keys())
        | set(test_goid_by_id.keys())
        | set(test_urn_by_id.keys())
    }
    return status_by_test, test_meta_by_id


def build_test_coverage_edges_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: TestCoverageOptions | None = None,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> list[TestCoverageEdgeRow]:
    """Populate analytics.test_coverage_edges by combining coverage contexts with GOIDs.

    This expects coverage.py to have been run with dynamic contexts enabled
    (e.g., dynamic_context = test_function) so contexts_by_lineno returns
    pytest nodeids.

    Parameters
    ----------
    gateway
        Storage gateway for DuckDB.
    snapshot
        Snapshot reference with repo, commit, and repo_root.
    options
        Optional coverage configuration options.
    catalog_provider
        Optional pre-loaded catalog provider.

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

    funcs_by_path = _functions_by_path(gateway, snapshot, catalog_provider=catalog_provider)
    if not funcs_by_path:
        log.info("No functions found; skipping test coverage edges")
        return []

    status_by_test, test_meta_by_id = _test_status_and_meta(gateway, snapshot)

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
