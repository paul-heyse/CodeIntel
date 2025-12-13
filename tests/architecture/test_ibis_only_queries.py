"""Enforce Ibis-first query patterns and raw SQL isolation.

This test ensures that:
1. Raw SQL execution is isolated to the storage layer
2. Application code uses Ibis expressions for queries
3. DDL operations go through the DuckDBPolicyBackend
4. Repository layer uses only Ibis (no SQL fallbacks)
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Maximum number of violations to display in output
MAX_VIOLATIONS_DISPLAY = 20

# Directories that should NOT contain raw SQL
APPLICATION_DIRS = (
    Path("src/codeintel/analytics"),
    Path("src/codeintel/cli"),
    Path("src/codeintel/docs_export"),
    Path("src/codeintel/mcp"),
    Path("src/codeintel/server"),
    Path("src/codeintel/serving"),
)

# Files explicitly allowed to contain raw SQL patterns
# These are in the storage layer or are query services
ALLOWED_RAW_SQL_FILES = frozenset(
    {
        # Storage layer - DDL and policy backend
        Path("src/codeintel/storage/duckdb_policy_backend.py"),
        Path("src/codeintel/storage/schema/ddl.py"),
        Path("src/codeintel/storage/schema/__init__.py"),
        Path("src/codeintel/storage/macros.py"),
        Path("src/codeintel/storage/helpers/db.py"),
        Path("src/codeintel/storage/sql/primitives.py"),
        # Base repository has deprecated SQL helpers for backward compat
        Path("src/codeintel/storage/repositories/base.py"),
        # Query services - bridge between storage and application
        Path("src/codeintel/server/query_service.py"),
        Path("src/codeintel/server/datasets.py"),
        Path("src/codeintel/mcp/query_service.py"),
        # Export modules that directly use the gateway
        Path("src/codeintel/docs_export/export_jsonl.py"),
        Path("src/codeintel/docs_export/export_parquet.py"),
    }
)

# Analytics files that still use raw SQL (pending Phase 4 migration)
# These files have complex multi-table joins or executemany patterns
# that require careful migration
PENDING_IBIS_MIGRATION_ANALYTICS = frozenset(
    {
        # Profile modules
        Path("src/codeintel/analytics/profiles/functions.py"),
        Path("src/codeintel/analytics/profiles/writer_guard.py"),
        Path("src/codeintel/analytics/profiles/graph_features.py"),
        Path("src/codeintel/analytics/profiles/utils.py"),
        Path("src/codeintel/analytics/profiles/files.py"),
        Path("src/codeintel/analytics/profiles/modules.py"),
        # Testing and coverage modules
        Path("src/codeintel/analytics/testing/coverage/edges.py"),
        Path("src/codeintel/analytics/testing/graph_metrics.py"),
        Path("src/codeintel/analytics/testing/coverage/functions.py"),
        Path("src/codeintel/analytics/testing/profiles/rows.py"),
        Path("src/codeintel/analytics/testing/profiles/builder.py"),
        Path("src/codeintel/analytics/testing/behavioral/tags.py"),
        # CFG/DFG analysis modules
        Path("src/codeintel/analytics/cfg_dfg/dfg_core.py"),
        Path("src/codeintel/analytics/cfg_dfg/cfg_core.py"),
        Path("src/codeintel/analytics/cfg_dfg/materialize.py"),
        # Graph analysis modules
        Path("src/codeintel/analytics/graphs/subsystem_graph_metrics.py"),
        Path("src/codeintel/analytics/graphs/symbol_graph_metrics.py"),
        Path("src/codeintel/analytics/graphs/config_data_flow.py"),
        Path("src/codeintel/analytics/graphs/subsystem_agreement.py"),
        Path("src/codeintel/analytics/graphs/graph_stats.py"),
        Path("src/codeintel/analytics/graphs/config_graph_metrics.py"),
        # Semantic roles
        Path("src/codeintel/analytics/semantic_roles/core.py"),
        # Entrypoints
        Path("src/codeintel/analytics/entrypoints/core.py"),
        # Data models
        Path("src/codeintel/analytics/data_models/core.py"),
        # Subsystems
        Path("src/codeintel/analytics/subsystems/materialize.py"),
        Path("src/codeintel/analytics/subsystems/affinity.py"),
        Path("src/codeintel/analytics/subsystems/risk.py"),
        # Dependencies
        Path("src/codeintel/analytics/dependencies/core.py"),
        # Functions - history/contracts/effects
        Path("src/codeintel/analytics/functions/function_history.py"),
        Path("src/codeintel/analytics/functions/effects.py"),
        Path("src/codeintel/analytics/functions/contracts.py"),
        Path("src/codeintel/analytics/functions/history.py"),
        Path("src/codeintel/analytics/functions/semantic_roles.py"),
        Path("src/codeintel/analytics/functions/function_contracts.py"),
        Path("src/codeintel/analytics/functions/function_effects.py"),
        # History timeseries
        Path("src/codeintel/analytics/history/history_timeseries.py"),
        # Compute modules
        Path("src/codeintel/analytics/compute/coverage/functions.py"),
        Path("src/codeintel/analytics/compute/data_models/usage.py"),
        Path("src/codeintel/analytics/compute/hotspots/metrics.py"),
        # Plugin files
        Path("src/codeintel/analytics/plugins/symbol_graph_metrics/compute.py"),
        Path("src/codeintel/analytics/plugins/subsystem_metrics/graph_metrics.py"),
        Path("src/codeintel/analytics/plugins/subsystem_metrics/agreement.py"),
        Path("src/codeintel/analytics/plugins/tests/graph_metrics.py"),
        Path("src/codeintel/analytics/plugins/data_models/usage.py"),
        Path("src/codeintel/analytics/plugins/cfg_dfg/metrics.py"),
        Path("src/codeintel/analytics/plugins/entrypoints/build.py"),
        Path("src/codeintel/analytics/plugins/tests/behavioral_coverage.py"),
        Path("src/codeintel/analytics/plugins/tests/profile.py"),
        Path("src/codeintel/analytics/plugins/hotspots/build.py"),
        Path("src/codeintel/analytics/plugins/risk/factors.py"),
        # Other
        Path("src/codeintel/analytics/parsing/validation.py"),
        # CLI handlers with history SQL
        Path("src/codeintel/cli/handlers/history.py"),
        # CLI services with test-related SQL examples
        Path("src/codeintel/cli/services/storage.py"),
        # Serving backend
        Path("src/codeintel/serving/backend/dataset_backend.py"),
        Path("src/codeintel/serving/backend/datasets.py"),
    }
)

# Ingestion files that still use raw SQL (pending migration)
PENDING_IBIS_MIGRATION_INGESTION = frozenset(
    {
        # Infrastructure files - some still have safe_sql helpers
        Path("src/codeintel/ingestion/infrastructure/safe_sql.py"),
        # Ingestion adapters with complex storage operations
        Path("src/codeintel/ingestion/adapters/duckdb_storage.py"),
    }
)

# Repository files that should NOT contain raw SQL (Ibis-only)
IBIS_ONLY_REPOSITORY_FILES = (
    Path("src/codeintel/storage/repositories/data_models.py"),
    Path("src/codeintel/storage/repositories/dataflow.py"),
    Path("src/codeintel/storage/repositories/datasets.py"),
    Path("src/codeintel/storage/repositories/functions.py"),
    Path("src/codeintel/storage/repositories/graphs.py"),
    Path("src/codeintel/storage/repositories/modules.py"),
    Path("src/codeintel/storage/repositories/subsystems.py"),
    Path("src/codeintel/storage/repositories/tests.py"),
)

# SQL keywords that indicate raw SQL in application code
SQL_KEYWORDS_PATTERN = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)\s+"
    r"(FROM|INTO|TABLE|VIEW|INDEX|SCHEMA)\b",
    re.IGNORECASE,
)

# Patterns that indicate direct SQL execution (not Ibis expressions)
# Note: We exclude patterns that are clearly Ibis execution (e.g., `expr.execute()`)
EXECUTE_PATTERNS = (
    r"\.con\.execute\s*\(",  # Direct connection execute
    r"\.con\.executemany\s*\(",  # Direct connection executemany
    r"con\.execute\s*\(",  # Local con variable execute
    r"con\.executemany\s*\(",  # Local con variable executemany
    r"con\.sql\s*\(",  # Raw SQL
    r"raw_sql\s*\(",  # Raw SQL helper
    r"gateway\.con\.execute\s*\(",  # Gateway connection execute
    r"gateway\.con\.executemany\s*\(",  # Gateway connection executemany
)


def _should_check_file(path: Path) -> bool:
    """Determine if a file should be checked for SQL patterns.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
    bool
        True if the file should be checked.
    """
    if path.suffix != ".py":
        return False
    if path in ALLOWED_RAW_SQL_FILES:
        return False
    # Files pending Ibis migration (Phase 4+)
    if path in PENDING_IBIS_MIGRATION_ANALYTICS:
        return False
    if path in PENDING_IBIS_MIGRATION_INGESTION:
        return False
    # Allow all storage layer files (they're the designated SQL zone)
    if "storage" in path.parts:
        return False
    # Check if in application directories
    return any(str(path).startswith(str(app_dir)) for app_dir in APPLICATION_DIRS)


def _find_sql_violations(path: Path) -> list[tuple[int, str]]:
    """Find SQL pattern violations in a file.

    Parameters
    ----------
    path
        File path to check.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, line_content) tuples for violations.
    """
    violations: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        # Skip comments and strings in simple cases
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for SQL keywords that suggest embedded SQL
        if SQL_KEYWORDS_PATTERN.search(line) and any(
            kw in line for kw in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")
        ):
            # Could be SQL - flag for review
            violations.append((i, stripped))

    return violations


def _find_execute_violations(path: Path) -> list[tuple[int, str]]:
    """Find direct execute call violations in a file.

    Parameters
    ----------
    path
        File path to check.

    Returns
    -------
    list[tuple[int, str]]
        List of (line_number, line_content) tuples for violations.
    """
    violations: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        for pattern in EXECUTE_PATTERNS:
            if not re.search(pattern, line):
                continue
            if "SELECT 1" in line:
                continue
            is_ibis_execute = ".count().execute(" in line or ".execute()" in line
            if is_ibis_execute and "con.execute" not in line and "gateway.con.execute" not in line:
                continue
            violations.append((i, stripped))
            break

    return violations


def test_no_raw_sql_in_application_code() -> None:
    """Verify application layers do not embed raw SQL queries.

    This test scans application directories for raw SQL patterns.
    SQL should be confined to:
    - storage/ layer for DDL and mutations
    - Query services that bridge storage and application

    Application code should use:
    - Ibis expressions for queries
    - StorageGateway methods for data access
    - Repository patterns for entity operations
    """
    all_violations: list[str] = []

    for app_dir in APPLICATION_DIRS:
        if not app_dir.exists():
            continue
        for path in app_dir.rglob("*.py"):
            if not _should_check_file(path):
                continue

            sql_violations = _find_sql_violations(path)
            for line_num, line in sql_violations:
                all_violations.append(f"{path}:{line_num}: SQL pattern found: {line[:80]}")

    if all_violations:
        # Strict enforcement: fail on SQL patterns in application code
        msg_lines = [
            f"Found {len(all_violations)} SQL patterns in application code:",
            *all_violations[:MAX_VIOLATIONS_DISPLAY],
        ]
        if len(all_violations) > MAX_VIOLATIONS_DISPLAY:
            remaining = len(all_violations) - MAX_VIOLATIONS_DISPLAY
            msg_lines.append(f"... and {remaining} more violations")
        pytest.fail("\n".join(msg_lines))


def test_no_direct_execute_in_application_code() -> None:
    """Verify application layers do not call execute() directly.

    Direct execute calls should go through:
    - StorageGateway for parameterized queries
    - Ibis expressions for typed queries
    - Repository methods for entity operations
    """
    all_violations: list[str] = []

    for app_dir in APPLICATION_DIRS:
        if not app_dir.exists():
            continue
        for path in app_dir.rglob("*.py"):
            if not _should_check_file(path):
                continue

            execute_violations = _find_execute_violations(path)
            for line_num, line in execute_violations:
                all_violations.append(f"{path}:{line_num}: execute() pattern: {line[:80]}")

    if all_violations:
        # Strict enforcement: fail on execute() patterns in application code
        msg_lines = [
            f"Found {len(all_violations)} execute() calls in application code:",
            *all_violations[:MAX_VIOLATIONS_DISPLAY],
        ]
        if len(all_violations) > MAX_VIOLATIONS_DISPLAY:
            remaining = len(all_violations) - MAX_VIOLATIONS_DISPLAY
            msg_lines.append(f"... and {remaining} more violations")
        pytest.fail("\n".join(msg_lines))


def test_repositories_are_ibis_only() -> None:
    """Verify repository files use only Ibis, not raw SQL.

    This test ensures that repository files do not contain:
    - Raw SQL strings (SELECT, INSERT, etc.)
    - Direct execute() calls
    - Fallback patterns that use raw SQL

    Repository files should use:
    - _ibis_table() for table access
    - _ibis_to_dicts(), _ibis_to_one() for execution
    - Ibis expressions for filtering, ordering, etc.
    """
    # Patterns that indicate SQL usage in repositories
    # NOTE: We do NOT flag `.execute()` because Ibis expressions also use
    # `.execute()` - e.g., `expr.execute()` is valid Ibis usage
    sql_patterns = (
        r"\bSELECT\s+",
        r"\bINSERT\s+",
        r"\bUPDATE\s+",
        r"\bDELETE\s+",
        r"\bWHERE\s+\w+\s*=\s*\?",  # Parameterized WHERE clauses
        r"\.con\.execute\s*\(",  # Direct connection execute calls
        r"fetch_one_dict\s*\(",  # Deprecated helper usage
        r"fetch_all_dicts\s*\(",  # Deprecated helper usage
    )

    violations: list[str] = []

    for repo_file in IBIS_ONLY_REPOSITORY_FILES:
        if not repo_file.exists():
            continue

        text = repo_file.read_text(encoding="utf-8")
        lines = text.splitlines()

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith(("#", '"""', "'''")):
                continue

            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(f"{repo_file}:{i}: SQL pattern: {stripped[:60]}")
                    break

    if violations:
        pytest.fail(
            f"Repositories should use Ibis only (found {len(violations)} violations):\n"
            + "\n".join(violations[:MAX_VIOLATIONS_DISPLAY])
        )


def test_policy_backend_is_single_ddl_source() -> None:
    """Verify DDL operations are centralized in the policy backend.

    This test ensures that CREATE TABLE, DROP TABLE, and similar DDL
    statements are only found in:
    - DuckDBPolicyBackend
    - Legacy DDL module (for backward compatibility)
    - Storage macros
    """
    ddl_patterns = (
        r"\bCREATE\s+TABLE\b",
        r"\bDROP\s+TABLE\b",
        r"\bCREATE\s+INDEX\b",
        r"\bDROP\s+INDEX\b",
        r"\bCREATE\s+SCHEMA\b",
    )

    allowed_ddl_files = frozenset(
        {
            Path("src/codeintel/storage/duckdb_policy_backend.py"),
            Path("src/codeintel/storage/schema/ddl.py"),
            Path("src/codeintel/storage/schema/__init__.py"),
            Path("src/codeintel/storage/macros.py"),
            Path("src/codeintel/storage/helpers/db.py"),
            Path("src/codeintel/storage/metadata/bootstrap.py"),
        }
    )

    violations: list[str] = []
    storage_dir = Path("src/codeintel/storage")

    if not storage_dir.exists():
        pytest.skip("Storage directory not found")

    for path in storage_dir.rglob("*.py"):
        if path in allowed_ddl_files:
            continue
        if "views" in path.parts:
            # View modules use Ibis for CREATE VIEW (handled separately)
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in ddl_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"{path}: contains DDL pattern matching {pattern}")
                break

    if violations:
        pytest.fail(
            "DDL should be centralized in DuckDBPolicyBackend:\n" + "\n".join(violations[:10])
        )


def test_no_raw_sql_views_outside_ibis_views() -> None:
    """Verify raw SQL CREATE VIEW statements are only in ibis_views.py.

    All view definitions should use Ibis expressions via the view registry.
    This test ensures that:
    1. Raw SQL CREATE VIEW is not found in view files (except ibis_views.py)
    2. View creation uses the Ibis registry pattern
    """
    # Pattern for raw SQL view creation
    sql_view_patterns = (
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\b",
        r"\.execute\s*\(\s*[\"'].*CREATE.*VIEW",
        r"con\.execute\s*\(\s*[\"'].*CREATE.*VIEW",
    )

    # Only ibis_views.py should use CREATE VIEW (via Ibis API)
    allowed_view_files = frozenset(
        {
            Path("src/codeintel/storage/views/ibis_views.py"),
        }
    )

    violations: list[str] = []
    views_dir = Path("src/codeintel/storage/views")

    if not views_dir.exists():
        pytest.skip("Views directory not found")

    for path in views_dir.rglob("*.py"):
        if path in allowed_view_files:
            continue
        if path.name == "__init__.py":
            continue
        if path.name == "ibis_registry.py":
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in sql_view_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"{path}: contains raw SQL CREATE VIEW")
                break

    if violations:
        pytest.fail(
            "View definitions should use Ibis via ibis_views.py:\n" + "\n".join(violations[:10])
        )


# Files allowed to use executemany directly (policy backend and storage layer)
ALLOWED_EXECUTEMANY_FILES = frozenset(
    {
        # Policy backend - centralized bulk operations
        Path("src/codeintel/storage/duckdb_policy_backend.py"),
        # Storage layer helpers - low-level DB utilities
        Path("src/codeintel/storage/helpers/db.py"),
        # Metadata bootstrap - schema initialization
        Path("src/codeintel/storage/metadata/bootstrap.py"),
        # Test fixtures and helpers
        Path("tests/_helpers/gateway.py"),
        # Ingestion adapters (pending migration)
        Path("src/codeintel/ingestion/adapters/duckdb_storage.py"),
    }
)


def _is_executemany_allowed(path: Path) -> bool:
    """Check if a file is allowed to use executemany.

    Parameters
    ----------
    path
        File path to check.

    Returns
    -------
    bool
        True if executemany is allowed in this file.
    """
    if path in ALLOWED_EXECUTEMANY_FILES:
        return True
    if path in PENDING_IBIS_MIGRATION_ANALYTICS:
        return True
    return path in PENDING_IBIS_MIGRATION_INGESTION


def _find_executemany_violations(path: Path, pattern: re.Pattern[str]) -> list[str]:
    """Find executemany usage violations in a file.

    Parameters
    ----------
    path
        File path to check.
    pattern
        Regex pattern to match executemany calls.

    Returns
    -------
    list[str]
        List of violation messages.
    """
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for i, line in enumerate(lines, start=1):
        if not pattern.search(line):
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        violations.append(f"{path}:{i}: {stripped[:60]}")

    return violations


def test_executemany_centralized_in_policy_backend() -> None:
    """Verify executemany calls are centralized in policy backend.

    Bulk insert operations should use DuckDBPolicyBackend.bulk_insert()
    instead of direct executemany calls. This test fails for new
    executemany usage outside the allowlist.

    New code should use:
    - DuckDBPolicyBackend.bulk_insert() for bulk inserts
    - DuckDBPolicyBackend.upsert() for insert-or-update
    """
    executemany_pattern = re.compile(r"\.executemany\s*\(")
    violations: list[str] = []

    src_dir = Path("src/codeintel")
    if not src_dir.exists():
        pytest.skip("Source directory not found")

    for path in src_dir.rglob("*.py"):
        if _is_executemany_allowed(path):
            continue
        violations.extend(_find_executemany_violations(path, executemany_pattern))

    if violations:
        msg_lines = [
            f"Found {len(violations)} executemany calls outside policy backend.",
            "Use DuckDBPolicyBackend.bulk_insert() instead.",
            "",
            *violations[:MAX_VIOLATIONS_DISPLAY],
        ]
        if len(violations) > MAX_VIOLATIONS_DISPLAY:
            remaining = len(violations) - MAX_VIOLATIONS_DISPLAY
            msg_lines.append(f"... and {remaining} more violations")
        pytest.fail("\n".join(msg_lines))
