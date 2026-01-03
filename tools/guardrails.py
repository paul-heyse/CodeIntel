"""Static guardrails for migration-sensitive anti-patterns.

This script scans source/test code for banned patterns called out in the
ibis+pandera+sqlglot migration plan. It is intended to be wired into the
quality gate and fail fast when deprecated surfaces reappear.

It also enforces Hamilton build invariants (graph validation) so build-breaking
tag/contract drift is caught without requiring a full test run.
"""

from __future__ import annotations

import inspect
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from codeintel.build.hamilton.graph_validation import (
    validate_graph,
    validation_result_to_json,
)
from codeintel.build.meta.contract_catalog import persist_contract_catalog_to_connection
from codeintel.build.schemas.contract_service import configure_contract_service
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.cli.resolution import resolve_from_params
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.duckdb_types import DuckDBError
from codeintel.storage.gateway import (
    DuckDBConnection,
    MemoryGatewayOptions,
    StorageConfig,
    StorageConnectionError,
    open_gateway,
    open_inference_gateway,
    open_memory_gateway,
)
from codeintel.storage.validation import ContractValidationMode

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.hamilton.validate import GraphValidationResult
    from codeintel.cli.resolution.types import ResolvedRuntime
    from codeintel.runtime.runtime_bundle import RuntimeBundle
    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord


@dataclass(frozen=True)
class GuardrailSettings:
    """Central guardrail configuration."""

    base_dirs: tuple[str, ...]
    streaming_include_prefixes: tuple[str, ...]
    streaming_allow_prefixes: tuple[str, ...]


GUARDRAIL_SETTINGS = GuardrailSettings(
    base_dirs=("src", "tests", "tools", "scripts"),
    streaming_include_prefixes=(
        "src/codeintel/build/",
        "src/codeintel/serving/",
        "src/codeintel/storage/validation/",
    ),
    streaming_allow_prefixes=(
        "tests/",
        "tools/",
        "docs/_scripts/",
    ),
)

BASE_DIRS = GUARDRAIL_SETTINGS.base_dirs
_SELF_REL_PATH = "tools/guardrails.py"
STREAMING_GUARDRAIL_PREFIXES = GUARDRAIL_SETTINGS.streaming_include_prefixes
STREAMING_GUARDRAIL_ALLOW_PREFIXES = GUARDRAIL_SETTINGS.streaming_allow_prefixes
SDK_GUARDRAIL_ALLOW_PREFIXES: tuple[str, ...] = (
    "src/codeintel/build/hamilton/tagging.py",
    "src/codeintel/sdk/",
)
TAG_GUARDRAIL_ALLOW_PREFIXES: tuple[str, ...] = (
    "src/codeintel/build/hamilton/tagging.py",
    "src/codeintel/sdk/",
    "src/codeintel/core/hamilton/tagging_helpers.py",
)
MODULE_DISCOVERY_ALLOW_PREFIXES: tuple[str, ...] = ("src/codeintel/runtime/module_resolver.py",)


@dataclass(frozen=True)
class Guardrail:
    """Guardrail rule with pattern and allowlist prefixes."""

    name: str
    pattern: re.Pattern[str]
    message: str
    include_prefixes: tuple[str, ...] = ()
    allow_prefixes: tuple[str, ...] = ()


class _SchemaObservationSource(Protocol):
    """Schema observation accessor used by guardrail checks."""

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None: ...

    def drift_summary_report(self, *, limit: int = 50) -> dict[str, object]: ...

    def has_contract_arrow_schema(self, *, table_key: str) -> bool: ...


GUARDRAILS: tuple[Guardrail, ...] = (
    Guardrail(
        name="normalized_macros",
        pattern=re.compile(
            r"\b(MacroRequirement|require_normalized_macros|requires_normalized_macro)\b"
        ),
        message="Normalized macro compatibility is removed; drop this surface.",
    ),
    Guardrail(
        name="legacy_sql_builder",
        pattern=re.compile(r"\b(SafeTable|SafeColumn|QueryBuilder|codeintel\.storage\.sql)\b"),
        message="Legacy SQL builder usage is forbidden; use DuckDBPolicyBackend or Ibis.",
    ),
    Guardrail(
        name="core_layer_imports",
        pattern=re.compile(r"(?m)^\\s*(?:from|import)\\s+codeintel\\.(?:storage|build|serving)\\b"),
        message="Core must not import build/storage/serving modules.",
        include_prefixes=("src/codeintel/core/",),
    ),
    Guardrail(
        name="legacy_macro_helpers",
        pattern=re.compile(r"\b(macro_exists|safe_macro_exists|INGEST_MACRO_TABLES)\b"),
        message="Legacy macro helpers are removed.",
    ),
    Guardrail(
        name="raw_con_execute",
        pattern=re.compile(r"\.con\.execute\("),
        message="Raw con.execute is only allowed inside storage internals.",
        allow_prefixes=(
            "src/codeintel/storage/",
            "tests/",
        ),
    ),
    Guardrail(
        name="direct_ibis_table_calls",
        pattern=re.compile(r"\.ibis\.table\("),
        message=(
            "Direct `.ibis.table(...)` is forbidden; use "
            "codeintel.storage.gateway.ibis_facade.table."
        ),
        allow_prefixes=(
            "src/codeintel/storage/gateway/ibis_facade.py",
            "src/codeintel/storage/ibis_adapter.py",
            "src/codeintel/build/hamilton/validate.py",
        ),
        include_prefixes=("src/codeintel/",),
    ),
    Guardrail(
        name="build_policy_write_calls",
        pattern=re.compile(
            r"\.policy\.(?:delete_for_snapshot|bulk_insert_mappings|bulk_insert|delete)\("
        ),
        message="Build must not call gateway.policy write methods; use Warehouse.materialize_*.",
        allow_prefixes=("src/codeintel/storage/warehouse.py",),
        include_prefixes=("src/codeintel/build/",),
    ),
    Guardrail(
        name="legacy_build_context_stack",
        pattern=re.compile(r"\bcodeintel\.build\.(context|context_base|result|protocols)\b"),
        message="Legacy build context stack is removed; use Hamilton BuildEnv/executor patterns.",
    ),
    Guardrail(
        name="hamilton_save_to_decorator",
        pattern=re.compile(
            r"\b(from hamilton\.function_modifiers\.adapters import SaveToDecorator|"
            r"@SaveToDecorator\b)"
        ),
        message=(
            "Hamilton SaveToDecorator is forbidden; use SaveToObjectMetadataDecorator "
            "(metadata typed as dict[str, object])."
        ),
    ),
    Guardrail(
        name="removed_ibis_typing_modules",
        pattern=re.compile(r"\bcodeintel\.(?:storage\.ibis_types|build\.ibis_typing)\b"),
        message=(
            "Legacy Ibis typing modules are removed; use codeintel.core.ibis_typing "
            "and codeintel.storage.gateway.ibis_facade."
        ),
    ),
    Guardrail(
        name="build_cast_any",
        pattern=re.compile(r"\bcast\(\s*(?:\"Any\"|'Any'|Any)\s*,"),
        message=(
            '`cast("Any", ...)` is forbidden outside codeintel.core.ibis_typing; use '
            "codeintel.core.ibis_typing helpers."
        ),
        include_prefixes=("src/codeintel/",),
        allow_prefixes=("src/codeintel/core/ibis_typing.py",),
    ),
    Guardrail(
        name="compute_result_any",
        pattern=re.compile(r"\b(type\s+)?ComputeResult\s*(?::\s*TypeAlias)?\s*=\s*Any\b"),
        message="ComputeResult = Any is forbidden; use ExecutionResult.",
        include_prefixes=("src/codeintel/build/",),
    ),
    Guardrail(
        name="build_direct_hamilton_tag",
        pattern=re.compile(r"\bfrom hamilton\.function_modifiers import [^\n]*\btag\b"),
        message=(
            "Direct imports of Hamilton's @tag decorator are forbidden; use SDK/tagging helpers."
        ),
        include_prefixes=("src/codeintel/",),
        allow_prefixes=TAG_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="core_to_pylist",
        pattern=re.compile(r"\.to_pylist\("),
        message="Core must not materialize Arrow arrays via to_pylist; use streaming-safe access.",
        include_prefixes=("src/codeintel/core/",),
        allow_prefixes=(
            "src/codeintel/core/exports/",
            "tests/",
        ),
    ),
    Guardrail(
        name="core_json_column_types",
        pattern=re.compile(r"Column\\(\\s*\"[^\"]+\"\\s*,\\s*\"JSON\""),
        message="Core table schemas must not declare JSON columns; use BLOB or Arrow-native types.",
        include_prefixes=("src/codeintel/core/",),
        allow_prefixes=("tests/",),
    ),
    Guardrail(
        name="direct_hamilton_schema_modifier",
        pattern=re.compile(
            r"\bfrom hamilton\.function_modifiers import [^\n]*\bschema\b"
            r"|\bimport hamilton\.function_modifiers\.schema\b"
        ),
        message=(
            "Direct imports of Hamilton schema modifiers are forbidden; use "
            "codeintel.build.hamilton.tagging helpers or codeintel.sdk annotations."
        ),
        include_prefixes=("src/codeintel/",),
        allow_prefixes=SDK_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="direct_hamilton_check_output_modifier",
        pattern=re.compile(
            r"\bfrom hamilton\.function_modifiers import [^\n]*\bcheck_output\b"
            r"|\bimport hamilton\.function_modifiers\.check_output\b"
        ),
        message=(
            "Direct imports of Hamilton check_output modifiers are forbidden; use "
            "codeintel.sdk validation helpers."
        ),
        include_prefixes=("src/codeintel/",),
        allow_prefixes=SDK_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="removed_build_registry_module",
        pattern=re.compile(r"\bcodeintel\.build\.registry\b"),
        message=(
            "codeintel.build.registry is removed; use "
            "codeintel.build.target_system.load_target_system()."
        ),
        include_prefixes=("src/", "tests/", "tools/"),
    ),
    Guardrail(
        name="streaming_to_table",
        pattern=re.compile(r"\.to_table\("),
        message="Avoid to_table(); use RecordBatchReader or batch iterators instead.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="streaming_read_all",
        pattern=re.compile(r"\.read_all\("),
        message="Avoid read_all(); use batch readers instead of eager materialization.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="streaming_relation_arrow",
        pattern=re.compile(r"\.arrow\("),
        message="Avoid relation.arrow(); use fetch_record_batch or scan_batches instead.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="streaming_relation_pl",
        pattern=re.compile(r"\.pl\("),
        message="Avoid relation.pl(); use fetch_record_batch or scan_batches instead.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="streaming_fetchall",
        pattern=re.compile(r"\.fetchall\("),
        message="Avoid fetchall(); use fetch_record_batch and stream batches instead.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="streaming_to_pandas",
        pattern=re.compile(r"\.to_pandas\("),
        message="Avoid to_pandas(); keep streaming Arrow/Polars primitives.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="pandas_values_property",
        pattern=re.compile(r"(?s)\b(?:import pandas|from pandas)\b.*?\.values\b(?!\s*\()"),
        message="Avoid pandas .values; prefer .to_numpy() or Arrow/Polars buffers.",
        include_prefixes=STREAMING_GUARDRAIL_PREFIXES,
        allow_prefixes=STREAMING_GUARDRAIL_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="module_discovery_static_targets",
        pattern=re.compile(r"(?m)^\s*TARGETS\s*=\s*\["),
        message="Static target registries are forbidden; targets are discovered via modules.",
        include_prefixes=("src/codeintel/",),
        allow_prefixes=MODULE_DISCOVERY_ALLOW_PREFIXES,
    ),
    Guardrail(
        name="module_discovery_codeintel_targets_import",
        pattern=re.compile(r"\bcodeintel_targets\."),
        message="Direct codeintel_targets imports are forbidden; use module resolver discovery.",
        include_prefixes=("src/codeintel/",),
        allow_prefixes=MODULE_DISCOVERY_ALLOW_PREFIXES,
    ),
)


def iter_candidate_files(repo_root: Path) -> Iterable[Path]:
    """Yield files under the configured base directories.

    Yields
    ------
    Path
        Python files to scan for guardrail violations.
    """
    for base in BASE_DIRS:
        root = repo_root / base
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.is_file():
                rel = path.relative_to(repo_root).as_posix()
                if rel == _SELF_REL_PATH:
                    continue
                yield path


def find_violations(repo_root: Path) -> list[str]:
    """Scan for guardrail violations and return human-friendly messages.

    Returns
    -------
    list[str]
        Collected violation messages.
    """
    violations: list[str] = []
    for path in iter_candidate_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for rule in GUARDRAILS:
            if rule.include_prefixes and not rel.startswith(rule.include_prefixes):
                continue
            if rule.allow_prefixes and rel.startswith(rule.allow_prefixes):
                continue
            for match in rule.pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{rel}:{line}: {rule.name}: {rule.message}")
    return violations


def _guardrails_storage_config(runtime: ResolvedRuntime) -> StorageConfig:
    """Build a storage config for guardrail checks.

    Parameters
    ----------
    runtime
        Resolved runtime containing database and snapshot paths.

    Returns
    -------
    StorageConfig
        Read-only config with contract validation disabled.
    """
    return StorageConfig(
        db_path=runtime.db_path,
        read_only=True,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
        validation_mode=ContractValidationMode.OFF,
        validation_summary_path=None,
        suppress_registry_health_log=True,
        repo=runtime.repo,
        commit=runtime.commit,
    )


def _module_name_for_node(node: object) -> str | None:
    originators = getattr(node, "originating_functions", None)
    if not isinstance(originators, (list, tuple)) or not originators:
        return None
    fn = originators[0]
    if not callable(fn):
        return None
    unwrapped = inspect.unwrap(fn)
    module_name = getattr(unwrapped, "__module__", None)
    return module_name if isinstance(module_name, str) else None


def _node_provenance_map(runtime_bundle: RuntimeBundle) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    for node in runtime_bundle.dr.graph.nodes.values():
        node_name = getattr(node, "name", None)
        if not isinstance(node_name, str):
            continue
        module_name = _module_name_for_node(node)
        if module_name is None:
            continue
        entry: dict[str, object] = {"module": module_name}
        provenance = runtime_bundle.module_provenance.get(module_name)
        if provenance is not None:
            entry["origin"] = provenance.origin
            entry["file_path"] = provenance.file_path
            entry["plugin_name"] = provenance.plugin_name
            entry["dist_name"] = provenance.dist_name
            entry["dist_version"] = provenance.dist_version
        mapping[node_name] = entry
    return mapping


def schema_observations_available(schemas: _SchemaObservationSource) -> bool:
    """Return True when schema observations can be queried.

    Parameters
    ----------
    schemas
        Schema observation accessor.

    Returns
    -------
    bool
        True when observation summaries are queryable.
    """
    try:
        summary = schemas.drift_summary_report()
    except (DuckDBError, RuntimeError, TypeError, ValueError):
        return False
    total_tables = _coerce_int(summary.get("total_tables"))
    return total_tables is not None


def _has_observation_or_contract(
    schemas: _SchemaObservationSource,
    *,
    table_key: str,
) -> bool:
    observation = schemas.load_latest_schema_observation(table_key=table_key)
    if observation is not None and observation.arrow_schema_ipc_b64.strip():
        return True
    return schemas.has_contract_arrow_schema(table_key=table_key)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
    return None


def _schema_observation_issues(
    runtime_bundle: RuntimeBundle,
    *,
    gateway: StorageGateway,
) -> list[str]:
    table_targets: dict[str, str] = {
        table_key: output.producer_target
        for table_key, output in runtime_bundle.catalog.table_outputs.items()
        if output.role == "contract"
    }
    if not table_targets:
        return []
    if not schema_observations_available(gateway.schemas):
        sys.stderr.write("Schema observations unavailable; skipping observation guardrails.\n")
        return []
    return missing_schema_observations(table_targets, schemas=gateway.schemas)


def missing_schema_observations(
    table_targets: Mapping[str, str],
    *,
    schemas: _SchemaObservationSource,
) -> list[str]:
    """Return table keys missing observation or renderer cache payloads.

    Parameters
    ----------
    table_targets
        Mapping of table key to producing target name.
    schemas
        Schema observation accessor.

    Returns
    -------
    list[str]
        Missing table keys with target context.
    """
    missing: list[str] = []
    for table_key, target in sorted(table_targets.items()):
        if _has_observation_or_contract(schemas, table_key=table_key):
            continue
        missing.append(f"{table_key} (target={target})")
    return missing


def main() -> int:
    """Entry point for the guardrail scanner.

    Returns
    -------
    int
        Zero when clean, non-zero when violations are found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    violations = find_violations(repo_root)
    if violations:
        for line in violations:
            sys.stderr.write(f"{line}\n")
        return 1

    runtime = resolve_from_params(
        {"project_root": repo_root, "repo_root": repo_root},
        allow_fallback=True,
    )
    config = _guardrails_storage_config(runtime)
    try:
        gateway = open_gateway(config)
    except (FileNotFoundError, StorageConnectionError):
        inference_gateway = open_inference_gateway(schema_provider=MappingSchemaProvider({}))
        try:
            runtime_bundle = compose_cli_runtime_bundle(runtime=runtime, gateway=inference_gateway)
            configure_contract_service(runtime=runtime_bundle)
        finally:
            inference_gateway.close()

        def _seed_contract_catalog(con: DuckDBConnection) -> None:
            persist_contract_catalog_to_connection(con, inputs={"source": "guardrails"})

        gateway = open_memory_gateway(
            options=MemoryGatewayOptions(
                apply_schema=False,
                ensure_views=False,
                validate_schema=False,
                suppress_registry_health_log=True,
            ),
            seed_contract_catalog=_seed_contract_catalog,
        )
    runtime_bundle: RuntimeBundle | None = None
    graph_result: GraphValidationResult | None = None
    observation_issues: list[str] = []
    try:
        runtime_bundle = compose_cli_runtime_bundle(runtime=runtime, gateway=gateway)
        graph_result = validate_graph(runtime=runtime_bundle, validate_schema=False)
        observation_issues = _schema_observation_issues(runtime_bundle, gateway=gateway)
    except ImportError as exc:
        sys.stderr.write("Hamilton graph validation could not run due to an import error.\n")
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.stderr.write(traceback.format_exc())
        return 1
    finally:
        gateway.close()
    if graph_result is None:
        sys.stderr.write("Hamilton graph validation did not run.\n")
        return 1
    if graph_result.has_errors:
        sys.stderr.write("Hamilton graph validation failed.\n")
        node_provenance = (
            _node_provenance_map(runtime_bundle) if runtime_bundle is not None else None
        )
        sys.stderr.write(
            validation_result_to_json(
                graph_result,
                indent=2,
                node_provenance=node_provenance,
            )
        )
        return 1
    if observation_issues:
        sys.stderr.write("Schema observation guardrails warning: missing observations.\n")
        for issue in observation_issues:
            sys.stderr.write(f"{issue}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
