"""Native Hamilton target for serving-layer build artifacts.

This module implements the ``serving_artifacts`` target, which compiles and
materializes deterministic file artifacts used by serving/publishing:

- ``semantic_registry.json`` (semantic view registry)
- ``schema_manifest.json`` (table schema manifest)
- ``buildspec.json`` (BuildSpec compiled from the Hamilton DAG)

All artifacts are written via ``FileArtifactSaver`` so I/O is DAG-visible and
recorded in the build manifest.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ibis
import pyarrow as pa
import sqlglot
from hamilton.function_modifiers import source, value

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tag_index import TagIndex
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_materialize
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.serving.semantic_compile import (
    compile_semantic_registry_from_tag_index,
)
from codeintel.build.spec import BuildSpecCompileOptions, compile_buildspec
from codeintel.build.spec.serdes import buildspec_to_json
from codeintel.build.targets import TargetGraph
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.metadata.sync import (
    sync_derived_lineage_columns,
    sync_derived_lineage_edges,
)
from codeintel.storage.sqlglot_tools import extract_column_lineage_duckdb
from codeintel.storage.views import ibis_views as _ibis_views
from codeintel.storage.views.dependencies import extract_referenced_table_keys
from codeintel.storage.views.diff import diff_view_sql_maps
from codeintel.storage.views.discovery import discover_view_builders

LOG = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

SERVING_ARTIFACTS_TARGET_NAME = "serving_artifacts"

SERVING_ARTIFACT_SEMANTIC_REGISTRY = "semantic_registry"
SERVING_ARTIFACT_SCHEMA_MANIFEST = "schema_manifest"
SERVING_ARTIFACT_BUILDSPEC = "buildspec"
SERVING_ARTIFACT_ENVIRONMENT = "environment"
SERVING_ARTIFACT_VIEWS_SQL = "views_sql"
SERVING_ARTIFACT_VIEWS_SQL_DIFF = "views_sql_diff"

SERVING_ARTIFACT_SPECS = (
    ArtifactSpec(
        SERVING_ARTIFACT_SEMANTIC_REGISTRY,
        "{build_dir}/serving/artifacts/semantic_registry.json",
        "Compiled semantic registry for serving",
    ),
    ArtifactSpec(
        SERVING_ARTIFACT_SCHEMA_MANIFEST,
        "{build_dir}/serving/artifacts/schema_manifest.json",
        "Compiled schema manifest for serving",
    ),
    ArtifactSpec(
        SERVING_ARTIFACT_BUILDSPEC,
        "{build_dir}/serving/artifacts/buildspec.json",
        "Compiled BuildSpec contract for serving",
    ),
    ArtifactSpec(
        SERVING_ARTIFACT_ENVIRONMENT,
        "{build_dir}/serving/artifacts/environment.json",
        "Captured tool and configuration metadata for this snapshot",
    ),
    ArtifactSpec(
        SERVING_ARTIFACT_VIEWS_SQL,
        "{build_dir}/serving/artifacts/views_sql.json",
        "Compiled SQL for all registered views",
    ),
    ArtifactSpec(
        SERVING_ARTIFACT_VIEWS_SQL_DIFF,
        "{build_dir}/serving/artifacts/views_sql_diff.json",
        "Diff summary between previous and current view SQL maps",
    ),
)

register_output_targets(
    make_output_target(
        name=SERVING_ARTIFACTS_TARGET_NAME,
        module="export",
        description=(
            "Compile deterministic serving artifacts (semantic registry, schema manifest, buildspec)."
        ),
        options=TargetSpecOptions(
            artifacts=SERVING_ARTIFACT_SPECS,
        ),
    ),
)


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _semantic_registry_json() -> str:
    schema_provider = get_schema_provider()
    tag_index = TagIndex.from_modules(modules=(_ibis_views,))
    compiled = compile_semantic_registry_from_tag_index(
        schema_provider=schema_provider,
        tag_index=tag_index,
        version="v1",
    )
    return compiled.to_json() + "\n"


def _schema_manifest_json() -> str:
    schema_provider = get_schema_provider()
    manifest = compile_schema_manifest(
        provider=schema_provider,
        request=SchemaManifestRequest(
            all_targets=True,
            stable=True,
            version="v1",
            include_views=False,
            include_artifacts=False,
        ),
    )
    return manifest.to_json() + "\n"


def _buildspec_json() -> str:
    spec = compile_buildspec(options=BuildSpecCompileOptions(include_columns=False))
    return buildspec_to_json(spec, indent=2)


def _environment_json(env: BuildEnv) -> str:
    codeintel_version = _package_version("codeintel")
    duckdb_version = _package_version("duckdb")
    gateway_cfg = getattr(env.gateway, "config", None)
    read_only = bool(getattr(gateway_cfg, "read_only", False))
    execution_settings = env.execution_settings
    extensions = ""
    if execution_settings.duckdb_extensions:
        extensions = ", ".join(execution_settings.duckdb_extensions)
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "repo": env.repo,
        "commit": env.commit,
        "codeintel": {"version": codeintel_version},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "tools": {
            "duckdb": duckdb_version,
            "ibis": str(getattr(ibis, "__version__", "unknown")),
            "pyarrow": str(getattr(pa, "__version__", "unknown")),
            "sqlglot": str(getattr(sqlglot, "__version__", "unknown")),
        },
        "duckdb": {
            "read_only": read_only,
            "extensions_env": extensions,
            "connect_env": {
                "threads": execution_settings.duckdb_threads,
                "memory_limit": execution_settings.duckdb_memory_limit,
                "temp_directory": (
                    str(execution_settings.duckdb_temp_directory)
                    if execution_settings.duckdb_temp_directory is not None
                    else None
                ),
                "enable_profiling": execution_settings.duckdb_enable_profiling,
                "profiling_output": (
                    str(execution_settings.duckdb_profiling_output)
                    if execution_settings.duckdb_profiling_output is not None
                    else None
                ),
            },
        },
        "argv0": sys.argv[0] if sys.argv else None,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _views_sql_json(env: BuildEnv) -> str:
    builders = discover_view_builders(modules=(_ibis_views,))
    ibis_gateway = env.gateway.ibis

    sql_by_view: dict[str, str] = {}
    for spec in builders:
        expr = spec.builder(ibis_gateway)
        sql_by_view[spec.table_key.lower()] = ibis_gateway.con.compile(expr)

    lineage: dict[str, frozenset[str]] = {}
    column_lineage: dict[str, dict[str, frozenset[str]]] = {}
    for view_key, sql in sql_by_view.items():
        lineage[view_key] = frozenset(extract_referenced_table_keys(sql) - {view_key})
        column_lineage[view_key] = extract_column_lineage_duckdb(sql)

    try:
        sync_derived_lineage_edges(
            env.gateway.con, repo=env.repo, commit=env.commit, lineage=lineage
        )
    except DuckDBError:
        LOG.exception(
            "Failed to sync derived lineage edges repo=%s commit=%s", env.repo, env.commit
        )

    try:
        sync_derived_lineage_columns(
            env.gateway.con,
            repo=env.repo,
            commit=env.commit,
            lineage=column_lineage,
        )
    except DuckDBError:
        LOG.exception(
            "Failed to sync derived lineage columns repo=%s commit=%s", env.repo, env.commit
        )

    return json.dumps(sql_by_view, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _views_sql_diff_json(env: BuildEnv, *, current_views_sql: str) -> str:
    after = json.loads(current_views_sql)
    if not isinstance(after, dict):
        msg = "views_sql artifact did not contain an object mapping"
        raise TypeError(msg)

    before: dict[str, str] = {}
    try:
        versions = env.gateway.assets.get_asset_versions(
            repo=env.repo,
            commit=env.commit,
            asset_kind="artifact",
            asset_key=SERVING_ARTIFACT_VIEWS_SQL,
            limit=1,
        )
    except DuckDBError:
        LOG.exception(
            "Failed to load prior views_sql versions repo=%s commit=%s",
            env.repo,
            env.commit,
        )
        versions = []
    if versions:
        location = versions[0].location
        if isinstance(location, str):
            path = Path(location)
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    text = ""
                if text:
                    try:
                        loaded = json.loads(text)
                    except ValueError:
                        loaded = None
                    if isinstance(loaded, dict):
                        before = {str(k).lower(): str(v) for k, v in loaded.items()}

    diff = diff_view_sql_maps(before=before, after={str(k): str(v) for k, v in after.items()})
    payload = {
        "repo": env.repo,
        "commit": env.commit,
        "previous_present": bool(before),
        "views": diff,
    }
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_SEMANTIC_REGISTRY}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_SEMANTIC_REGISTRY),
    path_template=value("{build_dir}/serving/artifacts/semantic_registry.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__semantic_registry",
)
def serving_artifacts__semantic_registry(_env: BuildEnv) -> str:
    """Compile semantic registry JSON for serving.

    Parameters
    ----------
    _env
        Build environment (unused; required for Hamilton input binding).

    Returns
    -------
    str
        Newline-terminated semantic registry JSON payload.
    """
    return _semantic_registry_json()


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_SCHEMA_MANIFEST}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_SCHEMA_MANIFEST),
    path_template=value("{build_dir}/serving/artifacts/schema_manifest.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__schema_manifest",
)
def serving_artifacts__schema_manifest(_env: BuildEnv) -> str:
    """Compile schema manifest JSON for serving.

    Parameters
    ----------
    _env
        Build environment (unused; required for Hamilton input binding).

    Returns
    -------
    str
        Newline-terminated schema manifest JSON payload.
    """
    return _schema_manifest_json()


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_BUILDSPEC}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_BUILDSPEC),
    path_template=value("{build_dir}/serving/artifacts/buildspec.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__buildspec",
)
def serving_artifacts__buildspec(_env: BuildEnv) -> str:
    """Compile BuildSpec JSON for serving.

    Parameters
    ----------
    _env
        Build environment (unused; required for Hamilton input binding).

    Returns
    -------
    str
        Newline-terminated BuildSpec JSON payload.
    """
    return _buildspec_json()


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_ENVIRONMENT}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_ENVIRONMENT),
    path_template=value("{build_dir}/serving/artifacts/environment.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__environment",
)
def serving_artifacts__environment(env: BuildEnv) -> str:
    """Capture environment metadata for serving snapshots.

    Parameters
    ----------
    env
        Build environment with gateway access.

    Returns
    -------
    str
        Newline-terminated JSON payload describing tool versions and settings.
    """
    return _environment_json(env)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_VIEWS_SQL}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_VIEWS_SQL),
    path_template=value("{build_dir}/serving/artifacts/views_sql.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__views_sql",
)
def serving_artifacts__views_sql(env: BuildEnv) -> str:
    """Compile compiled view SQL map JSON for serving.

    Parameters
    ----------
    env
        Build environment with gateway access.

    Returns
    -------
    str
        Newline-terminated JSON mapping of view_key -> compiled SQL.
    """
    return _views_sql_json(env)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_VIEWS_SQL_DIFF}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_VIEWS_SQL_DIFF),
    path_template=value("{build_dir}/serving/artifacts/views_sql_diff.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__views_sql_diff",
)
def serving_artifacts__views_sql_diff(env: BuildEnv, serving_artifacts__views_sql: str) -> str:
    """Compute a diff summary between the latest stored and current view SQL maps.

    Parameters
    ----------
    env
        Build environment with gateway access.
    serving_artifacts__views_sql
        Newline-terminated JSON mapping of view_key -> compiled SQL for this build.

    Returns
    -------
    str
        Newline-terminated JSON diff artifact.
    """
    return _views_sql_diff_json(env, current_views_sql=serving_artifacts__views_sql)


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations_base(
    m__artifact__semantic_registry: MaterializationMetadata,
    m__artifact__schema_manifest: MaterializationMetadata,
    m__artifact__buildspec: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect saver metadata for the base serving artifacts.

    Parameters
    ----------
    m__artifact__semantic_registry
        Saver metadata for the semantic registry artifact.
    m__artifact__schema_manifest
        Saver metadata for the schema manifest artifact.
    m__artifact__buildspec
        Saver metadata for the BuildSpec artifact.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_SEMANTIC_REGISTRY: m__artifact__semantic_registry,
        SERVING_ARTIFACT_SCHEMA_MANIFEST: m__artifact__schema_manifest,
        SERVING_ARTIFACT_BUILDSPEC: m__artifact__buildspec,
    }


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations_views(
    m__artifact__environment: MaterializationMetadata,
    m__artifact__views_sql: MaterializationMetadata,
    m__artifact__views_sql_diff: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect saver metadata for the view/metadata artifacts.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_ENVIRONMENT: m__artifact__environment,
        SERVING_ARTIFACT_VIEWS_SQL: m__artifact__views_sql,
        SERVING_ARTIFACT_VIEWS_SQL_DIFF: m__artifact__views_sql_diff,
    }


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations(
    serving_artifacts__materializations_base: dict[str, MaterializationMetadata],
    serving_artifacts__materializations_views: dict[str, MaterializationMetadata],
) -> dict[str, MaterializationMetadata]:
    """Merge all serving artifact materializations.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of artifact name to saver metadata.
    """
    merged = dict(serving_artifacts__materializations_base)
    merged.update(serving_artifacts__materializations_views)
    return merged


@tag_materialize(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def t__serving_artifacts(
    env: BuildEnv,
    graph: TargetGraph,
    serving_artifacts__materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Finalize serving artifacts materialization and persist manifest.

    Parameters
    ----------
    env
        Build environment for manifest persistence.
    graph
        Target graph containing the serving_artifacts contract.
    serving_artifacts__materializations
        Saver metadata mapping keyed by artifact name.

    Returns
    -------
    TargetRunRecord
        Record describing the artifact materialization outcome.
    """
    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name=SERVING_ARTIFACTS_TARGET_NAME,
        materializations=serving_artifacts__materializations,
    )


__all__ = [
    "t__serving_artifacts",
]
