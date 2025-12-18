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
import os
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import ibis
import pyarrow as pa
import sqlglot
from hamilton.function_modifiers import source, tag, value

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.schemas import SchemaManifest, get_schema_provider
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_views
from codeintel.build.serving.semantic_compile_hamilton import (
    collect_semantic_view_tags_from_hamilton,
)
from codeintel.build.spec import BuildSpecCompileOptions, compile_buildspec
from codeintel.build.spec.serdes import buildspec_to_json
from codeintel.build.targets import TargetGraph
from codeintel.storage.gateway import DuckDBError
from codeintel.storage.metadata.bootstrap import sync_derived_lineage_edges
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

TARGET_SPECS = (
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
    view_tags = collect_semantic_view_tags_from_hamilton(modules=(_ibis_views,))
    compiled = compile_semantic_registry_from_views(
        schema_provider=schema_provider,
        view_tags=view_tags,
        version="v1",
    )
    return compiled.to_json() + "\n"


def _schema_manifest_json() -> str:
    schema_provider = get_schema_provider()
    tables = sorted(schema_provider.iter_table_schemas(), key=lambda schema: schema.table_key)
    manifest = SchemaManifest(version="v1", tables=tuple(tables))
    return json.dumps(manifest.to_json_obj(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _buildspec_json() -> str:
    spec = compile_buildspec(options=BuildSpecCompileOptions(include_columns=False))
    return buildspec_to_json(spec, indent=2)


def _environment_json(env: BuildEnv) -> str:
    codeintel_version = _package_version("codeintel")
    duckdb_version = _package_version("duckdb")
    gateway_cfg = getattr(env.gateway, "config", None)
    read_only = bool(getattr(gateway_cfg, "read_only", False))
    extensions = os.environ.get("CODEINTEL_DUCKDB_EXTENSIONS", "").strip()
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
                "threads": os.environ.get("CODEINTEL_DUCKDB_THREADS", "").strip() or None,
                "memory_limit": os.environ.get("CODEINTEL_DUCKDB_MEMORY_LIMIT", "").strip() or None,
                "temp_directory": os.environ.get("CODEINTEL_DUCKDB_TEMP_DIRECTORY", "").strip()
                or None,
                "enable_profiling": os.environ.get("CODEINTEL_DUCKDB_ENABLE_PROFILING", "").strip()
                or None,
                "profiling_output": os.environ.get("CODEINTEL_DUCKDB_PROFILING_OUTPUT", "").strip()
                or None,
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
    for view_key, sql in sql_by_view.items():
        lineage[view_key] = frozenset(extract_referenced_table_keys(sql) - {view_key})

    try:
        sync_derived_lineage_edges(env.gateway.con, repo=env.repo, commit=env.commit, lineage=lineage)
    except DuckDBError:
        LOG.exception("Failed to sync derived lineage edges repo=%s commit=%s", env.repo, env.commit)

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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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
)
@tag(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    node_type="compute",
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


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="helper")
def serving_artifacts__materializations_base(
    m__artifact__semantic_registry: dict[str, object],
    m__artifact__schema_manifest: dict[str, object],
    m__artifact__buildspec: dict[str, object],
) -> dict[str, dict[str, object]]:
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
    dict[str, dict[str, object]]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_SEMANTIC_REGISTRY: m__artifact__semantic_registry,
        SERVING_ARTIFACT_SCHEMA_MANIFEST: m__artifact__schema_manifest,
        SERVING_ARTIFACT_BUILDSPEC: m__artifact__buildspec,
    }


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="helper")
def serving_artifacts__materializations_views(
    m__artifact__environment: dict[str, object],
    m__artifact__views_sql: dict[str, object],
    m__artifact__views_sql_diff: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Collect saver metadata for the view/metadata artifacts.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_ENVIRONMENT: m__artifact__environment,
        SERVING_ARTIFACT_VIEWS_SQL: m__artifact__views_sql,
        SERVING_ARTIFACT_VIEWS_SQL_DIFF: m__artifact__views_sql_diff,
    }


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="helper")
def serving_artifacts__materializations(
    serving_artifacts__materializations_base: dict[str, dict[str, object]],
    serving_artifacts__materializations_views: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Merge all serving artifact materializations.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of artifact name to saver metadata.
    """
    merged = dict(serving_artifacts__materializations_base)
    merged.update(serving_artifacts__materializations_views)
    return merged


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="materialize")
def t__serving_artifacts(
    env: BuildEnv,
    graph: TargetGraph,
    serving_artifacts__materializations: dict[str, dict[str, object]],
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
