"""Native Hamilton target for serving-layer build artifacts.

This module implements the ``serving_artifacts`` target, which compiles and
materializes deterministic file artifacts used by serving/publishing:

- ``semantic_registry.json`` (semantic view registry)
- ``schema_manifest.json`` (schema manifest v2: tables, views, artifacts)
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

import polars as pl
import pyarrow as pa
import sqlglot
from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.build.semantic.registry_compiler import compile_semantic_registry
from codeintel.build.spec import BuildSpecCompileOptions, compile_buildspec
from codeintel.build.spec.serdes import buildspec_to_json
from codeintel.build.tabular.frames import lazyframe_for_table_columns
from codeintel.core.columnar.rows import columnar_buffer_for_table_key
from codeintel.core.datasets.manifests import dataset_manifest_path
from codeintel.core.hamilton.tag_query import TagQuery

LOG = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, SchemaIndex, TagQuery, TargetRunRecord)

SERVING_ARTIFACTS_TARGET_NAME = "serving_artifacts"

SERVING_ARTIFACT_SEMANTIC_REGISTRY = "semantic_registry"
SERVING_ARTIFACT_SCHEMA_MANIFEST = "schema_manifest"
SERVING_ARTIFACT_BUILDSPEC = "buildspec"
SERVING_ARTIFACT_ENVIRONMENT = "environment"
SERVING_ARTIFACT_DATASET_MANIFEST_PATHS = "dataset_manifest_paths"
SCHEMA_INFERENCE_ERRORS_TABLE_KEY = "core.schema_inference_errors"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _semantic_registry_json(tag_query: TagQuery) -> str:
    schema_provider = get_schema_provider()
    compiled = compile_semantic_registry(
        tag_query=tag_query,
        schema_provider=schema_provider,
        version="v1",
    )
    return compiled.to_json() + "\n"


def _schema_manifest_json(
    *,
    catalog: DagCatalog,
    schema_index: SchemaIndex,
    tag_query: TagQuery,
) -> str:
    schema_provider = get_schema_provider()
    manifest = compile_schema_manifest(
        provider=schema_provider,
        context=SchemaManifestContext(
            catalog=catalog,
            schema_index=schema_index,
            tag_query=tag_query,
        ),
        request=SchemaManifestRequest(
            all_targets=True,
            stable=True,
            version="v2",
            include_views=True,
            include_artifacts=True,
            include_provenance=True,
        ),
    )
    return manifest.to_json() + "\n"


def _buildspec_json(catalog: DagCatalog) -> str:
    schema_provider = get_schema_provider()
    spec = compile_buildspec(
        catalog=catalog,
        provider=schema_provider,
        options=BuildSpecCompileOptions(include_columns=False),
    )
    return buildspec_to_json(spec, indent=2)


def _environment_json(env: BuildEnv) -> str:
    codeintel_version = _package_version("codeintel")
    duckdb_version = _package_version("duckdb")
    gateway_cfg = env.gateway.config if env.gateway is not None else None
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


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_SEMANTIC_REGISTRY}"),
    env=source("env"),
    catalog=source("catalog"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_SEMANTIC_REGISTRY),
    path_template=value("{build_dir}/serving/artifacts/semantic_registry.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__semantic_registry",
)
def serving_artifacts__semantic_registry(env: BuildEnv, tag_query: TagQuery) -> str:
    """Compile semantic registry JSON for serving.

    Parameters
    ----------
    env
        Build environment (unused; required for Hamilton input binding).
    tag_query
        Cached Hamilton tag query helper.

    Returns
    -------
    str
        Newline-terminated semantic registry JSON payload.
    """
    _ = env
    return _semantic_registry_json(tag_query)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_SCHEMA_MANIFEST}"),
    env=source("env"),
    catalog=source("catalog"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_SCHEMA_MANIFEST),
    path_template=value("{build_dir}/serving/artifacts/schema_manifest.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__schema_manifest",
)
def serving_artifacts__schema_manifest(
    env: BuildEnv,
    catalog: DagCatalog,
    schema_index: SchemaIndex,
    tag_query: TagQuery,
) -> str:
    """Compile schema manifest JSON for serving.

    Parameters
    ----------
    env
        Build environment (unused; required for Hamilton input binding).
    catalog
        DAG catalog used to compute schema manifest contents.
    schema_index
        Schema index used for inference and error tracking.
    tag_query
        TagQuery helper for view discovery.

    Returns
    -------
    str
        Newline-terminated schema manifest JSON payload.
    """
    _ = env
    return _schema_manifest_json(
        catalog=catalog,
        schema_index=schema_index,
        tag_query=tag_query,
    )


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_DATASET_MANIFEST_PATHS}"),
    env=source("env"),
    catalog=source("catalog"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_DATASET_MANIFEST_PATHS),
    path_template=value("{build_dir}/serving/artifacts/dataset_manifest_paths.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__dataset_manifest_paths",
)
def serving_artifacts__dataset_manifest_paths(env: BuildEnv, catalog: DagCatalog) -> str:
    """Emit dataset manifest paths for the current snapshot.

    Returns
    -------
    str
        Newline-terminated JSON payload with dataset_manifest_paths.
    """
    dataset_root = env.paths.dataset_root_dir
    snapshot_id = env.commit
    paths: list[str] = []
    for table_key in sorted(catalog.table_outputs):
        manifest_path = dataset_manifest_path(
            dataset_root=dataset_root,
            table_key=table_key,
            snapshot_id=snapshot_id,
        )
        if manifest_path.is_file():
            paths.append(str(manifest_path))
    payload = {"dataset_manifest_paths": paths}
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def serving_artifacts__schema_inference_errors_rows__base(
    env: BuildEnv,
    serving_artifacts__schema_manifest: str,
    schema_index: SchemaIndex,
) -> pl.LazyFrame:
    """Persist schema inference errors recorded during schema compile.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for the schema inference errors table.
    """
    _ = serving_artifacts__schema_manifest
    run_context = env.run_context
    run_id = run_context.run_id if run_context is not None else "unknown"
    buffer = columnar_buffer_for_table_key(SCHEMA_INFERENCE_ERRORS_TABLE_KEY)
    for row in schema_index.iter_inference_error_rows(
        repo=env.repo,
        commit=env.commit,
        run_id=run_id,
    ):
        buffer.append(row)
    return lazyframe_for_table_columns(SCHEMA_INFERENCE_ERRORS_TABLE_KEY, buffer.data)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_BUILDSPEC}"),
    env=source("env"),
    catalog=source("catalog"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_BUILDSPEC),
    path_template=value("{build_dir}/serving/artifacts/buildspec.json"),
)
@tag_compute(
    domain="export",
    target=SERVING_ARTIFACTS_TARGET_NAME,
    target_="serving_artifacts__buildspec",
)
def serving_artifacts__buildspec(env: BuildEnv, catalog: DagCatalog) -> str:
    """Compile BuildSpec JSON for serving.

    Parameters
    ----------
    env
        Build environment (unused; required for Hamilton input binding).
    catalog
        DAG catalog used to render the buildspec.

    Returns
    -------
    str
        Newline-terminated BuildSpec JSON payload.
    """
    _ = env
    return _buildspec_json(catalog)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_ENVIRONMENT}"),
    env=source("env"),
    catalog=source("catalog"),
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


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations_base(
    m__artifact__semantic_registry: MaterializationResult,
    m__artifact__schema_manifest: MaterializationResult,
    m__artifact__buildspec: MaterializationResult,
    m__artifact__dataset_manifest_paths: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect saver metadata for the base serving artifacts.

    Parameters
    ----------
    m__artifact__semantic_registry
        Saver metadata for the semantic registry artifact.
    m__artifact__schema_manifest
        Saver metadata for the schema manifest artifact.
    m__artifact__buildspec
        Saver metadata for the BuildSpec artifact.
    m__artifact__dataset_manifest_paths
        Saver metadata for the dataset manifest paths artifact.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_SEMANTIC_REGISTRY: m__artifact__semantic_registry,
        SERVING_ARTIFACT_SCHEMA_MANIFEST: m__artifact__schema_manifest,
        SERVING_ARTIFACT_BUILDSPEC: m__artifact__buildspec,
        SERVING_ARTIFACT_DATASET_MANIFEST_PATHS: m__artifact__dataset_manifest_paths,
    }


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations_views(
    m__artifact__environment: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect saver metadata for the view/metadata artifacts.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_ENVIRONMENT: m__artifact__environment,
    }


_MODULE = sys.modules[__name__]
_SERVING_ARTIFACTS_TABLE_TARGET_SPEC = TableTargetContext.build_relation_table_spec(
    context=TableTargetContext(
        domain="export",
        target_name=SERVING_ARTIFACTS_TARGET_NAME,
        table_key=SCHEMA_INFERENCE_ERRORS_TABLE_KEY,
        base_node="serving_artifacts__schema_inference_errors_rows__base",
        node_name="serving_artifacts__schema_inference_errors_rows",
        input_type=pl.LazyFrame,
        table_materializations_node="serving_artifacts__table_materializations",
        attach_anchor=False,
    )
)
attach_table_target_template(_MODULE, spec=_SERVING_ARTIFACTS_TABLE_TARGET_SPEC)
serving_artifacts__schema_inference_errors_rows = (
    _MODULE.serving_artifacts__schema_inference_errors_rows
)
serving_artifacts__table_materializations = _MODULE.serving_artifacts__table_materializations


@tag_helper(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def serving_artifacts__materializations(
    serving_artifacts__materializations_base: dict[str, MaterializationResult],
    serving_artifacts__materializations_views: dict[str, MaterializationResult],
) -> dict[str, MaterializationResult]:
    """Merge all serving artifact materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of artifact name to saver metadata.
    """
    merged = dict(serving_artifacts__materializations_base)
    merged.update(serving_artifacts__materializations_views)
    return merged


@codeintel_target(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
def t__serving_artifacts(
    env: BuildEnv,
    catalog: DagCatalog,
    serving_artifacts__materializations: dict[str, MaterializationResult],
    serving_artifacts__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Compile deterministic serving artifacts (semantic registry, schema manifest, buildspec).

    Parameters
    ----------
    env
        Build environment for manifest persistence.
    catalog
        DAG catalog containing the serving_artifacts contract.
    serving_artifacts__materializations
        Saver metadata mapping keyed by artifact name.
    serving_artifacts__table_materializations
        Saver metadata mapping keyed by table name.

    Returns
    -------
    TargetRunRecord
        Record describing the artifact materialization outcome.
    """
    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SERVING_ARTIFACTS_TARGET_NAME,
        ),
        artifact_materializations=serving_artifacts__materializations,
        table_materializations=serving_artifacts__table_materializations,
    )


__all__ = [
    "t__serving_artifacts",
]
