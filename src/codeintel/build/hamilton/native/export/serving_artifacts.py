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
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

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
from codeintel.build.schemas import SchemaManifest, get_schema_provider
from codeintel.build.serving.semantic_compile import compile_semantic_registry_from_views
from codeintel.build.serving.semantic_compile_hamilton import (
    collect_semantic_view_tags_from_hamilton,
)
from codeintel.build.spec import BuildSpecCompileOptions, compile_buildspec
from codeintel.build.spec.serdes import buildspec_to_json
from codeintel.build.targets import TargetGraph
from codeintel.storage.views import ibis_views as _ibis_views

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

SERVING_ARTIFACTS_TARGET_NAME = "serving_artifacts"

SERVING_ARTIFACT_SEMANTIC_REGISTRY = "semantic_registry"
SERVING_ARTIFACT_SCHEMA_MANIFEST = "schema_manifest"
SERVING_ARTIFACT_BUILDSPEC = "buildspec"

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


@SaveToDecorator(
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


@SaveToDecorator(
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


@SaveToDecorator(
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


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="helper")
def serving_artifacts__materializations(
    m__artifact__semantic_registry: dict[str, Any],
    m__artifact__schema_manifest: dict[str, Any],
    m__artifact__buildspec: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect saver metadata for all serving artifacts.

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
    dict[str, dict[str, Any]]
        Mapping of artifact name to saver metadata.
    """
    return {
        SERVING_ARTIFACT_SEMANTIC_REGISTRY: m__artifact__semantic_registry,
        SERVING_ARTIFACT_SCHEMA_MANIFEST: m__artifact__schema_manifest,
        SERVING_ARTIFACT_BUILDSPEC: m__artifact__buildspec,
    }


@tag(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME, node_type="materialize")
def t__serving_artifacts(
    env: BuildEnv,
    graph: TargetGraph,
    serving_artifacts__materializations: dict[str, dict[str, Any]],
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
