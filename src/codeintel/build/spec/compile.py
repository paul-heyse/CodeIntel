"""BuildSpec compiler.

Compiles a deterministic BuildSpec from the canonical target catalog.
The BuildSpec is the DAG-first compiled contract used for CI gating and (later)
serving metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.output_inventory import OutputInventory
from codeintel.build.schemas import get_schema_provider
from codeintel.build.spec.primitives import ArtifactOutSpec, BuildSpec, DatasetSpec, TargetSpec
from codeintel.build.spec.serdes import ensure_buildspec_hash
from codeintel.build.target_catalog import target_graph_from_catalog
from codeintel.build.target_inventory import get_output_inventory
from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.targets import OutputTarget, TargetGraph
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


@dataclass(frozen=True)
class BuildSpecCompileOptions:
    """Options for compiling a BuildSpec."""

    include_columns: bool = False
    spec_version: int = 1


def _artifact_specs_for_target(
    target: OutputTarget,
    derived_outputs: OutputInventory,
    *,
    artifact_names: Iterable[str],
) -> tuple[ArtifactOutSpec, ...]:
    """Build ArtifactOutSpec collection for a build target.

    Parameters
    ----------
    target
        OutputTarget with a contract that may declare artifacts.
    derived_outputs
        DAG-derived output inventory with artifact templates per target.
    artifact_names
        Names of artifacts to include.

    Returns
    -------
    tuple[ArtifactOutSpec, ...]
        Artifact output specifications for the target.
    """
    specs: list[ArtifactOutSpec] = []
    templates = derived_outputs.artifact_templates_for(target.name)
    for artifact_name in artifact_names:
        artifact = target.contract.get_artifact(artifact_name)
        template = templates.get(artifact_name)
        specs.append(
            ArtifactOutSpec(
                name=artifact_name,
                kind=None,
                path_template=template or (artifact.path_template if artifact is not None else None),
            )
        )
    return tuple(sorted(specs, key=lambda a: a.name))


def _compile_target_specs(
    *,
    graph: TargetGraph,
    derived_outputs: OutputInventory,
) -> tuple[tuple[TargetSpec, ...], frozenset[str]]:
    """Compile TargetSpec collection and table_key inventory.

    Parameters
    ----------
    graph
        Target graph describing target dependencies.
    derived_outputs
        Derived output table keys and artifact names by target.

    Returns
    -------
    tuple[tuple[TargetSpec, ...], frozenset[str]]
        Compiled TargetSpecs and the set of produced dataset table keys.

    """
    all_table_keys: set[str] = set()
    target_specs: list[TargetSpec] = []
    for target_name in sorted(graph):
        target = graph.get(target_name)
        impl_kind = "native"

        outputs = derived_outputs.datasets_for(target_name)
        artifacts = derived_outputs.artifacts_for(target_name)

        all_table_keys.update(outputs)

        target_specs.append(
            TargetSpec(
                name=target_name,
                domain=target.module,
                impl_kind=impl_kind,
                deps=target.dependencies,
                outputs=tuple(outputs),
                artifacts=_artifact_specs_for_target(
                    target,
                    derived_outputs,
                    artifact_names=artifacts,
                ),
            )
        )

    return tuple(sorted(target_specs, key=lambda t: t.name)), frozenset(all_table_keys)


def _compile_dataset_specs(
    *,
    provider: SchemaProvider,
    table_keys: Iterable[str],
    include_columns: bool,
) -> tuple[DatasetSpec, ...]:
    """Compile DatasetSpec collection from table keys.

    Parameters
    ----------
    provider
        Schema provider used to resolve TableSchema definitions.
    table_keys
        Produced dataset table keys.
    include_columns
        When True, include column names in the dataset specs.

    Returns
    -------
    tuple[DatasetSpec, ...]
        Compiled dataset specs in deterministic order.
    """
    specs: list[DatasetSpec] = []
    for table_key in sorted(table_keys):
        table_schema: TableSchema = provider.require_table_schema(table_key)
        columns = tuple(table_schema.column_names()) if include_columns else None
        specs.append(
            DatasetSpec(
                table_key=table_key,
                schema_hash=schema_hash(table_schema),
                columns=columns,
            )
        )
    return tuple(specs)


def compile_buildspec(*, options: BuildSpecCompileOptions | None = None) -> BuildSpec:
    """Compile a BuildSpec from the canonical target catalog.

    Parameters
    ----------
    options
        Compilation options controlling output detail.

    Returns
    -------
    BuildSpec
        Compiled BuildSpec with deterministic ordering and populated hash.
    """
    opts = options or BuildSpecCompileOptions()

    graph = target_graph_from_catalog()
    derived_outputs = get_output_inventory()

    provider = get_schema_provider()
    target_specs, all_table_keys = _compile_target_specs(
        graph=graph,
        derived_outputs=derived_outputs,
    )
    dataset_specs = _compile_dataset_specs(
        provider=provider,
        table_keys=all_table_keys,
        include_columns=opts.include_columns,
    )

    spec = BuildSpec(
        spec_version=opts.spec_version,
        targets=target_specs,
        datasets=dataset_specs,
        semantic=None,
    )
    return ensure_buildspec_hash(spec)


__all__ = [
    "BuildSpecCompileOptions",
    "compile_buildspec",
]
