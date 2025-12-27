"""BuildSpec compiler.

Compiles a deterministic BuildSpec from the Hamilton DAG-derived target graph.
The BuildSpec is the DAG-first compiled contract used for CI gating and serving metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.dag_catalog import DagCatalog, OutputDescriptor
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.schemas import get_schema_provider
from codeintel.build.spec.primitives import ArtifactOutSpec, BuildSpec, DatasetSpec, TargetSpec
from codeintel.build.spec.serdes import ensure_buildspec_hash
from codeintel.core.schemas.hashing import schema_hash

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


@dataclass(frozen=True)
class BuildSpecCompileOptions:
    """Options for compiling a BuildSpec."""

    include_columns: bool = False
    spec_version: int = 1


def _artifact_specs_for_target(
    outputs: Iterable[OutputDescriptor],
    *,
    artifact_names: Iterable[str],
) -> tuple[ArtifactOutSpec, ...]:
    """Build ArtifactOutSpec collection for a build target.

    Parameters
    ----------
    outputs
        DAG-derived output inventory for this target.
    artifact_names
        Names of artifacts to include.

    Returns
    -------
    tuple[ArtifactOutSpec, ...]
        Artifact output specifications for the target.

    Raises
    ------
    ValueError
        If a referenced artifact is missing its path template.
    """
    specs: list[ArtifactOutSpec] = []
    templates = {output.key: output.artifact_path_template for output in outputs}
    for artifact_name in artifact_names:
        template = templates.get(artifact_name)
        if template is None:
            msg = f"Missing artifact template for {artifact_name}"
            raise ValueError(msg)
        specs.append(
            ArtifactOutSpec(
                name=artifact_name,
                kind=None,
                path_template=template,
            )
        )
    return tuple(sorted(specs, key=lambda a: a.name))


def _compile_target_specs(
    *,
    catalog: DagCatalog,
) -> tuple[tuple[TargetSpec, ...], frozenset[str]]:
    """Compile TargetSpec collection and table_key inventory.

    Parameters
    ----------
    catalog
        DAG catalog describing targets and outputs.

    Returns
    -------
    tuple[tuple[TargetSpec, ...], frozenset[str]]
        Compiled TargetSpecs and the set of produced dataset table keys.

    """
    all_table_keys: set[str] = set()
    target_specs: list[TargetSpec] = []
    for target_name in sorted(catalog.targets):
        target = catalog.get(target_name)
        impl_kind = "native"

        outputs = tuple(
            output.key for output in catalog.table_outputs_by_target.get(target_name, ())
        )
        artifacts = tuple(
            output.key for output in catalog.artifact_outputs_by_target.get(target_name, ())
        )

        all_table_keys.update(outputs)

        target_specs.append(
            TargetSpec(
                name=target_name,
                domain=target.module,
                impl_kind=impl_kind,
                deps=target.dependencies,
                outputs=tuple(sorted(outputs)),
                artifacts=_artifact_specs_for_target(
                    catalog.artifact_outputs_by_target.get(target_name, ()),
                    artifact_names=tuple(sorted(artifacts)),
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
    """Compile a BuildSpec from the Hamilton DAG.

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

    runtime = build_driver()
    catalog = runtime.catalog

    provider = get_schema_provider()
    target_specs, all_table_keys = _compile_target_specs(
        catalog=catalog,
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
