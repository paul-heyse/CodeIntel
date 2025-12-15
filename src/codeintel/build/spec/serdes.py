"""BuildSpec serialization and hashing utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from codeintel.build.spec.primitives import (
    ArtifactOutSpec,
    BuildSpec,
    DatasetSpec,
    SemanticSpec,
    TargetSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.spec.primitives import ImplKind


JsonObject = dict[str, object]


def canonical_json(obj: object) -> str:
    """Serialize an object to canonical JSON for hashing.

    Parameters
    ----------
    obj
        JSON-serializable object.

    Returns
    -------
    str
        Canonical JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def buildspec_hash(spec: BuildSpec) -> str:
    """Compute the BuildSpec hash.

    Parameters
    ----------
    spec
        BuildSpec instance.

    Returns
    -------
    str
        SHA-256 hex digest of canonical JSON excluding the hash field.
    """
    obj = buildspec_to_json_obj(spec, include_hash=False)
    payload = canonical_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_buildspec_hash(spec: BuildSpec) -> BuildSpec:
    """Ensure buildspec_hash is populated and correct.

    Parameters
    ----------
    spec
        BuildSpec instance.

    Returns
    -------
    BuildSpec
        BuildSpec with buildspec_hash field set.
    """
    expected = buildspec_hash(spec)
    if spec.buildspec_hash == expected:
        return spec
    return replace(spec, buildspec_hash=expected)


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def buildspec_to_json_obj(spec: BuildSpec, *, include_hash: bool = True) -> JsonObject:
    """Convert BuildSpec to a JSON object with deterministic ordering.

    Parameters
    ----------
    spec
        BuildSpec instance to serialize.
    include_hash
        When True, include the computed ``buildspec_hash`` field.

    Returns
    -------
    JsonObject
        JSON-serializable mapping with stable ordering.
    """
    targets: list[TargetSpec] = sorted(spec.targets, key=lambda t: t.name)
    datasets: list[DatasetSpec] = sorted(spec.datasets, key=lambda d: d.table_key)

    obj: JsonObject = {
        "spec_version": spec.spec_version,
        "targets": [
            {
                "name": t.name,
                "domain": t.domain,
                "impl_kind": t.impl_kind,
                "deps": list(_sorted_unique(t.deps)),
                "outputs": list(_sorted_unique(t.outputs)),
                "artifacts": [
                    {
                        "name": a.name,
                        "kind": a.kind,
                        "path_template": a.path_template,
                    }
                    for a in sorted(t.artifacts, key=lambda a: a.name)
                ],
            }
            for t in targets
        ],
        "datasets": [
            {
                "table_key": d.table_key,
                "schema_hash": d.schema_hash,
                **({"columns": list(d.columns)} if d.columns is not None else {}),
            }
            for d in datasets
        ],
    }
    if spec.semantic is not None:
        obj["semantic"] = {"version": spec.semantic.version}

    if include_hash:
        obj["buildspec_hash"] = ensure_buildspec_hash(spec).buildspec_hash
    return obj


def buildspec_to_json(spec: BuildSpec, *, indent: int | None = 2) -> str:
    """Serialize BuildSpec to deterministic JSON text.

    Parameters
    ----------
    spec
        BuildSpec instance to serialize.
    indent
        Indentation level for pretty-printed JSON. When None, produces compact canonical JSON.

    Returns
    -------
    str
        Newline-terminated JSON text.
    """
    obj = buildspec_to_json_obj(spec, include_hash=True)
    if indent is None:
        return canonical_json(obj) + "\n"
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


def _expect_dict(value: object, *, ctx: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        msg = f"Expected object for {ctx}"
        raise TypeError(msg)
    return value


def _expect_list(value: object, *, ctx: str) -> list[object]:
    if not isinstance(value, list):
        msg = f"Expected array for {ctx}"
        raise TypeError(msg)
    return value


def _expect_str(value: object, *, ctx: str) -> str:
    if not isinstance(value, str):
        msg = f"Expected string for {ctx}"
        raise TypeError(msg)
    return value


def _expect_int(value: object, *, ctx: str) -> int:
    if not isinstance(value, int):
        msg = f"Expected integer for {ctx}"
        raise TypeError(msg)
    return value


def _parse_semantic(value: object) -> SemanticSpec | None:
    if value is None:
        return None
    obj = _expect_dict(value, ctx="semantic")
    version = obj.get("version")
    if version is None:
        return SemanticSpec(version=None)
    return SemanticSpec(version=_expect_str(version, ctx="semantic.version"))


def _parse_artifact(value: object, *, idx: int) -> ArtifactOutSpec:
    obj = _expect_dict(value, ctx=f"targets[].artifacts[{idx}]")
    name = _expect_str(obj.get("name"), ctx=f"targets[].artifacts[{idx}].name")
    kind_raw = obj.get("kind")
    kind = _expect_str(kind_raw, ctx=f"targets[].artifacts[{idx}].kind") if kind_raw else None
    path_raw = obj.get("path_template")
    path_template = (
        _expect_str(path_raw, ctx=f"targets[].artifacts[{idx}].path_template") if path_raw else None
    )
    return ArtifactOutSpec(name=name, kind=kind, path_template=path_template)


def _parse_target(value: object, *, idx: int) -> TargetSpec:
    obj = _expect_dict(value, ctx=f"targets[{idx}]")
    name = _expect_str(obj.get("name"), ctx=f"targets[{idx}].name")
    domain = _expect_str(obj.get("domain"), ctx=f"targets[{idx}].domain")
    impl_kind = _expect_str(obj.get("impl_kind"), ctx=f"targets[{idx}].impl_kind")
    impl_kind_typed: ImplKind
    if impl_kind == "native":
        impl_kind_typed = "native"
    elif impl_kind == "wrapper":
        impl_kind_typed = "wrapper"
    else:
        msg = f"Unsupported impl_kind: {impl_kind}"
        raise ValueError(msg)

    deps_raw = obj.get("deps", [])
    outputs_raw = obj.get("outputs", [])
    deps = tuple(
        _expect_str(x, ctx=f"targets[{idx}].deps[]")
        for x in _expect_list(deps_raw, ctx=f"targets[{idx}].deps")
    )
    outputs = tuple(
        _expect_str(x, ctx=f"targets[{idx}].outputs[]")
        for x in _expect_list(outputs_raw, ctx=f"targets[{idx}].outputs")
    )

    artifacts_raw = obj.get("artifacts", [])
    artifacts = tuple(
        _parse_artifact(item, idx=a_idx)
        for a_idx, item in enumerate(_expect_list(artifacts_raw, ctx=f"targets[{idx}].artifacts"))
    )

    return TargetSpec(
        name=name,
        domain=domain,
        impl_kind=impl_kind_typed,
        deps=deps,
        outputs=outputs,
        artifacts=artifacts,
    )


def _parse_dataset(value: object, *, idx: int) -> DatasetSpec:
    obj = _expect_dict(value, ctx=f"datasets[{idx}]")
    table_key = _expect_str(obj.get("table_key"), ctx=f"datasets[{idx}].table_key")
    schema_hash = _expect_str(obj.get("schema_hash"), ctx=f"datasets[{idx}].schema_hash")
    columns_raw = obj.get("columns")
    columns: tuple[str, ...] | None
    if columns_raw is None:
        columns = None
    else:
        columns = tuple(
            _expect_str(x, ctx=f"datasets[{idx}].columns[]")
            for x in _expect_list(columns_raw, ctx="columns")
        )
    return DatasetSpec(table_key=table_key, schema_hash=schema_hash, columns=columns)


def buildspec_from_json(text: str) -> BuildSpec:
    """Parse a BuildSpec from JSON text.

    Parameters
    ----------
    text
        JSON text.

    Returns
    -------
    BuildSpec
        Parsed BuildSpec.
    """
    payload = json.loads(text)
    obj = _expect_dict(payload, ctx="buildspec")

    spec_version = _expect_int(obj.get("spec_version"), ctx="spec_version")

    targets_raw = obj.get("targets", [])
    targets = tuple(
        _parse_target(item, idx=idx)
        for idx, item in enumerate(_expect_list(targets_raw, ctx="targets"))
    )

    datasets_raw = obj.get("datasets", [])
    datasets = tuple(
        _parse_dataset(item, idx=idx)
        for idx, item in enumerate(_expect_list(datasets_raw, ctx="datasets"))
    )

    semantic = _parse_semantic(obj.get("semantic"))

    buildspec_hash_raw = obj.get("buildspec_hash")
    buildspec_hash_value = (
        _expect_str(buildspec_hash_raw, ctx="buildspec_hash") if buildspec_hash_raw else ""
    )

    return ensure_buildspec_hash(
        BuildSpec(
            spec_version=spec_version,
            targets=targets,
            datasets=datasets,
            semantic=semantic,
            buildspec_hash=buildspec_hash_value,
        )
    )


__all__ = [
    "JsonObject",
    "buildspec_from_json",
    "buildspec_hash",
    "buildspec_to_json",
    "buildspec_to_json_obj",
    "canonical_json",
    "ensure_buildspec_hash",
]
