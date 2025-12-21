"""Serialization helpers for OutputTarget metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.parameters import TargetParameters
from codeintel.build.resources import TargetExecution, TargetResources
from codeintel.build.targets import OutputTarget, TargetModule
from codeintel.core.schemas.serde import table_schema_from_json_obj, table_schema_to_json_obj


def _artifact_to_json_obj(artifact: ArtifactSpec) -> dict[str, object]:
    return {
        "name": artifact.name,
        "path_template": artifact.path_template,
        "description": artifact.description,
        "required": artifact.required,
    }


def _artifact_from_json_obj(obj: Mapping[str, object]) -> ArtifactSpec:
    name = obj.get("name")
    path_template = obj.get("path_template")
    if not isinstance(name, str) or not isinstance(path_template, str):
        msg = "ArtifactSpec requires name and path_template"
        raise TypeError(msg)
    description = obj.get("description")
    required = obj.get("required", True)
    return ArtifactSpec(
        name=name,
        path_template=path_template,
        description=description if isinstance(description, str) else None,
        required=bool(required) if isinstance(required, bool) else True,
    )


def _output_contract_to_json_obj(contract: OutputContract) -> dict[str, object]:
    return {
        "tables": [table_schema_to_json_obj(schema) for schema in contract.tables],
        "artifacts": [_artifact_to_json_obj(artifact) for artifact in contract.artifacts],
        "json_schema_ids": list(contract.json_schema_ids),
        "jsonl_filenames": list(contract.jsonl_filenames),
        "parquet_filenames": list(contract.parquet_filenames),
        "owner": contract.owner,
        "description": contract.description,
        "family": contract.family,
        "freshness_sla": contract.freshness_sla,
        "retention_policy": contract.retention_policy,
        "upstream_dependencies": list(contract.upstream_dependencies),
        "tags": sorted(contract.tags),
        "validation_profile": contract.validation_profile,
    }


def _output_contract_from_json_obj(obj: Mapping[str, object]) -> OutputContract:
    tables_raw = obj.get("tables", [])
    artifacts_raw = obj.get("artifacts", [])
    if not isinstance(tables_raw, list) or not isinstance(artifacts_raw, list):
        msg = "OutputContract tables/artifacts must be lists"
        raise TypeError(msg)
    tables = tuple(
        table_schema_from_json_obj(item) for item in tables_raw if isinstance(item, Mapping)
    )
    artifacts = tuple(
        _artifact_from_json_obj(item) for item in artifacts_raw if isinstance(item, Mapping)
    )
    json_schema_ids = _tuple_of_str(obj.get("json_schema_ids", []))
    jsonl_filenames = _tuple_of_str(obj.get("jsonl_filenames", []))
    parquet_filenames = _tuple_of_str(obj.get("parquet_filenames", []))
    upstream = _tuple_of_str(obj.get("upstream_dependencies", []))
    tags_raw = obj.get("tags", [])
    tags = frozenset(tags_raw) if isinstance(tags_raw, list) else frozenset()
    validation_profile_raw = obj.get("validation_profile", "strict")
    if validation_profile_raw == "lenient":
        validation_profile: Literal["strict", "lenient"] = "lenient"
    else:
        validation_profile = "strict"
    return OutputContract(
        tables=tables,
        artifacts=artifacts,
        json_schema_ids=json_schema_ids,
        jsonl_filenames=jsonl_filenames,
        parquet_filenames=parquet_filenames,
        owner=_as_optional_str(obj.get("owner")),
        description=_as_optional_str(obj.get("description")),
        family=_as_optional_str(obj.get("family")),
        freshness_sla=_as_optional_str(obj.get("freshness_sla")),
        retention_policy=_as_optional_str(obj.get("retention_policy")),
        upstream_dependencies=upstream,
        tags=tags,
        validation_profile=validation_profile,
    )


def _target_resources_to_json_obj(resources: TargetResources) -> dict[str, object]:
    return {
        "tracker": resources.tracker,
        "modules": resources.modules,
        "gateway": resources.gateway,
        "tools": list(resources.tools),
    }


def _target_resources_from_json_obj(obj: Mapping[str, object]) -> TargetResources:
    tools = _tuple_of_str(obj.get("tools", []))
    return TargetResources(
        tracker=_as_bool(obj.get("tracker"), default=False),
        modules=_as_bool(obj.get("modules"), default=False),
        gateway=_as_bool(obj.get("gateway"), default=True),
        tools=tools,
    )


def _target_execution_to_json_obj(execution: TargetExecution) -> dict[str, object]:
    return {
        "cpu_intensive": execution.cpu_intensive,
        "io_intensive": execution.io_intensive,
        "memory_intensive": execution.memory_intensive,
        "max_runtime_ms": execution.max_runtime_ms,
        "isolation": execution.isolation,
        "supports_incremental": execution.supports_incremental,
        "max_parallelism": execution.max_parallelism,
    }


def _target_execution_from_json_obj(obj: Mapping[str, object]) -> TargetExecution:
    isolation_raw = obj.get("isolation")
    if isolation_raw == "none":
        isolation: Literal["none", "thread", "process"] = "none"
    elif isolation_raw == "process":
        isolation = "process"
    else:
        isolation = "thread"
    max_parallelism = obj.get("max_parallelism")
    return TargetExecution(
        cpu_intensive=_as_bool(obj.get("cpu_intensive"), default=False),
        io_intensive=_as_bool(obj.get("io_intensive"), default=False),
        memory_intensive=_as_bool(obj.get("memory_intensive"), default=False),
        max_runtime_ms=_as_int(obj.get("max_runtime_ms"), default=60000),
        isolation=isolation,
        supports_incremental=_as_bool(obj.get("supports_incremental"), default=True),
        max_parallelism=max_parallelism if isinstance(max_parallelism, int) else None,
    )


def _target_parameters_to_json_obj(parameters: TargetParameters) -> dict[str, object]:
    return dict(parameters)


def _target_parameters_from_json_obj(obj: Mapping[str, object]) -> TargetParameters:
    values = {str(key): value for key, value in obj.items()}
    return TargetParameters(values)


def output_target_to_json_obj(target: OutputTarget) -> dict[str, object]:
    """Serialize an OutputTarget into a JSON object.

    Returns
    -------
    dict[str, object]
        JSON-serializable representation of the output target.
    """
    return {
        "name": target.name,
        "module": target.module,
        "description": target.description,
        "dependencies": list(target.dependencies),
        "contract": _output_contract_to_json_obj(target.contract),
        "resources": _target_resources_to_json_obj(target.resources),
        "execution": _target_execution_to_json_obj(target.execution),
        "parameters": _target_parameters_to_json_obj(target.parameters),
    }


def output_target_from_json_obj(obj: Mapping[str, object]) -> OutputTarget:
    """Parse an OutputTarget from a JSON object.

    Returns
    -------
    OutputTarget
        Parsed output target instance.

    Raises
    ------
    TypeError
        If required fields are missing or of the wrong type.
    """
    name = obj.get("name")
    module_raw = obj.get("module")
    if not isinstance(name, str) or not isinstance(module_raw, str):
        msg = "OutputTarget requires name and module"
        raise TypeError(msg)
    if module_raw == "ingestion":
        module: TargetModule = "ingestion"
    elif module_raw == "graphs":
        module = "graphs"
    elif module_raw == "analytics":
        module = "analytics"
    elif module_raw == "export":
        module = "export"
    else:
        msg = (
            "OutputTarget module must be one of ingestion/graphs/analytics/export, "
            f"got {module_raw}"
        )
        raise TypeError(msg)
    deps = _tuple_of_str(obj.get("dependencies", []))
    contract_raw = obj.get("contract")
    resources_raw = obj.get("resources")
    execution_raw = obj.get("execution")
    parameters_raw = obj.get("parameters")
    if not isinstance(contract_raw, Mapping):
        msg = "OutputTarget requires contract object"
        raise TypeError(msg)
    if not isinstance(resources_raw, Mapping):
        msg = "OutputTarget requires resources object"
        raise TypeError(msg)
    if not isinstance(execution_raw, Mapping):
        msg = "OutputTarget requires execution object"
        raise TypeError(msg)
    if not isinstance(parameters_raw, Mapping):
        msg = "OutputTarget requires parameters object"
        raise TypeError(msg)
    return OutputTarget(
        name=name,
        module=module,
        contract=_output_contract_from_json_obj(contract_raw),
        dependencies=deps,
        resources=_target_resources_from_json_obj(resources_raw),
        execution=_target_execution_from_json_obj(execution_raw),
        parameters=_target_parameters_from_json_obj(parameters_raw),
        description=_as_optional_str(obj.get("description")) or "",
    )


def _tuple_of_str(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _as_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


__all__ = [
    "output_target_from_json_obj",
    "output_target_to_json_obj",
]
