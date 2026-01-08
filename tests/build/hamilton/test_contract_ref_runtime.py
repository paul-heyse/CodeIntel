"""ContractRef and ContractRuntime behavior tests."""

from __future__ import annotations

from types import ModuleType

from codeintel.build.contracts.ref import ContractRef, contract_ref_for_table
from codeintel.build.contracts.runtime import ContractRuntime, configure_contract_runtime
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.core.schemas import clear_schema_service, get_schema_service, set_schema_service
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.service import SchemaService
from codeintel.core.schemas.table_registry import TABLE_SCHEMAS


def _schema_service() -> SchemaService:
    return SchemaService(table_provider=MappingSchemaProvider(dict(TABLE_SCHEMAS)))


def _capture_schema_service() -> SchemaService | None:
    try:
        return get_schema_service()
    except RuntimeError:
        return None


def _restore_schema_service(service: SchemaService | None) -> None:
    if service is None:
        clear_schema_service()
        return
    set_schema_service(service)


def test_contract_ref_for_table_is_schema_safe() -> None:
    """Ensure ContractRef creation does not require SchemaService."""
    previous = _capture_schema_service()
    clear_schema_service()
    try:
        ref = contract_ref_for_table(
            table_key="analytics.function_types",
            target_name="contract_ref_test",
            input_name="function_types__base",
        )
    finally:
        _restore_schema_service(previous)
    assert isinstance(ref, ContractRef)
    assert ref.table_key == "analytics.function_types"


def test_contract_runtime_resolves_contract_ref() -> None:
    """Ensure ContractRuntime resolves refs into concrete contract specs."""
    runtime = ContractRuntime(schema_service=_schema_service())
    ref = contract_ref_for_table(
        table_key="analytics.function_types",
        target_name="contract_ref_test",
        input_name="function_types__base",
    )
    spec = runtime.resolve(ref)
    assert spec.table_key == ref.table_key
    assert spec.target == ref.target_name
    assert spec.input_name == ref.input_name
    assert spec.contract_hash is not None


def test_attach_table_target_template_resolves_ref() -> None:
    """Ensure table targets can attach with ContractRef inputs."""
    configure_contract_runtime(schema_service=_schema_service())
    ref = contract_ref_for_table(
        table_key="analytics.function_types",
        target_name="contract_ref_target",
        input_name="function_types__base",
    )
    context = TableTargetContext.from_contract_ref(contract_ref=ref)
    spec = build_single_table_target_spec(context=context)
    module = ModuleType("tests.contract_ref_module")
    attach_table_target_template(module, spec=spec)
    assert hasattr(module, "contract_ref_target__table")
