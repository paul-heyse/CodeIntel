"""Msgspec schemas for wiring packs."""

from __future__ import annotations

import msgspec

from tools.advanced_query_engine.contracts import JSONValue


class RpygrepStage(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="rpygrep",
    tag_field="engine",
):
    """Rpygrep stage definition."""

    pattern_group_file: str
    preset: str | None = None
    purpose: str | None = None


class AstGrepStage(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="ast_grep",
    tag_field="engine",
):
    """Ast-grep stage definition."""

    rules_file: str
    rule_ids: list[str] = []


WiringStage = RpygrepStage | AstGrepStage


class UnquoteCaptureOp(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="python.unquote_capture",
    tag_field="op",
):
    """Postprocess op for unquoting captures."""

    capture_names: list[str]
    output_field_suffix: str | None = None


class NormalizeHttpMethodOp(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="python.normalize_http_method",
    tag_field="op",
):
    """Postprocess op for normalizing HTTP methods."""

    capture_name: str | None = None
    output_field: str | None = None


class UpperCaptureOp(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="python.upper_capture",
    tag_field="op",
):
    """Postprocess op for uppercasing captures."""

    capture_name: str
    output_field: str | None = None


class JoinArgparseSubcommandsOp(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    tag="python.join_argparse_subcommands",
    tag_field="op",
):
    """Postprocess op for argparse subcommand joining."""

    add_parser_rule_id: str | None = None
    set_defaults_rule_id: str | None = None
    subparser_var_capture: str | None = None
    command_capture: str | None = None
    handler_capture: str | None = None


PostprocessOp = (
    UnquoteCaptureOp | NormalizeHttpMethodOp | UpperCaptureOp | JoinArgparseSubcommandsOp
)


class EmitConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Emit configuration for wiring packs."""

    entry_key_template: str
    entry_key_by_rule: dict[str, str] | None = None
    hook_span_capture: str | None = None
    hook_span_by_rule: dict[str, str] | None = None
    target_symbol_hint_capture: str | None = None
    target_symbol_hint_by_rule: dict[str, str] | None = None


class WiringPack(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Wiring pack schema."""

    pack_id: str
    entry_kind: str
    stages: list[WiringStage]
    emit: EmitConfig
    framework: str | None = None
    description: str | None = None
    postprocess: list[PostprocessOp] | None = None


def wiring_pack_schema() -> dict[str, JSONValue]:
    """Return the JSON schema for wiring packs.

    Returns
    -------
    dict[str, JSONValue]
        JSON schema payload.
    """
    return msgspec.json.schema(WiringPack)


__all__ = [
    "AstGrepStage",
    "EmitConfig",
    "JoinArgparseSubcommandsOp",
    "NormalizeHttpMethodOp",
    "PostprocessOp",
    "RpygrepStage",
    "UnquoteCaptureOp",
    "UpperCaptureOp",
    "WiringPack",
    "WiringStage",
    "wiring_pack_schema",
]
