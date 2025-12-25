"""Canonical MaterializeOptions construction for build targets."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.build.hamilton.env import BuildEnv
from codeintel.storage.warehouse import MaterializeOptions, ReplaceScope, UpsertConfig, WriteMode


@dataclass(frozen=True, slots=True)
class MaterializeOptionsConfig:
    """Configuration for MaterializeOptions derivation.

    Parameters
    ----------
    mode
        Warehouse write mode.
    replace_scope
        Scope for replace operations.
    input_hash
        Optional manifest input hash to attach to the write (for observability).
    upsert
        Upsert configuration when using upsert mode.
    use_staging
        Whether to use a staging relation for writes.
    """

    mode: WriteMode = "replace"
    replace_scope: ReplaceScope = "snapshot"
    input_hash: str | None = None
    upsert: UpsertConfig | None = None
    use_staging: bool = False


def materialize_options(
    env: BuildEnv,
    *,
    owner_target: str,
    config: MaterializeOptionsConfig | None = None,
) -> MaterializeOptions:
    """Build a consistent MaterializeOptions for snapshot-scoped table writes.

    Parameters
    ----------
    env
        Build environment providing snapshot.
    owner_target
        Target name that owns the materialization.
    config
        Optional configuration overrides for write mode and upsert behavior.

    Returns
    -------
    MaterializeOptions
        Options object with snapshot and owner_target populated.
    """
    resolved = config or MaterializeOptionsConfig()
    return MaterializeOptions(
        snapshot=env.snapshot,
        mode=resolved.mode,
        replace_scope=resolved.replace_scope,
        owner_target=owner_target,
        input_hash=resolved.input_hash,
        upsert=resolved.upsert,
        use_staging=resolved.use_staging,
    )


def append_materialize_options(options: MaterializeOptions) -> MaterializeOptions:
    """Return append-mode options derived from an existing options object.

    This is used by validated materializers that perform an explicit
    delete-for-snapshot and then need to append the validated rows.

    Parameters
    ----------
    options
        Existing options (typically snapshot-scoped replace-mode options).

    Returns
    -------
    MaterializeOptions
        Options object with the same snapshot/owner_target/input_hash but with
        mode set to ``"append"``.
    """
    return MaterializeOptions(
        snapshot=options.snapshot,
        mode="append",
        replace_scope=options.replace_scope,
        owner_target=options.owner_target,
        input_hash=options.input_hash,
        asset_type=options.asset_type,
        upsert=options.upsert,
        use_staging=options.use_staging,
    )


__all__ = [
    "MaterializeOptionsConfig",
    "append_materialize_options",
    "materialize_options",
]
