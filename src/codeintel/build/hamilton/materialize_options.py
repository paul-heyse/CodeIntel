"""Canonical MaterializeOptions construction for build targets."""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.storage.warehouse import MaterializeOptions, WriteMode


def materialize_options(
    env: BuildEnv,
    *,
    owner_target: str,
    mode: WriteMode = "replace",
    input_hash: str | None = None,
) -> MaterializeOptions:
    """Build a consistent MaterializeOptions for snapshot-scoped table writes.

    Parameters
    ----------
    env
        Build environment providing snapshot.
    owner_target
        Target name that owns the materialization.
    mode
        Warehouse write mode.
    input_hash
        Optional manifest input hash to attach to the write (for observability).

    Returns
    -------
    MaterializeOptions
        Options object with snapshot and owner_target populated.
    """
    return MaterializeOptions(
        snapshot=env.snapshot,
        mode=mode,
        owner_target=owner_target,
        input_hash=input_hash,
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
        owner_target=options.owner_target,
        input_hash=options.input_hash,
        asset_type=options.asset_type,
        upsert=options.upsert,
    )


__all__ = [
    "append_materialize_options",
    "materialize_options",
]
