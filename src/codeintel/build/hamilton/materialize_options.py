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


__all__ = [
    "materialize_options",
]

