"""Entry point loader for the example target pack."""

from __future__ import annotations

from codeintel.runtime.plugins.spec import TargetPack, TargetPackModule


def target_pack() -> TargetPack:
    """Return the target pack descriptor.

    Returns
    -------
    TargetPack
        Target pack metadata.
    """
    return TargetPack(
        name="hello_pack",
        version="0.1.0",
        modules=(TargetPackModule(import_path="hello_pack.targets"),),
        requires_codeintel=">=0.0.0",
        default_enabled=False,
        config_namespace="hello_pack",
        capabilities=frozenset({"example"}),
    )


__all__ = ["target_pack"]
