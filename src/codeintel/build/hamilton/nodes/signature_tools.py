"""Internal signature helpers for dynamic Hamilton node generation."""

from __future__ import annotations

import inspect
from typing import Protocol, cast


class _SignaturedCallable(Protocol):
    """Protocol for callables with mutable signature/annotation metadata."""

    __signature__: inspect.Signature
    __annotations__: dict[str, object]


def set_signature[TCallable](fn: TCallable, signature: inspect.Signature) -> TCallable:
    """Attach an inspect.Signature to a callable for Hamilton compatibility.

    Parameters
    ----------
    fn
        Callable object to annotate.
    signature
        Signature that Hamilton should use for dependency resolution.

    Returns
    -------
    TCallable
        The input callable with signature metadata applied.
    """
    meta = cast("_SignaturedCallable", fn)
    meta.__signature__ = signature

    annotations: dict[str, object] = dict(getattr(fn, "__annotations__", {}))
    for name, param in signature.parameters.items():
        if param.annotation is inspect.Signature.empty:
            continue
        annotations[name] = param.annotation
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    meta.__annotations__ = annotations
    return fn


__all__ = ["set_signature"]
