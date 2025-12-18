"""Environment variable helpers for tests.

The test suite bans runtime patching utilities. These context managers provide small, explicit
helpers for temporarily modifying environment variables in a reversible way.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def temporary_env(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable for the duration of the context.

    Parameters
    ----------
    key
        Environment variable name.
    value
        Value to set.

    Yields
    ------
    None
        Control to the caller with the environment modified.
    """
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def unset_env(key: str) -> Iterator[None]:
    """Temporarily unset an environment variable for the duration of the context.

    Parameters
    ----------
    key
        Environment variable name.

    Yields
    ------
    None
        Control to the caller with the environment modified.
    """
    previous = os.environ.get(key)
    os.environ.pop(key, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[key] = previous
