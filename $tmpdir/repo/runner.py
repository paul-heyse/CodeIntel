"""Lightweight driver to execute tooling fixture functions."""

from contextlib import suppress

from pkg import mod

mod.add(1, 2)
with suppress(Exception):
    mod.bad_type(1)
