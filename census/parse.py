# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Stdout contract: one `name=integer` line per record, nothing else."""

from __future__ import annotations

import re

_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+)$")


def parse_stdout(text: str) -> dict[str, int] | None:
    """Return parsed keys, or None when the contract is broken."""
    keys: dict[str, int] = {}
    body = text.strip("\n")
    if body == "":
        return None
    for line in body.split("\n"):
        match = _LINE.match(line)
        if match is None:
            return None
        name = match.group(1)
        if name in keys:
            return None
        keys[name] = int(match.group(2))
    return keys


def diffs(gold: dict[str, int], actual: dict[str, int]) -> dict[str, int]:
    """Map every key that disagrees to actual - gold."""
    names = sorted(set(gold) | set(actual))
    out: dict[str, int] = {}
    for name in names:
        if gold.get(name) != actual.get(name):
            out[name] = actual.get(name, 0) - gold.get(name, 0)
    return out
