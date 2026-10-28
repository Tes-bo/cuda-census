# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Stdout contract: one `name=integer` line per record, nothing else."""

from __future__ import annotations

import re

KEY_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_LINE = re.compile(rf"^({KEY_PATTERN})=(-?[0-9]+)$")


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


def diffs(gold: dict[str, int], actual: dict[str, int]) -> dict[str, int | None]:
    """Map every key that disagrees to actual - gold.

    A key that only one side printed maps to None, so a missing key is never
    reported as a numeric difference.
    """
    out: dict[str, int | None] = {}
    for name in sorted(set(gold) | set(actual)):
        if name not in gold or name not in actual:
            out[name] = None
        elif gold[name] != actual[name]:
            out[name] = actual[name] - gold[name]
    return out
