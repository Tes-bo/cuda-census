# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Official toolchain command lines. cuda-retarget is not one of them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_NAMES = ("cuda-retarget", "retarget")


class ToolchainError(Exception):
    """The requested command line is not an official toolchain."""


def load_profiles() -> dict:
    return json.loads((ROOT / "toolchains.json").read_text(encoding="utf-8"))


def official_names() -> list[str]:
    return list(load_profiles())


def _reject(argv0: str) -> None:
    base = Path(argv0).name
    if base in FORBIDDEN_NAMES or "cuda-retarget" in argv0:
        raise ToolchainError(
            f"{argv0} is not an official toolchain; cuda-census does not call cuda-retarget"
        )


def which_allowed(argv0: str) -> str | None:
    _reject(argv0)
    return shutil.which(argv0)


def render(argv: list[str], slots: dict[str, str]) -> list[str]:
    out = []
    for part in argv:
        text = part
        for key, value in slots.items():
            text = text.replace("{" + key + "}", value)
        _reject(text)
        out.append(text)
    return out
