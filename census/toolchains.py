# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Official toolchain command lines. cuda-retarget is not one of them."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_TOOLCHAIN = "nvcc"
FORBIDDEN_NAMES = ("cuda-retarget", "retarget")
SLOTS = ("source", "binary", "translated", "copied_source")
PROFILE_KEYS = ("role", "compile", "translate", "translate_stdout", "copy_source_to")
_SLOT = re.compile(r"\{([A-Za-z_]+)\}")


class ToolchainError(Exception):
    """The requested command line is not an official toolchain."""


def load_profiles() -> dict:
    return json.loads((ROOT / "toolchains.json").read_text(encoding="utf-8"))


def official_names() -> list[str]:
    return list(load_profiles())


def _reject(text: str) -> None:
    if Path(text).name in FORBIDDEN_NAMES or "cuda-retarget" in text:
        raise ToolchainError(
            f"{text} is not an official toolchain; cuda-census does not call cuda-retarget"
        )


def which_allowed(argv0: str) -> str | None:
    """Resolve argv0 on PATH.

    The name, the PATH hit, and the file a symlink points to are all checked,
    so a shim directory such as /opt/cuda-retarget/bin cannot stand in for an
    official binary.
    """
    _reject(argv0)
    found = shutil.which(argv0)
    if found is None:
        return None
    _reject(found)
    _reject(os.path.realpath(found))
    return found


def fill(template: str, slots: dict[str, str]) -> str:
    """Replace each {slot} in one argument. Unknown slots are left as written."""
    text = _SLOT.sub(lambda m: slots.get(m.group(1), m.group(0)), template)
    _reject(text)
    return text


def render(argv: list[str], slots: dict[str, str]) -> list[str]:
    return [fill(part, slots) for part in argv]


def profile_problems(profiles: dict) -> list[str]:
    """Describe every way toolchains.json breaks the profile contract."""
    problems: list[str] = []
    if GOLD_TOOLCHAIN not in profiles:
        problems.append(f"toolchains.json has no {GOLD_TOOLCHAIN} profile")
    for name, profile in profiles.items():
        where = f"toolchains.json {name}"
        try:
            _reject(name)
        except ToolchainError as exc:
            problems.append(f"{where}: {exc}")
        if not isinstance(profile, dict):
            problems.append(f"{where}: not an object")
            continue
        for key in profile:
            if key not in PROFILE_KEYS:
                problems.append(f"{where}: unknown key {key}")
        want_role = "reference" if name == GOLD_TOOLCHAIN else "vendor"
        if profile.get("role") != want_role:
            problems.append(f"{where}: role is {profile.get('role')!r}, expected {want_role!r}")
        if "compile" not in profile:
            problems.append(f"{where}: no compile command")
        if "translate_stdout" in profile and "translate" not in profile:
            problems.append(f"{where}: translate_stdout without translate")
        templates: list[str] = []
        for key in ("compile", "translate"):
            if key not in profile:
                continue
            argv = profile[key]
            if not isinstance(argv, list) or not argv or not all(isinstance(p, str) for p in argv):
                problems.append(f"{where}: {key} must be a non-empty list of strings")
                continue
            templates.extend(argv)
        for key in ("translate_stdout", "copy_source_to"):
            if key not in profile:
                continue
            if not isinstance(profile[key], str):
                problems.append(f"{where}: {key} must be a string")
                continue
            templates.append(profile[key])
        for template in templates:
            for slot in _SLOT.findall(template):
                if slot not in SLOTS:
                    problems.append(f"{where}: unknown slot {{{slot}}}")
            try:
                _reject(template)
            except ToolchainError as exc:
                problems.append(f"{where}: {exc}")
    return problems
