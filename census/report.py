# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Collect results/<toolchain>/<machine>/<case>.json into one Markdown table."""

from __future__ import annotations

import json
import re

from census import runner
from census.toolchains import GOLD_TOOLCHAIN

NOT_RUN = "not run"

Column = tuple[str, str | None]


def columns(toolchains: list[str]) -> list[Column]:
    """One column per machine directory, or one empty column for a toolchain that never ran."""
    out: list[Column] = []
    for toolchain in toolchains:
        base = runner.RESULTS_DIR / toolchain
        machines = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
        if machines:
            out.extend((toolchain, machine) for machine in machines)
        else:
            out.append((toolchain, None))
    return out


def load_result(toolchain: str, machine: str | None, case_id: str) -> dict | None:
    if machine is None:
        return None
    path = runner.result_path(toolchain, machine, case_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stale_reasons(case: dict, toolchain: str, result: dict) -> list[str]:
    """Explain why a result no longer describes the committed source and gold."""
    reasons = []
    if "source_sha256" not in result:
        reasons.append("written before results recorded the source hash")
    elif result["source_sha256"] != runner.file_sha256(runner.source_path(case)):
        reasons.append("source changed")
    if toolchain != GOLD_TOOLCHAIN and result.get("status") not in ("compile-fail", "toolchain-missing"):
        if result.get("gold_sha256") != runner.file_sha256(runner.gold_path(case["id"])):
            reasons.append("gold changed")
    return reasons


def _label(column: Column) -> str:
    toolchain, machine = column
    return f"`{toolchain}`" if machine is None else f"`{toolchain}` ({machine})"


def _cell(case: dict, toolchain: str, result: dict | None) -> str:
    if result is None:
        return NOT_RUN
    text = result["status"]
    if text == "numeric-mismatch":
        text += f" ({len(result['diffs'])})"
    if stale_reasons(case, toolchain, result):
        text += " (stale)"
    return text


def _diff_text(name: str, diff: int | None) -> str:
    if diff is None:
        return f"`{name}` printed by one side only"
    return f"`{name}` {diff:+d}"


def _version_line(text: str | None) -> str:
    """The first line that carries a dotted version number, else the first line."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines:
        if re.search(r"\d+\.\d+", line):
            return line
    return lines[0] if lines else ""


def _md(text: str) -> str:
    return text.replace("|", "\\|") if text else "-"


def build_report(toolchains: list[str]) -> str:
    cases = runner.load_manifest()["cases"]
    cols = columns(toolchains)
    table = {(case["id"], col): load_result(*col, case["id"]) for case in cases for col in cols}

    lines = [
        "| Case | " + " | ".join(_label(col) for col in cols) + " |",
        "| --- |" + " --- |" * len(cols),
    ]
    for case in cases:
        cells = [_cell(case, col[0], table[case["id"], col]) for col in cols]
        lines.append(f"| `{case['id']}` | " + " | ".join(cells) + " |")
    passed = []
    for col in cols:
        count = sum(
            1
            for case in cases
            if (r := table[case["id"], col]) is not None
            and r["status"] == "pass"
            and not stale_reasons(case, col[0], r)
        )
        passed.append(f"{count}/{len(cases)}")
    lines.append("| **pass** | " + " | ".join(passed) + " |")

    notes = []
    for case in cases:
        for col in cols:
            result = table[case["id"], col]
            if result is None:
                continue
            reasons = stale_reasons(case, col[0], result)
            if reasons:
                notes.append(f"- {_label(col)} `{case['id']}` is stale: {', '.join(reasons)}.")
            if result["status"] == "numeric-mismatch":
                diffs = ", ".join(_diff_text(k, v) for k, v in sorted(result["diffs"].items()))
                notes.append(f"- {_label(col)} `{case['id']}` differs from gold (actual minus gold): {diffs}.")
    if notes:
        lines += ["", *notes]

    ran = [col for col in cols if col[1] is not None]
    if ran:
        lines += ["", "| Column | GPU | Tool version |", "| --- | --- | --- |"]
        for col in ran:
            env = runner.load_environment(*col) or {}
            gpu = (env.get("gpu") or "").splitlines()
            lines.append(f"| {_label(col)} | {_md(gpu[0].strip() if gpu else '')} | {_md(_version_line(env.get('version')))} |")
    return "\n".join(lines) + "\n"
