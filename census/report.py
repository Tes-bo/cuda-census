# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Collect results/<toolchain>/<case>.json into one Markdown table."""

from __future__ import annotations

import json

from census import runner
from census.toolchains import GOLD_TOOLCHAIN

NOT_RUN = "not run"


def load_result(toolchain: str, case_id: str) -> dict | None:
    path = runner.result_path(toolchain, case_id)
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


def build_report(toolchains: list[str]) -> str:
    cases = runner.load_manifest()["cases"]
    table = {
        (case["id"], toolchain): load_result(toolchain, case["id"])
        for case in cases
        for toolchain in toolchains
    }

    lines = [
        "| Case | " + " | ".join(f"`{t}`" for t in toolchains) + " |",
        "| --- |" + " --- |" * len(toolchains),
    ]
    for case in cases:
        cells = [_cell(case, t, table[case["id"], t]) for t in toolchains]
        lines.append(f"| `{case['id']}` | " + " | ".join(cells) + " |")
    passed = []
    for toolchain in toolchains:
        count = sum(
            1
            for case in cases
            if (r := table[case["id"], toolchain]) is not None
            and r["status"] == "pass"
            and not stale_reasons(case, toolchain, r)
        )
        passed.append(f"{count}/{len(cases)}")
    lines.append("| **pass** | " + " | ".join(passed) + " |")

    notes = []
    for case in cases:
        for toolchain in toolchains:
            result = table[case["id"], toolchain]
            if result is None:
                continue
            reasons = stale_reasons(case, toolchain, result)
            if reasons:
                notes.append(f"- `{toolchain}` `{case['id']}` is stale: {', '.join(reasons)}.")
            if result["status"] == "numeric-mismatch":
                diffs = ", ".join(_diff_text(k, v) for k, v in sorted(result["diffs"].items()))
                notes.append(f"- `{toolchain}` `{case['id']}` differs from gold (actual minus gold): {diffs}.")
    if notes:
        lines += ["", *notes]
    return "\n".join(lines) + "\n"
