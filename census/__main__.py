# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""cuda-census command line."""

from __future__ import annotations

import argparse
import json
import sys

from census.runner import (
    ROOT,
    cases_by_id,
    execute_case,
    load_manifest,
    nvcc_provenance,
    write_gold,
    write_result,
)
from census.toolchains import ToolchainError, official_names


def _selected(name: str) -> list[dict]:
    table = cases_by_id()
    if name == "all":
        return list(table.values())
    if name not in table:
        raise SystemExit(f"unknown case {name}")
    return [table[name]]


def cmd_list(_args: argparse.Namespace) -> int:
    for case in load_manifest()["cases"]:
        print(case["id"])
    return 0


def cmd_self_check(_args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if manifest["gold_toolchain"] != "nvcc":
        print("gold_toolchain must be nvcc", file=sys.stderr)
        return 1
    for case in manifest["cases"]:
        source = ROOT / case["source"]
        if not source.is_file():
            print(f"missing source {source}", file=sys.stderr)
            return 1
        if not isinstance(case["expect"], dict) or not case["expect"]:
            print(f"{case['id']} has no expect keys", file=sys.stderr)
            return 1
        gold = ROOT / "gold" / f"{case['id']}.stdout"
        if gold.is_file():
            from census.parse import parse_stdout

            parsed = parse_stdout(gold.read_text(encoding="utf-8"))
            if parsed != case["expect"]:
                print(f"gold mismatch for {case['id']}: {parsed} != {case['expect']}", file=sys.stderr)
                return 1
    try:
        from census.toolchains import which_allowed

        which_allowed("cuda-retarget")
    except ToolchainError:
        pass
    else:
        print("cuda-retarget was accepted as a toolchain", file=sys.stderr)
        return 1
    if "cuda-retarget" in official_names():
        print("cuda-retarget is listed in toolchains.json", file=sys.stderr)
        return 1
    print(f"self-check ok ({len(manifest['cases'])} cases)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.toolchain not in official_names():
        print(f"unknown toolchain {args.toolchain}", file=sys.stderr)
        print("official toolchains: " + ", ".join(official_names()), file=sys.stderr)
        return 2
    if args.write_gold and args.toolchain != "nvcc":
        print("gold is the nvcc stdout only", file=sys.stderr)
        return 2
    provenance = nvcc_provenance() if args.write_gold else None
    failed = 0
    for case in _selected(args.case):
        try:
            result = execute_case(case, args.toolchain)
        except ToolchainError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        path = write_result(result)
        print(f"{result['status']:20} {result['case']:20} {path.relative_to(ROOT)}")
        if args.write_gold and result["status"] == "pass":
            if result["keys"] != case["expect"]:
                print(
                    f"nvcc keys {result['keys']} != manifest {case['expect']}",
                    file=sys.stderr,
                )
                return 1
            write_gold(case["id"], result["_stdout"], provenance)
        if result["status"] == "toolchain-missing":
            if failed == 0:
                failed = 2
        elif result["status"] != "pass":
            failed = 1
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuda-census")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("self-check")
    run = sub.add_parser("run")
    run.add_argument("--toolchain", required=True)
    run.add_argument("--case", default="all")
    run.add_argument("--write-gold", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "self-check":
        return cmd_self_check(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
