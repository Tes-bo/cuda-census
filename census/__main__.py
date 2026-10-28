# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""cuda-census command line."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from census.parse import KEY_PATTERN, diffs, parse_stdout
from census.runner import (
    DEFAULT_TIMEOUT,
    RESULT_KEYS,
    STATUSES,
    cases_by_id,
    execute_case,
    gold_path,
    load_manifest,
    load_provenance,
    nvcc_provenance,
    provenance_conflict,
    write_gold,
    write_result,
)
from census.toolchains import (
    GOLD_TOOLCHAIN,
    ROOT,
    ToolchainError,
    load_profiles,
    official_names,
    profile_problems,
    which_allowed,
)

_CASE_ID = re.compile(r"[a-z0-9_]+")


def cmd_list(_args: argparse.Namespace) -> int:
    for case in load_manifest()["cases"]:
        print(case["id"])
    return 0


def _manifest_problems(manifest: dict) -> list[str]:
    problems: list[str] = []
    if manifest.get("gold_toolchain") != GOLD_TOOLCHAIN:
        problems.append(f"corpus/manifest.json: gold_toolchain must be {GOLD_TOOLCHAIN}")
    seen: set[str] = set()
    for case in manifest.get("cases", []):
        cid = case.get("id", "")
        if not isinstance(cid, str) or not _CASE_ID.fullmatch(cid):
            problems.append(f"case id {cid!r} is not lowercase letters, digits, and underscores")
            continue
        if cid in seen:
            problems.append(f"{cid}: listed twice in the manifest")
        seen.add(cid)
        want_source = f"corpus/{cid}/main.cu"
        if case.get("source") != want_source:
            problems.append(f"{cid}: source must be {want_source}")
        elif not (ROOT / want_source).is_file():
            problems.append(f"{cid}: missing {want_source}")
        expect = case.get("expect")
        if not isinstance(expect, dict) or not expect:
            problems.append(f"{cid}: no expect keys")
            continue
        bad = [k for k, v in expect.items() if not re.fullmatch(KEY_PATTERN, k) or type(v) is not int]
        if bad:
            problems.append(f"{cid}: expect entries are not name=integer: {', '.join(bad)}")
        gold = gold_path(cid)
        if not gold.is_file():
            problems.append(f"{cid}: no gold/{cid}.stdout; produce it with nvcc --write-gold")
            continue
        parsed = parse_stdout(gold.read_text(encoding="utf-8"))
        if parsed is None:
            problems.append(f"{cid}: gold/{cid}.stdout is not name=integer lines")
        elif parsed != expect:
            problems.append(f"{cid}: gold differs from manifest expect (gold minus expect): {diffs(expect, parsed)}")
    for path in sorted((ROOT / "corpus").iterdir()):
        if path.is_dir() and path.name not in seen:
            problems.append(f"corpus/{path.name}/ is not in the manifest")
    for path in sorted((ROOT / "gold").glob("*.stdout")):
        if path.stem not in seen:
            problems.append(f"gold/{path.name} has no manifest case")
    return problems


def _provenance_problems() -> list[str]:
    gold_ids = {path.stem for path in (ROOT / "gold").glob("*.stdout")}
    recorded = load_provenance()
    if recorded is None:
        return ["gold/provenance.json is missing"] if gold_ids else []
    problems: list[str] = []
    if recorded.get("toolchain") != GOLD_TOOLCHAIN:
        problems.append(f"gold/provenance.json: toolchain must be {GOLD_TOOLCHAIN}")
    for field in ("nvcc", "gpu"):
        if not recorded.get(field):
            problems.append(f"gold/provenance.json: no {field}")
    listed = recorded.get("cases", [])
    if listed != sorted(set(listed)):
        problems.append("gold/provenance.json: cases are not sorted and unique")
    unlisted = sorted(gold_ids - set(listed))
    stale = sorted(set(listed) - gold_ids)
    if unlisted:
        problems.append(f"gold/provenance.json does not list {', '.join(unlisted)}")
    if stale:
        problems.append(f"gold/provenance.json lists cases without gold: {', '.join(stale)}")
    return problems


def _toolchain_problems() -> list[str]:
    profiles = load_profiles()
    problems = profile_problems(profiles)
    try:
        which_allowed("cuda-retarget")
    except ToolchainError:
        pass
    else:
        problems.append("cuda-retarget was accepted as a toolchain")
    schema = json.loads((ROOT / "schema" / "result.schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]
    if props["toolchain"]["enum"] != list(profiles):
        problems.append("schema toolchain enum differs from toolchains.json")
    if sorted(props["status"]["enum"]) != sorted(STATUSES):
        problems.append("schema status enum differs from the runner statuses")
    if sorted(schema["required"]) != sorted(RESULT_KEYS) or sorted(props) != sorted(RESULT_KEYS):
        problems.append("schema keys differ from the runner result keys")
    return problems


def cmd_self_check(_args: argparse.Namespace) -> int:
    manifest = load_manifest()
    problems = _manifest_problems(manifest) + _provenance_problems() + _toolchain_problems()
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"self-check failed ({len(problems)} problems)", file=sys.stderr)
        return 1
    print(f"self-check ok ({len(manifest['cases'])} cases, {len(official_names())} toolchains)")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    table = cases_by_id()
    if args.case == "all":
        cases = list(table.values())
    elif args.case in table:
        cases = [table[args.case]]
    else:
        print(f"unknown case {args.case}; `list` prints the case ids", file=sys.stderr)
        return 2
    if args.write_gold and args.toolchain != GOLD_TOOLCHAIN:
        print(f"gold is the {GOLD_TOOLCHAIN} stdout only", file=sys.stderr)
        return 2

    provenance = None
    if args.write_gold:
        provenance = nvcc_provenance()
        if not provenance["nvcc"]:
            print("--write-gold needs nvcc on PATH", file=sys.stderr)
            return 2
        if not provenance["gpu"]:
            print("--write-gold needs nvidia-smi to name the GPU", file=sys.stderr)
            return 1
        conflict = provenance_conflict(provenance)
        if conflict:
            print(conflict, file=sys.stderr)
            return 1

    counts: Counter[str] = Counter()
    for case in cases:
        try:
            result, stdout = execute_case(case, args.toolchain, timeout=args.timeout)
        except ToolchainError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        path = write_result(result)
        counts[result["status"]] += 1
        print(f"{result['status']:20} {result['case']:20} {path.relative_to(ROOT)}", flush=True)
        if provenance is not None and result["status"] == "pass":
            write_gold(case["id"], stdout, provenance)

    summary = ", ".join(f"{counts[status]} {status}" for status in STATUSES if counts[status])
    print(f"{args.toolchain}: {summary}")
    failed = set(counts) - {"pass"}
    if not failed:
        return 0
    if failed == {"toolchain-missing"}:
        return 2
    return 1


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cuda-census")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="print the case ids")
    sub.add_parser("self-check", help="check manifest, gold, provenance, toolchains, and schema")
    run = sub.add_parser("run", help="build and run cases with one toolchain")
    run.add_argument("--toolchain", required=True, choices=official_names())
    run.add_argument("--case", default="all", help="one case id, or all (default)")
    run.add_argument(
        "--timeout",
        type=_positive,
        default=DEFAULT_TIMEOUT,
        help=f"seconds allowed for each translate, compile, and run step (default {DEFAULT_TIMEOUT})",
    )
    run.add_argument("--write-gold", action="store_true", help="nvcc only: record passing stdout as gold")
    args = parser.parse_args(argv)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "self-check":
        return cmd_self_check(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
