# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Compile and run one corpus case with one official toolchain."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from census.parse import diffs, parse_stdout
from census.toolchains import ROOT, ToolchainError, load_profiles, render, which_allowed

RESULT_KEYS = (
    "case",
    "toolchain",
    "status",
    "argv",
    "returncode",
    "stdout_sha256",
    "gold_sha256",
    "keys",
    "gold_keys",
    "diffs",
    "stderr_tail",
    "seconds",
)


def load_manifest() -> dict:
    return json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))


def cases_by_id() -> dict[str, dict]:
    return {case["id"]: case for case in load_manifest()["cases"]}


def gold_path(case_id: str) -> Path:
    return ROOT / "gold" / f"{case_id}.stdout"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _result(**fields) -> dict:
    empty = {
        "case": "",
        "toolchain": "",
        "status": "run-fail",
        "argv": [],
        "returncode": None,
        "stdout_sha256": None,
        "gold_sha256": None,
        "keys": None,
        "gold_keys": None,
        "diffs": {},
        "stderr_tail": "",
        "seconds": 0.0,
    }
    empty.update(fields)
    return {key: empty[key] for key in RESULT_KEYS}


def _run(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _missing(case_id: str, toolchain: str, argv: list[str], started: float) -> dict:
    return _result(
        case=case_id,
        toolchain=toolchain,
        status="toolchain-missing",
        argv=argv,
        stderr_tail=f"not on PATH: {argv[0]}",
        seconds=time.monotonic() - started,
    )


def prepare_workdir(case: dict, toolchain: str, profile: dict) -> tuple[Path, dict[str, str], list[list[str]]]:
    work = ROOT / "build" / toolchain / case["id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    source = ROOT / case["source"]
    binary = work / "case"
    slots = {
        "source": str(source),
        "binary": str(binary),
        "translated": str(work / "translated.cu"),
        "copied_source": str(work / "main.cu"),
    }
    steps: list[list[str]] = []
    if "copy_source_to" in profile:
        dest = Path(slots["copied_source"])
        dest.write_bytes(source.read_bytes())
        slots["source"] = str(dest)
    if "translate" in profile:
        steps.append(render(profile["translate"], slots))
    steps.append(render(profile["compile"], slots))
    return work, slots, steps


def execute_case(case: dict, toolchain: str, timeout: int = 120) -> dict:
    started = time.monotonic()
    profiles = load_profiles()
    if toolchain not in profiles:
        raise ToolchainError(f"unknown toolchain {toolchain}")
    profile = profiles[toolchain]
    try:
        work, slots, steps = prepare_workdir(case, toolchain, profile)
    except ToolchainError as exc:
        return _result(
            case=case["id"],
            toolchain=toolchain,
            status="compile-fail",
            stderr_tail=str(exc),
            seconds=time.monotonic() - started,
        )

    for step in steps:
        resolved = which_allowed(step[0])
        if resolved is None:
            return _missing(case["id"], toolchain, step, started)
        if "translate_stdout" in profile and step == steps[0]:
            proc = _run([resolved, *step[1:]], work, timeout)
            if proc.returncode != 0:
                return _result(
                    case=case["id"],
                    toolchain=toolchain,
                    status="compile-fail",
                    argv=step,
                    returncode=proc.returncode,
                    stderr_tail=_tail(proc.stderr),
                    seconds=time.monotonic() - started,
                )
            Path(slots["translated"]).write_text(proc.stdout, encoding="utf-8")
            continue
        proc = _run([resolved, *step[1:]], work, timeout)
        if proc.returncode != 0:
            return _result(
                case=case["id"],
                toolchain=toolchain,
                status="compile-fail",
                argv=step,
                returncode=proc.returncode,
                stderr_tail=_tail(proc.stderr + proc.stdout),
                seconds=time.monotonic() - started,
            )

    binary = Path(slots["binary"])
    proc = _run([str(binary)], work, timeout)
    stdout_bytes = proc.stdout.encode()
    keys = parse_stdout(proc.stdout)
    gold_file = gold_path(case["id"])
    gold_bytes = gold_file.read_bytes() if gold_file.is_file() else None
    gold_keys = parse_stdout(gold_bytes.decode()) if gold_bytes is not None else None
    status = "pass"
    key_diffs: dict[str, int] = {}
    if proc.returncode != 0:
        status = "run-fail"
    elif keys is None:
        status = "unparsed-stdout"
    elif gold_keys is not None:
        key_diffs = diffs(gold_keys, keys)
        if key_diffs:
            status = "numeric-mismatch"
    elif toolchain != "nvcc":
        status = "gold-missing"
    result = _result(
        case=case["id"],
        toolchain=toolchain,
        status=status,
        argv=[str(binary)],
        returncode=proc.returncode,
        stdout_sha256=_sha256(stdout_bytes),
        gold_sha256=_sha256(gold_bytes) if gold_bytes is not None else None,
        keys=keys,
        gold_keys=gold_keys,
        diffs=key_diffs,
        stderr_tail=_tail(proc.stderr),
        seconds=time.monotonic() - started,
    )
    result["_stdout"] = proc.stdout
    return result


def write_result(result: dict) -> Path:
    dest = ROOT / "results" / result["toolchain"] / f"{result['case']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    public = {key: result[key] for key in RESULT_KEYS}
    dest.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def write_gold(case_id: str, stdout: str, provenance: dict) -> None:
    path = gold_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not stdout.endswith("\n"):
        stdout += "\n"
    path.write_text(stdout, encoding="utf-8")
    prov_path = ROOT / "gold" / "provenance.json"
    current = {}
    if prov_path.is_file():
        current = json.loads(prov_path.read_text(encoding="utf-8"))
    current.update(provenance)
    current.setdefault("cases", [])
    if case_id not in current["cases"]:
        current["cases"].append(case_id)
        current["cases"].sort()
    prov_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nvcc_provenance() -> dict:
    version = subprocess.run(
        ["nvcc", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    release = ""
    for line in version.stdout.splitlines():
        if "release" in line:
            release = line.strip()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    name = gpu.stdout.strip().splitlines()[0] if gpu.stdout.strip() else ""
    return {"toolchain": "nvcc", "nvcc": release, "gpu": name}
