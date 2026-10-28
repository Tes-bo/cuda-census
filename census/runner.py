# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Compile and run one corpus case with one official toolchain."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from census.parse import diffs, parse_stdout
from census.toolchains import (
    GOLD_TOOLCHAIN,
    ROOT,
    ToolchainError,
    fill,
    load_profiles,
    render,
    which_allowed,
)

DEFAULT_TIMEOUT = 120
PROVENANCE_PATH = ROOT / "gold" / "provenance.json"
PROVENANCE_FIELDS = ("toolchain", "nvcc", "gpu")

STATUSES = (
    "pass",
    "numeric-mismatch",
    "compile-fail",
    "run-fail",
    "unparsed-stdout",
    "gold-missing",
    "toolchain-missing",
)

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

_ROOT_PREFIX = str(ROOT) + os.sep


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


def _portable(text: str) -> str:
    """Drop the checkout prefix so a published result does not name the local tree."""
    return text.replace(_ROOT_PREFIX, "")


def _parse(data: bytes) -> dict[str, int] | None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return parse_stdout(text)


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
    if empty["argv"]:
        empty["argv"] = [_portable(part) for part in empty["argv"]]
    return {key: empty[key] for key in RESULT_KEYS}


@dataclass
class _Proc:
    """Outcome of one child process. returncode is None when it never finished."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    error: str = ""

    def log(self, with_stdout: bool) -> str:
        parts = [self.error, self.stderr.decode("utf-8", "replace")]
        if with_stdout:
            parts.append(self.stdout.decode("utf-8", "replace"))
        return _tail(_portable("\n".join(part for part in parts if part)))


def _run(argv: list[str], cwd: Path, timeout: int) -> _Proc:
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _Proc(None, exc.stdout or b"", exc.stderr or b"", f"timed out after {timeout} s")
    except OSError as exc:
        return _Proc(None, b"", b"", f"could not start {argv[0]}: {exc}")
    return _Proc(proc.returncode, proc.stdout, proc.stderr)


@dataclass
class Step:
    """One translate or compile command. stdout_to names the file that keeps its stdout."""

    argv: list[str]
    stdout_to: str | None = None


def prepare_workdir(case: dict, toolchain: str, profile: dict) -> tuple[Path, dict[str, str], list[Step]]:
    work = ROOT / "build" / toolchain / case["id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    source = ROOT / case["source"]
    slots = {
        "source": str(source),
        "binary": str(work / "case"),
        "translated": str(work / "translated.cu"),
        "copied_source": str(work / "main.cu"),
    }
    if "copy_source_to" in profile:
        dest = fill(profile["copy_source_to"], slots)
        shutil.copyfile(source, dest)
        slots["source"] = dest
    steps: list[Step] = []
    if "translate" in profile:
        stdout_to = fill(profile["translate_stdout"], slots) if "translate_stdout" in profile else None
        steps.append(Step(render(profile["translate"], slots), stdout_to))
    steps.append(Step(render(profile["compile"], slots)))
    return work, slots, steps


def execute_case(case: dict, toolchain: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[dict, bytes]:
    """Build and run one case. Return the public result and the raw stdout of the binary."""
    started = time.monotonic()
    profiles = load_profiles()
    if toolchain not in profiles:
        raise ToolchainError(f"unknown toolchain {toolchain}")

    def finish(**fields) -> dict:
        return _result(case=case["id"], toolchain=toolchain, seconds=time.monotonic() - started, **fields)

    work, slots, steps = prepare_workdir(case, toolchain, profiles[toolchain])

    # Resolve every tool first, so a half-installed toolchain is reported as
    # missing before any step runs.
    resolved: list[str] = []
    for step in steps:
        path = which_allowed(step.argv[0])
        if path is None:
            missing = finish(
                status="toolchain-missing",
                argv=step.argv,
                stderr_tail=f"not on PATH: {step.argv[0]}",
            )
            return missing, b""
        resolved.append(path)

    for step, path in zip(steps, resolved):
        proc = _run([path, *step.argv[1:]], work, timeout)
        if proc.returncode != 0:
            failed = finish(
                status="compile-fail",
                argv=step.argv,
                returncode=proc.returncode,
                stderr_tail=proc.log(with_stdout=step.stdout_to is None),
            )
            return failed, b""
        if step.stdout_to is not None:
            Path(step.stdout_to).write_bytes(proc.stdout)

    binary = slots["binary"]
    proc = _run([binary], work, timeout)
    keys = _parse(proc.stdout)
    gold_file = gold_path(case["id"])
    gold_bytes = gold_file.read_bytes() if gold_file.is_file() else None
    gold_keys = _parse(gold_bytes) if gold_bytes is not None else None
    # nvcc is checked against the integers the manifest promises; vendors are
    # checked against the committed nvcc stdout.
    reference = case["expect"] if toolchain == GOLD_TOOLCHAIN else gold_keys

    key_diffs: dict[str, int | None] = {}
    if proc.returncode != 0:
        status = "run-fail"
    elif keys is None:
        status = "unparsed-stdout"
    elif reference is None:
        status = "gold-missing"
    else:
        key_diffs = diffs(reference, keys)
        status = "numeric-mismatch" if key_diffs else "pass"

    result = finish(
        status=status,
        argv=[binary],
        returncode=proc.returncode,
        stdout_sha256=_sha256(proc.stdout) if proc.returncode is not None else None,
        gold_sha256=_sha256(gold_bytes) if gold_bytes is not None else None,
        keys=keys,
        gold_keys=gold_keys,
        diffs=key_diffs,
        stderr_tail=proc.log(with_stdout=False),
    )
    return result, proc.stdout


def write_result(result: dict) -> Path:
    dest = ROOT / "results" / result["toolchain"] / f"{result['case']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    public = {key: result[key] for key in RESULT_KEYS}
    dest.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def load_provenance() -> dict | None:
    if not PROVENANCE_PATH.is_file():
        return None
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def provenance_conflict(current: dict) -> str | None:
    """Explain why gold from this setup cannot join the committed gold, or return None."""
    recorded = load_provenance()
    if not recorded or not recorded.get("cases"):
        return None
    changed = [
        f"{field} recorded {recorded.get(field)!r}, here {current.get(field)!r}"
        for field in PROVENANCE_FIELDS
        if recorded.get(field) != current.get(field)
    ]
    if not changed:
        return None
    return (
        "gold/provenance.json comes from a different setup ("
        + "; ".join(changed)
        + "). Gold from two setups is not mixed. To move all gold to this setup, "
        "remove gold/*.stdout and gold/provenance.json, then rerun --write-gold for every case."
    )


def write_gold(case_id: str, stdout: bytes, provenance: dict) -> None:
    path = gold_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not stdout.endswith(b"\n"):
        stdout += b"\n"
    path.write_bytes(stdout)
    recorded = load_provenance() or {}
    recorded.update(provenance)
    recorded["cases"] = sorted(set(recorded.get("cases", [])) | {case_id})
    PROVENANCE_PATH.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_line(argv: list[str], needle: str = "") -> str:
    """First non-empty stdout line of a tool that contains needle, or "" if the tool is absent."""
    path = which_allowed(argv[0])
    if path is None:
        return ""
    proc = _run([path, *argv[1:]], ROOT, 60)
    if proc.returncode != 0:
        return ""
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line and needle in line:
            return line
    return ""


def nvcc_provenance() -> dict:
    return {
        "toolchain": GOLD_TOOLCHAIN,
        "nvcc": _first_line(["nvcc", "--version"], "release"),
        "gpu": _first_line(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
    }
