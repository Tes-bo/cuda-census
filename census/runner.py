# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Compile and run one corpus case with one official toolchain."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
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
PROVENANCE_FIELDS = ("toolchain", "nvcc", "gpu")
MACHINE_PATTERN = r"[a-z0-9][a-z0-9._-]{0,63}"
ENVIRONMENT_FILE = "environment.json"

# Module attributes, read at call time, so tests can point them elsewhere.
GOLD_DIR = ROOT / "gold"
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "results"

STATUSES = (
    "pass",
    "compiled",
    "numeric-mismatch",
    "compile-fail",
    "run-fail",
    "unparsed-stdout",
    "gold-missing",
    "toolchain-missing",
)
# "compiled" is the best outcome of a profile with "run": false.
SUCCESS_STATUSES = ("pass", "compiled")

RESULT_KEYS = (
    "case",
    "toolchain",
    "machine",
    "status",
    "argv",
    "returncode",
    "source_sha256",
    "stdout_sha256",
    "gold_sha256",
    "keys",
    "gold_keys",
    "diffs",
    "stderr_tail",
    "cuda_headers",
    "seconds",
)

_ROOT_PREFIX = str(ROOT) + os.sep

# A CUDA include root is a directory that holds cuda_runtime_api.h. Under such
# a root, these top-level directories and file names belong to the CUDA
# toolkit; everything else there (stdio.h, bits/, tbb/) is an ordinary system
# header. ROCm ships no cuda_runtime_api.h, so its own trees never qualify.
CUDA_DIRS = frozenset({"crt", "cooperative_groups", "cub", "thrust", "cuda", "nv", "nvtx3", "cccl"})
_CUDA_FILE = re.compile(
    r"^(cuda|cublas|cufft|curand|cusparse|cusolver|cudnn|nvrtc|nvml|nvjpeg|npp|nvToolsExt|"
    r"vector_types|vector_functions|builtin_types|driver_types|driver_functions|host_defines|host_config|"
    r"device_types|device_functions|device_launch_parameters|device_atomic_functions|device_double_functions|"
    r"surface_|texture_|sm_\d|mma\.h|math_constants|library_types|channel_descriptor|common_functions|"
    r"math_functions|cooperative_groups|fatbinary)"
)


def parse_depends(text: str) -> list[str]:
    """The prerequisites of make-style dependency output such as `-M` prints."""
    return [token for token in text.replace("\\\n", " ").split() if not token.endswith(":")]


def cuda_headers(paths: list[str], cwd: Path) -> list[str]:
    """The CUDA toolkit headers among a compile's dependencies, sorted and unique."""
    roots: dict[Path, bool] = {}

    def is_root(directory: Path) -> bool:
        if directory not in roots:
            roots[directory] = (directory / "cuda_runtime_api.h").is_file()
        return roots[directory]

    found: set[str] = set()
    for raw in paths:
        path = Path(os.path.realpath(raw if os.path.isabs(raw) else cwd / raw))
        root = next((parent for parent in path.parents if is_root(parent)), None)
        if root is None:
            continue
        rel = path.relative_to(root)
        if rel.parts[0] in CUDA_DIRS or (len(rel.parts) == 1 and _CUDA_FILE.match(rel.name)):
            found.add(str(path))
    return sorted(found)


def load_manifest() -> dict:
    return json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))


def cases_by_id() -> dict[str, dict]:
    return {case["id"]: case for case in load_manifest()["cases"]}


def gold_path(case_id: str) -> Path:
    return GOLD_DIR / f"{case_id}.stdout"


def provenance_path() -> Path:
    return GOLD_DIR / "provenance.json"


def default_machine() -> str:
    """The host name as a machine label, or "local" when it does not fit the label pattern."""
    name = re.sub(r"[^a-z0-9._-]+", "-", socket.gethostname().lower()).strip("-._")[:64]
    return name if re.fullmatch(MACHINE_PATTERN, name) else "local"


def result_dir(toolchain: str, machine: str) -> Path:
    return RESULTS_DIR / toolchain / machine


def result_path(toolchain: str, machine: str, case_id: str) -> Path:
    return result_dir(toolchain, machine) / f"{case_id}.json"


def environment_path(toolchain: str, machine: str) -> Path:
    return result_dir(toolchain, machine) / ENVIRONMENT_FILE


def source_path(case: dict) -> Path:
    return ROOT / case["source"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    return _sha256(path.read_bytes()) if path.is_file() else None


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
        "machine": "",
        "status": "run-fail",
        "argv": [],
        "returncode": None,
        "source_sha256": None,
        "stdout_sha256": None,
        "gold_sha256": None,
        "keys": None,
        "gold_keys": None,
        "diffs": {},
        "stderr_tail": "",
        "cuda_headers": None,
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


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run(argv: list[str], cwd: Path, timeout: int) -> _Proc:
    # hipcc and nvcc are drivers that start clang, cicc, and ptxas. Each step
    # gets its own process group, so a timeout stops those as well.
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return _Proc(None, b"", b"", f"could not start {argv[0]}: {exc}")
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired as exc:
            # Something left the group and still holds the pipes.
            stdout, stderr = exc.stdout or b"", exc.stderr or b""
        return _Proc(None, stdout, stderr, f"timed out after {timeout} s")
    except BaseException:
        # In its own session the step does not see the terminal's Ctrl-C.
        _kill_group(proc)
        proc.wait()
        raise
    return _Proc(proc.returncode, stdout, stderr)


@dataclass
class Step:
    """One translate or compile command. stdout_to names the file that keeps its stdout."""

    argv: list[str]
    stdout_to: str | None = None


def prepare_workdir(case: dict, toolchain: str, profile: dict) -> tuple[Path, dict[str, str], list[Step]]:
    work = BUILD_DIR / toolchain / case["id"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    source = source_path(case)
    slots = {
        "source": str(source),
        "binary": str(work / "case"),
        # hip-clang decides the language from the extension, and .hip is the
        # extension HIP documents for HIP sources.
        "translated": str(work / "translated.hip"),
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


def execute_case(case: dict, toolchain: str, machine: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[dict, bytes]:
    """Build and run one case. Return the public result and the raw stdout of the binary."""
    started = time.monotonic()
    profiles = load_profiles()
    if toolchain not in profiles:
        raise ToolchainError(f"unknown toolchain {toolchain}")

    source_sha256 = file_sha256(source_path(case))

    def finish(**fields) -> dict:
        return _result(
            case=case["id"],
            toolchain=toolchain,
            machine=machine,
            source_sha256=source_sha256,
            seconds=time.monotonic() - started,
            **fields,
        )

    profile = profiles[toolchain]
    work, slots, steps = prepare_workdir(case, toolchain, profile)
    search = profile.get("search", [])

    # Resolve every tool first, so a half-installed toolchain is reported as
    # missing before any step runs.
    resolved: list[str] = []
    for step in steps:
        path = which_allowed(step.argv[0], search)
        if path is None:
            missing = finish(
                status="toolchain-missing",
                argv=step.argv,
                stderr_tail=f"not on PATH: {step.argv[0]}",
            )
            return missing, b""
        resolved.append(path)

    def scan() -> list[str] | None:
        """CUDA headers the compile read, from the profile's depends command, or None without one."""
        if "depends" not in profile:
            return None
        argv = render(profile["depends"], slots)
        tool = which_allowed(argv[0], search)
        if tool is None:
            return None
        proc = _run([tool, *argv[1:]], work, timeout)
        if proc.returncode != 0:
            return None
        found = cuda_headers(parse_depends(proc.stdout.decode("utf-8", "replace")), work)
        return [_portable(path) for path in found]

    logs: list[str] = []
    for step, path in zip(steps, resolved):
        proc = _run([path, *step.argv[1:]], work, timeout)
        if proc.returncode != 0:
            failed = finish(
                status="compile-fail",
                argv=step.argv,
                returncode=proc.returncode,
                stderr_tail=proc.log(with_stdout=step.stdout_to is None),
                cuda_headers=scan() if step is steps[-1] else None,
            )
            return failed, b""
        logs.append(proc.log(with_stdout=step.stdout_to is None))
        if step.stdout_to is not None:
            Path(step.stdout_to).write_bytes(proc.stdout)

    binary = slots["binary"]
    if not Path(binary).is_file():
        missing_binary = finish(
            status="compile-fail",
            argv=steps[-1].argv,
            returncode=0,
            stderr_tail=_tail(f"the compiler exited 0 but wrote no {_portable(binary)}\n" + "\n".join(filter(None, logs))),
        )
        return missing_binary, b""
    headers = scan()
    if profile.get("run", True) is False:
        # A compile-only profile targets hardware this machine does not have.
        # The translate and compile logs are kept, since their warnings are the result.
        compiled = finish(
            status="compiled",
            argv=steps[-1].argv,
            returncode=0,
            stderr_tail=_tail("\n".join(filter(None, logs))),
            cuda_headers=headers,
        )
        return compiled, b""
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
        cuda_headers=headers,
    )
    return result, proc.stdout


def _write_json(dest: Path, data: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def write_result(result: dict) -> Path:
    public = {key: result[key] for key in RESULT_KEYS}
    return _write_json(result_path(result["toolchain"], result["machine"], result["case"]), public)


def _capture(argv: list[str], search: list[str] | tuple[str, ...] = ()) -> str:
    """Trimmed stdout of a short query command, or "" when the tool is absent or fails."""
    path = which_allowed(argv[0], search)
    if path is None:
        return ""
    proc = _run([path, *argv[1:]], ROOT, 60)
    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", "replace").strip()[:2000]


def describe_environment(toolchain: str, machine: str) -> dict:
    """What a run used: the resolved tools, their version text, and the GPU, where the profile says how to ask."""
    profile = load_profiles()[toolchain]
    search = profile.get("search", [])
    tools: dict[str, str | None] = {}
    for key in ("translate", "compile"):
        if key in profile:
            name = profile[key][0]
            found = which_allowed(name, search)
            tools[name] = os.path.realpath(found) if found else None
    return {
        "toolchain": toolchain,
        "machine": machine,
        "tools": tools,
        "version": _capture(profile["version"], search) if "version" in profile else None,
        "gpu": _capture(profile["gpu"], search) if "gpu" in profile else None,
        "platform": platform.platform(),
    }


def write_environment(environment: dict) -> Path:
    return _write_json(environment_path(environment["toolchain"], environment["machine"]), environment)


def load_environment(toolchain: str, machine: str) -> dict | None:
    path = environment_path(toolchain, machine)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_provenance() -> dict | None:
    path = provenance_path()
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
    provenance_path().write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
