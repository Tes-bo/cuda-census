# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Host-side checks. They do not launch a GPU.

Runner tests put small Python stand-ins for the official tools on PATH. The
stand-in compiler writes a "binary" whose stdout, exit code, and run time are
set through FAKE_* environment variables.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from census import runner
from census.__main__ import _docs_problems, main
from census.parse import diffs, parse_stdout
from census.report import build_report
from census.runner import ROOT, RESULT_KEYS, execute_case, load_manifest
from census.toolchains import ToolchainError, load_profiles, profile_problems, render, which_allowed

PY = sys.executable

FAKE_BINARY = f"""#!{PY}
import os, sys, time
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
sys.stdout.buffer.write(bytes.fromhex(os.environ.get("FAKE_STDOUT_HEX", "")))
sys.stderr.write(os.environ.get("FAKE_STDERR", ""))
sys.exit(int(os.environ.get("FAKE_RC", "0")))
"""

FAKE_COMPILER = f"""#!{PY}
import os, sys
if os.environ.get("FAKE_COMPILE_FAIL"):
    sys.stderr.write("fake: error: no such intrinsic\\n")
    sys.exit(1)
out = sys.argv[sys.argv.index("-o") + 1]
with open(out, "w") as f:
    f.write({FAKE_BINARY!r})
os.chmod(out, 0o755)
"""

FAKE_HIPIFY = f"""#!{PY}
import sys
sys.stdout.write(open(sys.argv[1]).read())
"""

FAKE_MUSIFY = f"""#!{PY}
import sys
with open(sys.argv[-1], "a") as f:
    f.write("// musified\\n")
"""

FAKE_TOOLS = {
    "nvcc": FAKE_COMPILER,
    "hipcc": FAKE_COMPILER,
    "mcc": FAKE_COMPILER,
    "cucc": FAKE_COMPILER,
    "hipify-perl": FAKE_HIPIFY,
    "musify-text": FAKE_MUSIFY,
}


def _write_tool(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


class ParseTests(unittest.TestCase):
    def test_one_integer_line(self) -> None:
        self.assertEqual(parse_stdout("sum=8256\n"), {"sum": 8256})

    def test_wide_and_negative_integers(self) -> None:
        self.assertEqual(
            parse_stdout("a=-9223372036854775808\nb=18446744073709551615\n"),
            {"a": -9223372036854775808, "b": 18446744073709551615},
        )

    def test_rejects_prose_duplicates_and_empty(self) -> None:
        self.assertIsNone(parse_stdout("sum = 1\n"))
        self.assertIsNone(parse_stdout("sum=1\nsum=1\n"))
        self.assertIsNone(parse_stdout(""))
        self.assertIsNone(parse_stdout("sum=1\n\nx=2\n"))

    def test_diff_is_actual_minus_gold(self) -> None:
        self.assertEqual(diffs({"sum": 3}, {"sum": 5}), {"sum": 2})
        self.assertEqual(diffs({"sum": 3}, {"sum": 3}), {})

    def test_key_on_one_side_is_not_a_number(self) -> None:
        self.assertEqual(diffs({"a": 0, "b": 1}, {"b": 1, "c": 0}), {"a": None, "c": None})


class PolicyTests(unittest.TestCase):
    def test_retarget_is_not_a_toolchain(self) -> None:
        self.assertNotIn("cuda-retarget", load_profiles())
        with self.assertRaises(ToolchainError):
            which_allowed("cuda-retarget")
        with self.assertRaises(ToolchainError):
            render(["/opt/cuda-retarget", "{source}"], {"source": "main.cu"})

    def test_retarget_path_hit_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shim = Path(tmp) / "cuda-retarget" / "bin"
            shim.mkdir(parents=True)
            target = _write_tool(shim, "hipcc", FAKE_COMPILER)
            link_dir = Path(tmp) / "link"
            link_dir.mkdir()
            (link_dir / "hipcc").symlink_to(target)
            for directory in (shim, link_dir):
                with mock.patch.dict(os.environ, {"PATH": str(directory)}):
                    with self.assertRaises(ToolchainError):
                        which_allowed("hipcc")

    def test_committed_profiles_are_valid(self) -> None:
        self.assertEqual(profile_problems(load_profiles()), [])

    def test_profile_problems_are_reported(self) -> None:
        profiles = {
            "hipify": {"role": "vendor", "compile": ["hipcc", "{sauce}"], "flags": []},
        }
        problems = "\n".join(profile_problems(profiles))
        self.assertIn("no nvcc profile", problems)
        self.assertIn("unknown slot {sauce}", problems)
        self.assertIn("unknown key flags", problems)

    def test_gold_matches_manifest(self) -> None:
        for case in load_manifest()["cases"]:
            gold = ROOT / "gold" / f"{case['id']}.stdout"
            self.assertTrue(gold.is_file(), case["id"])
            parsed = parse_stdout(gold.read_text(encoding="utf-8"))
            self.assertEqual(parsed, case["expect"], case["id"])

    def test_result_schema_matches_runner(self) -> None:
        schema = json.loads((ROOT / "schema" / "result.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(schema["required"]), sorted(RESULT_KEYS))
        self.assertEqual(sorted(schema["properties"]), sorted(RESULT_KEYS))
        self.assertEqual(schema["properties"]["toolchain"]["enum"], list(load_profiles()))
        self.assertEqual(sorted(schema["properties"]["status"]["enum"]), sorted(runner.STATUSES))


class _Sandbox(unittest.TestCase):
    """Fake tools on PATH and build, results, and gold under a temporary directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        for name, body in FAKE_TOOLS.items():
            _write_tool(self.bin, name, body)
        self.gold = self.tmp / "gold"
        self.gold.mkdir()
        for name, value in (
            ("BUILD_DIR", self.tmp / "build"),
            ("RESULTS_DIR", self.tmp / "results"),
            ("GOLD_DIR", self.gold),
        ):
            patcher = mock.patch.object(runner, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"PATH": str(self.bin)})
        env.start()
        self.addCleanup(env.stop)
        for key in [k for k in os.environ if k.startswith("FAKE_")]:
            os.environ.pop(key)
        self.source = self.tmp / "probe.cu"
        self.source.write_text("// probe\n", encoding="utf-8")
        self.case = {"id": "probe", "source": str(self.source), "expect": {"sum": 3}}
        (self.gold / "probe.stdout").write_text("sum=3\n", encoding="utf-8")

    def run_case(self, toolchain: str = "hipify", stdout: bytes = b"sum=3\n", timeout: int = 30, **env) -> dict:
        os.environ["FAKE_STDOUT_HEX"] = stdout.hex()
        for key, value in env.items():
            os.environ[f"FAKE_{key.upper()}"] = str(value)
        result, raw = execute_case(self.case, toolchain, timeout=timeout)
        self.assertEqual(list(result), list(RESULT_KEYS))
        if result["status"] not in ("compile-fail", "toolchain-missing") and result["returncode"] is not None:
            self.assertEqual(raw, stdout)
        return result


class RunnerTests(_Sandbox):
    def test_vendor_pass(self) -> None:
        result = self.run_case()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["diffs"], {})
        self.assertEqual(result["source_sha256"], hashlib.sha256(b"// probe\n").hexdigest())
        self.assertEqual(result["gold_sha256"], hashlib.sha256(b"sum=3\n").hexdigest())

    def test_numeric_mismatch(self) -> None:
        result = self.run_case(stdout=b"sum=5\n")
        self.assertEqual(result["status"], "numeric-mismatch")
        self.assertEqual(result["diffs"], {"sum": 2})

    def test_missing_key(self) -> None:
        result = self.run_case(stdout=b"other=1\n")
        self.assertEqual(result["status"], "numeric-mismatch")
        self.assertEqual(result["diffs"], {"other": None, "sum": None})

    def test_bytes_that_are_not_utf8(self) -> None:
        self.assertEqual(self.run_case(stdout=b"sum=\xff\n")["status"], "unparsed-stdout")

    def test_nonzero_exit_is_run_fail(self) -> None:
        result = self.run_case(rc=3, stderr="boom\n")
        self.assertEqual(result["status"], "run-fail")
        self.assertEqual(result["returncode"], 3)
        self.assertIn("boom", result["stderr_tail"])

    def test_timeout_is_run_fail(self) -> None:
        result = self.run_case(timeout=1, sleep=5)
        self.assertEqual(result["status"], "run-fail")
        self.assertIsNone(result["returncode"])
        self.assertIn("timed out after 1 s", result["stderr_tail"])

    def test_compile_fail(self) -> None:
        result = self.run_case(compile_fail=1)
        self.assertEqual(result["status"], "compile-fail")
        self.assertEqual(result["argv"][0], "hipcc")
        self.assertIn("no such intrinsic", result["stderr_tail"])

    def test_missing_tool_is_found_before_any_step(self) -> None:
        (self.bin / "hipcc").unlink()
        result = self.run_case()
        self.assertEqual(result["status"], "toolchain-missing")
        self.assertEqual(result["argv"][0], "hipcc")
        self.assertFalse((runner.BUILD_DIR / "hipify" / "probe" / "translated.cu").exists())

    def test_vendor_without_gold(self) -> None:
        (self.gold / "probe.stdout").unlink()
        self.assertEqual(self.run_case()["status"], "gold-missing")

    def test_nvcc_is_checked_against_the_manifest(self) -> None:
        (self.gold / "probe.stdout").unlink()
        self.assertEqual(self.run_case("nvcc")["status"], "pass")
        result = self.run_case("nvcc", stdout=b"sum=4\n")
        self.assertEqual(result["status"], "numeric-mismatch")
        self.assertEqual(result["diffs"], {"sum": 1})

    def test_musify_edits_a_copy(self) -> None:
        self.assertEqual(self.run_case("musify")["status"], "pass")
        self.assertEqual(self.source.read_text(encoding="utf-8"), "// probe\n")
        copy = runner.BUILD_DIR / "musify" / "probe" / "main.cu"
        self.assertEqual(copy.read_text(encoding="utf-8"), "// probe\n// musified\n")

    def test_paths_in_the_checkout_are_relative(self) -> None:
        result = runner._result(argv=[str(ROOT / "build" / "nvcc" / "x" / "case")])
        self.assertEqual(result["argv"], [os.path.join("build", "nvcc", "x", "case")])


class GoldTests(_Sandbox):
    def test_write_gold_adds_newline_and_sorted_case(self) -> None:
        provenance = {"toolchain": "nvcc", "nvcc": "release 12.0", "gpu": "GPU"}
        runner.write_gold("zeta", b"a=1", provenance)
        runner.write_gold("alpha", b"a=2\n", provenance)
        self.assertEqual((self.gold / "zeta.stdout").read_bytes(), b"a=1\n")
        recorded = json.loads((self.gold / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["cases"], ["alpha", "zeta"])
        self.assertEqual(recorded["gpu"], "GPU")

    def test_gold_from_a_second_setup_is_refused(self) -> None:
        here = {"toolchain": "nvcc", "nvcc": "release 12.0", "gpu": "V100"}
        self.assertIsNone(runner.provenance_conflict(here))
        runner.write_gold("probe", b"sum=3\n", here)
        self.assertIsNone(runner.provenance_conflict(here))
        conflict = runner.provenance_conflict(dict(here, gpu="A100"))
        self.assertIn("gpu recorded 'V100', here 'A100'", conflict)


class CliTests(_Sandbox):
    def call(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_self_check_passes_on_the_committed_tree(self) -> None:
        with mock.patch.object(runner, "GOLD_DIR", ROOT / "gold"):
            code, out, err = self.call("self-check")
        self.assertEqual(code, 0, err)
        self.assertIn("self-check ok", out)

    def test_only_missing_tools_exits_2(self) -> None:
        os.environ["PATH"] = str(self.tmp / "empty")
        code, out, _ = self.call("run", "--toolchain", "hipify")
        self.assertEqual(code, 2)
        self.assertIn(f"{len(load_manifest()['cases'])} toolchain-missing", out)

    def test_write_gold_is_nvcc_only(self) -> None:
        code, _, err = self.call("run", "--toolchain", "hipify", "--write-gold")
        self.assertEqual(code, 2)
        self.assertIn("nvcc stdout only", err)

    def test_unknown_case(self) -> None:
        code, _, err = self.call("run", "--toolchain", "nvcc", "--case", "nope")
        self.assertEqual(code, 2)
        self.assertIn("unknown case nope", err)


class DocsTests(unittest.TestCase):
    def test_docs_drift_is_reported(self) -> None:
        manifest = load_manifest()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        svg = (ROOT / "docs" / "overview.svg").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            r, s = Path(tmp) / "README.md", Path(tmp) / "overview.svg"
            r.write_text(readme, encoding="utf-8")
            s.write_text(svg, encoding="utf-8")
            self.assertEqual(_docs_problems(manifest, r, s), [])
            r.write_text(readme.replace("`sum=8256`", "`sum=8257`"), encoding="utf-8")
            s.write_text(svg.replace(f">{len(manifest['cases'])} cases<", ">1 cases<"), encoding="utf-8")
            problems = "\n".join(_docs_problems(manifest, r, s))
        self.assertIn("vector_add_int shows sum=8257, manifest has 8256", problems)
        self.assertIn("docs/overview.svg", problems)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(runner, "RESULTS_DIR", Path(tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cases = load_manifest()["cases"]

    def put(self, toolchain: str, case: dict, **fields) -> None:
        current = {
            "case": case["id"],
            "toolchain": toolchain,
            "source_sha256": runner.file_sha256(runner.source_path(case)),
            "gold_sha256": runner.file_sha256(runner.gold_path(case["id"])),
        }
        runner.write_result(runner._result(**{**current, **fields}))

    def test_cells_counts_and_notes(self) -> None:
        first, second, third = self.cases[:3]
        self.put("nvcc", first, status="pass")
        self.put("hipify", first, status="numeric-mismatch", diffs={"sum": -1, "x": None})
        self.put("hipify", second, status="pass", source_sha256="0" * 64)
        text = build_report(["nvcc", "hipify"])
        lines = text.splitlines()
        self.assertEqual(lines[0], "| Case | `nvcc` | `hipify` |")
        self.assertIn(f"| `{first['id']}` | pass | numeric-mismatch (2) |", lines)
        self.assertIn(f"| `{second['id']}` | not run | pass (stale) |", lines)
        self.assertIn(f"| `{third['id']}` | not run | not run |", lines)
        self.assertIn(f"| **pass** | 1/{len(self.cases)} | 0/{len(self.cases)} |", lines)
        self.assertIn("`sum` -1, `x` printed by one side only", text)
        self.assertIn(f"`hipify` `{second['id']}` is stale: source changed.", text)


if __name__ == "__main__":
    unittest.main()
