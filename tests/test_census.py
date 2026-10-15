# Copyright 2026 Tesbo
# SPDX-License-Identifier: Apache-2.0
"""Host-side checks. They do not launch a GPU."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from census.parse import diffs, parse_stdout
from census.runner import ROOT, load_manifest
from census.toolchains import ToolchainError, load_profiles, render, which_allowed


class ParseTests(unittest.TestCase):
    def test_one_integer_line(self) -> None:
        self.assertEqual(parse_stdout("sum=8256\n"), {"sum": 8256})

    def test_rejects_prose(self) -> None:
        self.assertIsNone(parse_stdout("sum = 1\n"))

    def test_diff_is_actual_minus_gold(self) -> None:
        self.assertEqual(diffs({"sum": 3}, {"sum": 5}), {"sum": 2})


class PolicyTests(unittest.TestCase):
    def test_retarget_is_not_a_toolchain(self) -> None:
        self.assertNotIn("cuda-retarget", load_profiles())
        with self.assertRaises(ToolchainError):
            which_allowed("cuda-retarget")
        with self.assertRaises(ToolchainError):
            render(["/opt/cuda-retarget", "{source}"], {"source": "main.cu"})

    def test_gold_matches_manifest(self) -> None:
        for case in load_manifest()["cases"]:
            gold = ROOT / "gold" / f"{case['id']}.stdout"
            self.assertTrue(gold.is_file(), case["id"])
            parsed = parse_stdout(gold.read_text(encoding="utf-8"))
            self.assertEqual(parsed, case["expect"])

    def test_result_schema_required_keys(self) -> None:
        schema = json.loads((ROOT / "schema" / "result.schema.json").read_text(encoding="utf-8"))
        self.assertIn("numeric-mismatch", schema["properties"]["status"]["enum"])
        self.assertNotIn("cuda-retarget", schema["properties"]["toolchain"]["enum"])


if __name__ == "__main__":
    unittest.main()
