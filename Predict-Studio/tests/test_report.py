"""
test_report.py — runs tests/report_check.mjs, which checks ui/report.js (the one-page report).

The report is browser code, but its numbers come from a pure function, so Node can
test it without a browser. Skipped when Node is not installed.

Does NOT: open a browser or print anything.
Usage: python -m pytest tests/test_report.py -v
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_report_numbers_match_result_files():
    r = subprocess.run(["node", "tests/report_check.mjs"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr

