"""Tests for predict.provenance: the cluster copy has no up-to-date .git, so
the REVISION file written by the sync must take precedence."""
from __future__ import annotations

from pathlib import Path

from predict.provenance import REVISION_FILE, code_revision


def test_revision_file_takes_precedence(tmp_path: Path):
    (tmp_path / REVISION_FILE).write_text("0b144f1247721a8db947bc660935ae446b79aa5f-dirty\n")
    assert code_revision(tmp_path) == "0b144f1247721a8db947bc660935ae446b79aa5f-dirty"


def test_empty_revision_file_is_ignored(tmp_path: Path):
    (tmp_path / REVISION_FILE).write_text("\n")
    assert code_revision(tmp_path) == "unknown"   # tmp_path is not a git checkout


def test_no_revision_and_no_git_is_unknown(tmp_path: Path):
    assert code_revision(tmp_path) == "unknown"
