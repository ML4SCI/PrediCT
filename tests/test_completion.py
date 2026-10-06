"""Tests for predict.completion (fingerprints and completion markers)."""
from __future__ import annotations

from pathlib import Path

import pytest

from predict.completion import (
    MARKER, clear_marker, code_sha, digest, file_sha, require_marker, write_marker,
)


def test_file_sha_missing_and_content(tmp_path: Path):
    assert file_sha(tmp_path / "nope") == "missing"
    (tmp_path / "a").write_bytes(b"x")
    (tmp_path / "b").write_bytes(b"x")
    assert file_sha(tmp_path / "a") == file_sha(tmp_path / "b")


def test_digest_is_order_independent_for_dict_keys():
    assert digest({"a": 1, "b": [1, 2]}) == digest({"b": [1, 2], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})


def test_code_sha_changes_when_any_module_changes(tmp_path: Path):
    mod = tmp_path / "src" / "predict" / "m.py"
    mod.parent.mkdir(parents=True)
    mod.write_text("x = 1\n")
    before = code_sha(repo_root=tmp_path)
    mod.write_text("x = 2\n")
    assert code_sha(repo_root=tmp_path) != before


def test_marker_round_trip_and_mismatch(tmp_path: Path):
    with pytest.raises(SystemExit, match="missing"):
        require_marker(tmp_path, {"features_csv_sha": "abc"})
    write_marker(tmp_path, {"features_csv_sha": "abc", "arms": ["primary"]})
    assert require_marker(tmp_path, {"features_csv_sha": "abc"})["arms"] == ["primary"]
    with pytest.raises(SystemExit, match="features_csv_sha"):
        require_marker(tmp_path, {"features_csv_sha": "changed"})
    clear_marker(tmp_path)
    assert not (tmp_path / MARKER).exists()
    clear_marker(tmp_path)   # no error when already absent


def test_upstream_code_ignores_stage5_packages(tmp_path: Path):
    """Fixing stage 5+ code must not invalidate stage 4 results; fixing
    stage 1-4 code must."""
    from predict.completion import UPSTREAM_PACKAGES
    pkg = tmp_path / "src" / "predict"
    for sub in ("features", "reduce"):
        (pkg / sub).mkdir(parents=True)
        (pkg / sub / "m.py").write_text("x = 1\n")
    up = code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES)
    (pkg / "reduce" / "m.py").write_text("x = 2\n")
    assert code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES) == up
    (pkg / "features" / "m.py").write_text("x = 2\n")
    assert code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES) != up


def test_upstream_config_ignores_stage5_sections():
    from types import SimpleNamespace

    from predict.completion import upstream_fingerprint
    base = {"seed": 42, "stability": {"icc_threshold": 0.75}, "reduce": {"pca_cumvar": 0.9}}
    fp = upstream_fingerprint(SimpleNamespace(raw=base))
    assert upstream_fingerprint(SimpleNamespace(raw={**base, "reduce": {"pca_cumvar": 0.8}})) == fp
    assert upstream_fingerprint(SimpleNamespace(raw={**base, "stability": {"icc_threshold": 0.8}})) != fp


def test_upstream_code_ignores_bookkeeping_modules(tmp_path: Path):
    """Editing publish.py / provenance.py / completion.py changes no computed
    value, so it must not invalidate stage 4 (it did: the publish.py fix made
    5b refuse valid draws). Editing config.py must."""
    from predict.completion import UPSTREAM_PACKAGES
    pkg = tmp_path / "src" / "predict"
    (pkg / "features").mkdir(parents=True)
    for m in ("__init__.py", "config.py", "publish.py", "provenance.py", "completion.py"):
        (pkg / m).write_text("x = 1\n")
    (pkg / "features" / "m.py").write_text("x = 1\n")
    up = code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES)
    for m in ("publish.py", "provenance.py", "completion.py"):
        (pkg / m).write_text("x = 2\n")
    assert code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES) == up
    (pkg / "config.py").write_text("x = 2\n")
    assert code_sha(repo_root=tmp_path, packages=UPSTREAM_PACKAGES) != up
