"""Tests for predict.config: the config loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from predict.config import (
    REPO_ROOT,
    VESSEL_NAME_MAP,
    VESSEL_NAMES,
    load_config,
)


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    # D005: in-plane 0.375 mm, the cohort native median.
    assert cfg.resample.target_spacing == (0.375, 0.375, 3.0)
    assert cfg.hu.agatston_threshold == 130
    assert cfg.hu.clip_min == -200
    assert cfg.hu.clip_max == 3000
    assert cfg.hu.metal_artifact_threshold == 2000
    assert cfg.lesion_grouping.max_inplane_dist_mm == 5.0
    assert cfg.lesion_grouping.max_slice_gap == 1


def test_default_exclusions_present():
    cfg = load_config()
    assert "12" in cfg.cohort.exclude_pids
    assert "197" in cfg.cohort.exclude_pids
    assert "268" in cfg.cohort.exclude_pids
    assert cfg.cohort.exclude_ge_scanners is True


def test_vessel_name_map_keys_are_full_strings():
    expected = {
        "Left Anterior Descending Artery",
        "Right Coronary Artery",
        "Left Circumflex Artery",
        "Left Coronary Artery",
    }
    assert set(VESSEL_NAME_MAP) == expected
    assert set(VESSEL_NAMES) == set(VESSEL_NAME_MAP.values())


def test_load_config_from_custom_path(tmp_path: Path):
    custom = tmp_path / "alt.yaml"
    base = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base["seed"] = 7
    custom.write_text(yaml.safe_dump(base), encoding="utf-8")

    cfg = load_config(custom)
    assert cfg.seed == 7


def test_resample_target_spacing_property():
    cfg = load_config()
    assert cfg.resample.target_spacing == (
        cfg.resample.in_plane_mm,
        cfg.resample.in_plane_mm,
        cfg.resample.slice_mm,
    )


def test_data_raw_must_be_named_raw(tmp_path: Path):
    """The loaders derive raw/<pid> from data_raw.parent; another name would
    make them read a different folder than discovery."""
    custom = tmp_path / "alt.yaml"
    base = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    base["paths"]["data_raw"] = str(tmp_path / "dicoms")
    custom.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="named 'raw'"):
        load_config(custom)
