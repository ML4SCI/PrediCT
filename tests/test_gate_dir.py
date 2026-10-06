"""stability.primary_gate selects the gate directory stages 6 onward read."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from predict.config import DEFAULT_CONFIG_PATH, GATE_DIRS, gate_dir, load_config


def _write(tmp_path: Path, gate: str | None) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["paths"]["outputs"] = str(tmp_path / "out")
    if gate is None:
        raw["stability"].pop("primary_gate", None)
    else:
        raw["stability"]["primary_gate"] = gate
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_default_config_uses_the_realistic_gate():
    cfg = load_config()
    assert cfg.stability.primary_gate == "realistic"
    assert gate_dir(cfg) == cfg.paths.outputs / "05b_icc_realistic"


@pytest.mark.parametrize("gate", sorted(GATE_DIRS))
def test_each_gate_maps_to_its_directory(tmp_path, gate):
    cfg = load_config(_write(tmp_path, gate))
    assert gate_dir(cfg) == tmp_path / "out" / GATE_DIRS[gate]


def test_missing_key_means_classic(tmp_path):
    cfg = load_config(_write(tmp_path, None))
    assert gate_dir(cfg) == tmp_path / "out" / "05_icc"


def test_unknown_gate_raises(tmp_path):
    cfg = load_config(_write(tmp_path, "nonsense"))
    with pytest.raises(ValueError, match="primary_gate"):
        gate_dir(cfg)


def test_partial_smoke_gate_is_refused(tmp_path):
    """Downstream stages resolve the gate through
    gate_dir, so a 5b --allow-partial output must be refused there."""
    from predict.config import PARTIAL_SMOKE_MARKER
    cfg = load_config(_write(tmp_path, "realistic"))
    d = tmp_path / "out" / GATE_DIRS["realistic"]
    d.mkdir(parents=True)
    (d / PARTIAL_SMOKE_MARKER).write_text("smoke run\n")
    with pytest.raises(ValueError, match="partial smoke"):
        gate_dir(cfg)
