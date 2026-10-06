"""default_n_jobs resolution order.

Some PBS Pro clusters set NCPUS, not PBS_NCPUS; reading only PBS_NCPUS would
silently fall back to 4 workers there.
"""
from __future__ import annotations

import pytest

from predict.config import default_n_jobs

VARS = ("PREDICT_N_JOBS", "PBS_NCPUS", "NCPUS", "PBS_JOBID", "SLURM_CPUS_ON_NODE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in VARS:
        monkeypatch.delenv(v, raising=False)


def test_fallback_is_four():
    assert default_n_jobs() == 4


def test_ncpus_inside_pbs_job(monkeypatch):
    monkeypatch.setenv("PBS_JOBID", "24610.cluster")
    monkeypatch.setenv("NCPUS", "12")
    assert default_n_jobs() == 12


def test_ncpus_ignored_outside_pbs_job(monkeypatch):
    monkeypatch.setenv("NCPUS", "12")
    assert default_n_jobs() == 4


def test_order(monkeypatch):
    monkeypatch.setenv("PBS_JOBID", "1.x")
    monkeypatch.setenv("NCPUS", "12")
    monkeypatch.setenv("PBS_NCPUS", "8")
    assert default_n_jobs() == 8
    monkeypatch.setenv("PREDICT_N_JOBS", "3")
    assert default_n_jobs() == 3


def test_cap(monkeypatch):
    monkeypatch.setenv("PBS_JOBID", "1.x")
    monkeypatch.setenv("NCPUS", "32")
    assert default_n_jobs(cap=16) == 16
