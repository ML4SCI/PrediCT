"""All-or-nothing publication of a stage's output folder.

A stage writes into ``<name>.partial/`` and replaces ``<name>/`` only when
the whole run succeeded, so a reader never sees files from one run next to
files from another.

Publication is two renames: ``<name>`` -> ``<name>.old``, then
``<name>.partial`` -> ``<name>``. Between them ``<name>`` briefly does not
exist; a reader then fails instead of reading a mix. If a crash lands
exactly there, the next ``publish`` first restores ``<name>.old`` (it is
the last complete version) instead of deleting it.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def fresh_partial(target: Path) -> Path:
    """Return an empty ``<target>.partial`` folder next to ``target``."""
    _recover(target)
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    return partial


def _recover(target: Path) -> None:
    """Restore ``<target>.old`` if a publish was interrupted between its two
    renames (target missing, old present)."""
    old = target.with_name(target.name + ".old")
    if not target.exists() and old.exists():
        old.rename(target)


def publish(partial: Path, target: Path) -> None:
    """Replace ``target`` with ``partial``. The old folder is moved aside
    first and removed only after the rename, so a failure in between leaves
    either the old or the new folder complete."""
    _recover(target)
    old = target.with_name(target.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    partial.rename(target)
    if old.exists():
        shutil.rmtree(old)


def publish_keeping_subdirs(partial: Path, target: Path) -> None:
    """Like :func:`publish` (atomic swap of the whole folder), for a cohort
    folder that also holds other cohorts' folders, e.g. ``06_reduce/`` with
    ``stratified_*/`` and ``sensitivity_single_block/``.

    The current sub-folders are COPIED into ``partial`` first, which leaves
    the published folder untouched until the swap; then the folder is
    swapped as a whole. Every top-level file of the old folder is dropped,
    including stage 6 files written into the cohort folder, which a stage 5
    rerun invalidates. Interrupted ``*.partial`` / ``*.old`` sub-folders are
    not carried over.
    """
    _recover(target)
    if target.exists():
        for d in target.iterdir():
            if (d.is_dir() and not d.name.endswith((".partial", ".old"))
                    and not (partial / d.name).exists()):
                shutil.copytree(d, partial / d.name, symlinks=True)
    publish(partial, target)
