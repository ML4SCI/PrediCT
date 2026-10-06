"""Which code produced an output.

The cluster copy is synced without ``.git`` (rsync excludes it, so a sync
never re-sends git objects). ``git rev-parse`` there would read whatever
repository was copied once, long ago, and report a stale commit. The sync
command therefore writes a ``REVISION`` file next to the code: the full
commit hash, with ``-dirty`` appended when uncommitted changes were synced.
:func:`code_revision` reads it first and falls back to git only for a full
checkout without the file (local runs, tests).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from predict.config import REPO_ROOT

REVISION_FILE = "REVISION"


def code_revision(repo_root: Path = REPO_ROOT) -> str:
    """Commit of the running code (``-dirty`` if uncommitted changes), or
    ``"unknown"``."""
    try:
        text = (Path(repo_root) / REVISION_FILE).read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=str(repo_root), capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001  (no git binary)
        return "unknown"
