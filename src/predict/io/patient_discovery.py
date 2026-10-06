"""Patient discovery and cohort manifest construction.

Intersects the DICOM folders (``paths.data_raw``) with the XML annotations
(``paths.data_xml``), applies the cohort exclusions from configuration (D004)
and the slice geometry check (D039), and labels each included patient with
scanner/kernel metadata for downstream kernel-harmonisation and
stratification. The labels come from the series the loaders will use
(:func:`predict.io.dicom_loader.select_series`), never from an arbitrary file.

The output is a manifest (a list of :class:`PatientRecord`), persisted by
the orchestration script to ``outputs/01_manifest/manifest.csv``.

Decisions referencing this module:
    D004: Cohort exclusions at discovery
    D039: Slice geometry check (irregular_geometry exclusions)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path

from predict.io.dicom_loader import IrregularGeometryError, check_geometry, read_selected_headers

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatientRecord:
    pid: str
    raw_path: Path
    xml_path: Path
    manufacturer: str
    scanner_model: str
    kernel: str
    slice_thickness: float
    series_uid: str
    n_slices: int
    z_step_mm: float


@dataclass(frozen=True)
class DiscoveryResult:
    included: tuple[PatientRecord, ...]
    excluded_no_dicom: tuple[str, ...]
    excluded_no_xml: tuple[str, ...]
    excluded_by_config: tuple[str, ...]
    excluded_ge: tuple[str, ...]
    # (pid, reason) for patients whose selected series fails D039.
    excluded_geometry: tuple[tuple[str, str], ...] = ()


def _listdir_sorted_int(path: Path, *, dirs: bool = True, suffix: str | None = None) -> list[str]:
    """List entries under ``path``. ``dirs=True`` keeps directories, ``False`` keeps files."""
    out: list[str] = []
    for name in os.listdir(path):
        full = path / name
        if dirs and not full.is_dir():
            continue
        if not dirs and not full.is_file():
            continue
        if suffix is not None and not name.endswith(suffix):
            continue
        out.append(name)
    return sorted(out, key=lambda s: int(s.replace(suffix or "", "")) if (suffix and s.replace(suffix, "").isdigit()) or s.isdigit() else s)


def discover_patients(
    raw_dir: Path,
    xml_dir: Path,
    *,
    exclude_pids: tuple[str, ...] = (),
    exclude_ge_scanners: bool = True,
) -> DiscoveryResult:
    """Build the cohort manifest.

    Steps:
      1. Intersect raw DICOM folder names with XML basenames.
      2. Apply ``exclude_pids`` (config-driven, default: 12, 197, 268).
      3. For each remaining patient, read every header and select the series
         the loaders will use (D006); label scanner + kernel + thickness from it.
      4. If ``exclude_ge_scanners`` is True, drop any patient whose
         ``Manufacturer`` does not start with "SIEMENS" (case-insensitive).
      5. Drop any patient whose selected series fails the geometry check
         (D039), recording the reason.
      6. Return :class:`DiscoveryResult` carrying included + excluded sets.

    Patients listed in ``exclude_pids`` are excluded silently (they are
    expected exclusions, not anomalies). Patients failing the GE filter or
    missing DICOM/XML are logged.
    """
    raw_ids = {n for n in os.listdir(raw_dir) if (raw_dir / n).is_dir()}
    xml_ids = {n[:-4] for n in os.listdir(xml_dir) if n.endswith(".xml")}

    no_dicom = sorted(xml_ids - raw_ids, key=_int_key)
    no_xml = sorted(raw_ids - xml_ids, key=_int_key)
    if no_dicom:
        _log.warning("XML present, no DICOM: %s", no_dicom)
    if no_xml:
        _log.warning("DICOM present, no XML: %s", no_xml)

    candidates = sorted(raw_ids & xml_ids, key=_int_key)
    excluded_by_config = tuple(p for p in candidates if p in exclude_pids)
    eligible = [p for p in candidates if p not in exclude_pids]

    included: list[PatientRecord] = []
    excluded_ge: list[str] = []
    excluded_geometry: list[tuple[str, str]] = []
    for pid in eligible:
        headers = read_selected_headers(raw_dir / pid, pid)
        head = headers[0]
        manuf = head["manufacturer"].upper()
        if exclude_ge_scanners and not manuf.startswith("SIEMENS"):
            excluded_ge.append(pid)
            continue
        try:
            z_step = check_geometry(headers, pid)
        except IrregularGeometryError as exc:
            _log.warning("Excluded (D039): %s", exc)
            excluded_geometry.append((pid, str(exc)))
            continue
        included.append(PatientRecord(
            pid=pid,
            raw_path=raw_dir / pid,
            xml_path=xml_dir / f"{pid}.xml",
            manufacturer=head["manufacturer"],
            scanner_model=head["scanner_model"],
            kernel=head["kernel"],
            slice_thickness=head["slice_thickness"],
            series_uid=head["series_uid"],
            n_slices=len(headers),
            z_step_mm=z_step,
        ))

    _log.info(
        "Discovery: %d included, %d config-excluded, %d GE-excluded, "
        "%d geometry-excluded, %d no-DICOM, %d no-XML",
        len(included), len(excluded_by_config), len(excluded_ge),
        len(excluded_geometry), len(no_dicom), len(no_xml),
    )

    return DiscoveryResult(
        included=tuple(included),
        excluded_no_dicom=tuple(no_dicom),
        excluded_no_xml=tuple(no_xml),
        excluded_by_config=excluded_by_config,
        excluded_ge=tuple(excluded_ge),
        excluded_geometry=tuple(excluded_geometry),
    )


def _int_key(s: str) -> tuple[int, str]:
    """Sort key that orders numeric strings numerically, others lexicographically."""
    return (0, s) if not s.isdigit() else (-1, f"{int(s):010d}")


def manifest_rows(result: DiscoveryResult) -> list[dict]:
    """Flatten included records into dicts suitable for a CSV."""
    rows = []
    for rec in result.included:
        row = asdict(rec)
        row["raw_path"] = str(rec.raw_path)
        row["xml_path"] = str(rec.xml_path)
        rows.append(row)
    return rows
