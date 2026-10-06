"""Tests for predict.io.patient_discovery.

Builds synthetic ``data/raw`` and ``data/calcium_xml`` trees and verifies
that:
  * intersect of folders and XML files defines the candidate set
  * configured PIDs are excluded
  * GE-manufacturer patients are excluded when the flag is set
  * orphan XMLs (no DICOM) and orphan DICOMs (no XML) are reported
  * patients whose selected series fails the slice geometry check (D039)
    are excluded with a reason, and both loaders refuse to load them
  * both loaders agree on the z spacing of an accepted series
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from predict.io.dicom_loader import (
    IrregularGeometryError,
    load_patient_dicom,
    load_patient_metadata,
)
from predict.io.patient_discovery import discover_patients


def _write_synthetic_dicom(path: Path, *, manufacturer: str = "SIEMENS", model: str = "SOMATOM Force",
                           kernel: str = "Qr36d/2", thickness: float = 3.0,
                           ipp_z: float = 0.0, instance_number: int = 1,
                           series_number: int = 1, series_uid: str | None = None,
                           pixel_spacing: float = 0.5) -> None:
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.Manufacturer = manufacturer
    ds.ManufacturerModelName = model
    ds.ConvolutionKernel = kernel
    ds.SliceThickness = thickness
    ds.InstanceNumber = instance_number
    ds.SeriesNumber = series_number
    ds.ImagePositionPatient = [0.0, 0.0, ipp_z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.Rows = 16
    ds.Columns = 16
    ds.PixelSpacing = [pixel_spacing, pixel_spacing]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = -1024
    ds.PixelData = np.zeros((16, 16), dtype=np.int16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(path))


def _make_patient(root: Path, pid: str, *, manufacturer: str = "SIEMENS",
                  zs: tuple[float, ...] = (0.0, 3.0, 6.0), thickness: float = 3.0,
                  spacings: tuple[float, ...] | None = None) -> None:
    series_uid = generate_uid()
    pdir = root / "raw" / pid / "series"
    for i, z in enumerate(zs):
        _write_synthetic_dicom(
            pdir / f"IM-{i:04d}.dcm",
            manufacturer=manufacturer,
            thickness=thickness,
            ipp_z=float(z),
            instance_number=i + 1,
            series_uid=series_uid,
            pixel_spacing=spacings[i] if spacings else 0.5,
        )
    (root / "calcium_xml").mkdir(parents=True, exist_ok=True)
    (root / "calcium_xml" / f"{pid}.xml").write_bytes(b"<plist></plist>")


@pytest.fixture
def synthetic_cohort(tmp_path: Path) -> Path:
    _make_patient(tmp_path, "1")
    _make_patient(tmp_path, "2")
    _make_patient(tmp_path, "12")
    _make_patient(tmp_path, "5", manufacturer="GE MEDICAL SYSTEMS")
    # D039: one missing slice (6 mm step) and an overlapping reconstruction.
    _make_patient(tmp_path, "135", zs=(0.0, 3.0, 9.0, 12.0))
    _make_patient(tmp_path, "33", zs=(0.0, 1.5, 3.0, 4.5))
    # D039.1: regular z, but two in-plane grids in one series (pid 156 shape).
    _make_patient(tmp_path, "156", zs=(0.0, 3.0, 6.0, 9.0), spacings=(0.38, 0.38, 0.6, 0.6))
    # Orphan XML (no DICOM)
    (tmp_path / "calcium_xml" / "99.xml").write_bytes(b"<plist></plist>")
    # Orphan DICOM (no XML)
    pdir = tmp_path / "raw" / "200" / "series"
    _write_synthetic_dicom(pdir / "IM-0001.dcm", ipp_z=0.0)
    return tmp_path


def _dirs(root: Path) -> tuple[Path, Path]:
    return root / "raw", root / "calcium_xml"


def test_discovery_intersects_raw_and_xml(synthetic_cohort: Path):
    result = discover_patients(*_dirs(synthetic_cohort), exclude_pids=(), exclude_ge_scanners=False)
    included = {r.pid for r in result.included}
    assert included == {"1", "2", "12", "5"}
    assert result.excluded_no_dicom == ("99",)
    assert result.excluded_no_xml == ("200",)


def test_discovery_applies_config_exclusions(synthetic_cohort: Path):
    result = discover_patients(
        *_dirs(synthetic_cohort), exclude_pids=("12",), exclude_ge_scanners=False
    )
    assert {r.pid for r in result.included} == {"1", "2", "5"}
    assert result.excluded_by_config == ("12",)


def test_discovery_excludes_ge_when_flagged(synthetic_cohort: Path):
    result = discover_patients(
        *_dirs(synthetic_cohort), exclude_pids=(), exclude_ge_scanners=True
    )
    assert {r.pid for r in result.included} == {"1", "2", "12"}
    assert "5" in result.excluded_ge


def test_discovery_records_scanner_metadata(synthetic_cohort: Path):
    result = discover_patients(
        *_dirs(synthetic_cohort), exclude_pids=(), exclude_ge_scanners=False
    )
    rec = next(r for r in result.included if r.pid == "1")
    assert rec.manufacturer == "SIEMENS"
    assert rec.scanner_model == "SOMATOM Force"
    assert rec.kernel == "Qr36d/2"
    assert rec.slice_thickness == 3.0
    assert rec.series_uid
    assert rec.n_slices == 3
    assert rec.z_step_mm == pytest.approx(3.0)


def test_discovery_excludes_irregular_geometry(synthetic_cohort: Path):
    result = discover_patients(
        *_dirs(synthetic_cohort), exclude_pids=(), exclude_ge_scanners=False
    )
    reasons = dict(result.excluded_geometry)
    assert set(reasons) == {"135", "33", "156"}
    assert "irregular z steps" in reasons["135"]
    assert "overlapping or gapped" in reasons["33"]
    assert "mixed in-plane grid" in reasons["156"]
    assert not {"135", "33", "156"} & {r.pid for r in result.included}


def test_discovery_uses_configured_xml_dir(tmp_path: Path):
    """The XML folder is whatever is passed, not
    data_root / "calcium_xml"."""
    _make_patient(tmp_path, "1")
    moved = tmp_path / "annotations"
    (tmp_path / "calcium_xml").rename(moved)
    result = discover_patients(tmp_path / "raw", moved, exclude_ge_scanners=False)
    assert {r.pid for r in result.included} == {"1"}


@pytest.mark.parametrize("pid", ["135", "33", "156"])
def test_loaders_refuse_irregular_geometry(synthetic_cohort: Path, pid: str):
    with pytest.raises(IrregularGeometryError):
        load_patient_metadata(pid, synthetic_cohort)
    with pytest.raises(IrregularGeometryError):
        load_patient_dicom(pid, synthetic_cohort)


def test_loaders_agree_on_z_spacing(tmp_path: Path):
    """The header-only loader reports the same z spacing SimpleITK assigns,
    the checked z step, not SliceThickness."""
    _make_patient(tmp_path, "38", zs=tuple(i * 2.95 for i in range(5)), thickness=3.0)
    vol = load_patient_dicom("38", tmp_path)
    meta = load_patient_metadata("38", tmp_path)
    assert vol.pixel_spacing[2] == pytest.approx(2.95, abs=1e-3)
    assert meta.pixel_spacing[2] == pytest.approx(vol.pixel_spacing[2], abs=1e-3)
    assert meta.slice_positions == vol.slice_positions
    assert meta.series_uid == vol.series_uid
    assert meta.slice_thickness == vol.slice_thickness == 3.0
