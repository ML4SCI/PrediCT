"""
VTP Export Module for Coronary Endothelial Shear Stress Predictions.

Converts PINN-predicted ESS CSV data into VTK XML PolyData (.vtp) files
for direct loading in ParaView, supporting scalar colouring, vector glyphs,
thresholding, clipping, interpolation, stream tracing, and point picking.

Physical Context
----------------
Endothelial Shear Stress (ESS) is the tangential viscous force per unit area
exerted by blood flow on the arterial endothelium. It is computed from the
velocity gradient tensor (Jacobian) at the vessel wall:

    tau_w = mu * (grad(u) + grad(u)^T) . n  -  [n^T . mu * (grad(u) + grad(u)^T) . n] * n

where mu is the dynamic viscosity, n is the outward wall normal, and u is
the velocity field predicted by the Physics-Informed Neural Network.

Low and oscillatory ESS regions correlate with atherosclerotic plaque
localisation — making accurate ESS visualisation a critical deliverable
for coronary biomechanics research.

VTP Format
----------
The VTK XML PolyData format (.vtp) stores:
  - Unstructured point clouds with explicit (x, y, z) coordinates.
  - Named PointData arrays (scalars, vectors) with full 64-bit precision.
  - XML header metadata for ParaView auto-detection of field types.

Author: Computational Scientist
Framework: PyVista / VTK
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError as e:
    raise ImportError(
        "PyVista is required for VTP export. Install via: pip install pyvista"
    ) from e

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Required CSV schema
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = [
    "x", "y", "z",
    "wss_x", "wss_y", "wss_z",
    "ess_magnitude",
]

COORDINATE_COLUMNS: List[str] = ["x", "y", "z"]
WSS_COMPONENT_COLUMNS: List[str] = ["wss_x", "wss_y", "wss_z"]


# ---------------------------------------------------------------------------
# Data Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """
    Structured report from ESS CSV validation.

    Attributes
    ----------
    is_valid : bool
        Overall pass/fail status of the validation suite.
    errors : List[str]
        Critical failures that prevent VTP construction.
    warnings : List[str]
        Non-fatal issues that may indicate data quality concerns.
    num_points : int
        Number of wall points in the dataset.
    ess_range : Tuple[float, float]
        (min, max) of ESS magnitude values.
    coordinate_bounds : Dict[str, Tuple[float, float]]
        Per-axis (min, max) of spatial coordinates.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    num_points: int = 0
    ess_range: Tuple[float, float] = (0.0, 0.0)
    coordinate_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"Validation {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Points         : {self.num_points:,}",
            f"  ESS Range      : [{self.ess_range[0]:.6e}, {self.ess_range[1]:.6e}] Pa",
        ]
        for axis, (lo, hi) in self.coordinate_bounds.items():
            lines.append(f"  {axis}-bounds     : [{lo:.4f}, {hi:.4f}]")
        if self.errors:
            lines.append("  ERRORS:")
            for err in self.errors:
                lines.append(f"    - {err}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")
        return "\n".join(lines)


class ESSDataValidator:
    """
    Validates ESS CSV data for structural and numerical integrity before
    VTP construction.

    Checks performed:
      1. All required columns are present.
      2. DataFrame is non-empty.
      3. Coordinates (x, y, z) contain no NaN or Inf values.
      4. ESS magnitude contains no NaN or Inf values.
      5. ESS magnitude is non-negative (physical constraint).
      6. WSS vector components contain no NaN or Inf values.
      7. WSS vectors have exactly 3 components per point.
    """

    @staticmethod
    def validate(df: pd.DataFrame) -> ValidationReport:
        """
        Execute the full validation suite on a loaded ESS DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The loaded ESS CSV data.

        Returns
        -------
        ValidationReport
            Structured report containing pass/fail status and diagnostics.
        """
        report = ValidationReport()

        # --- 1. Schema validation ---
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report.is_valid = False
            report.errors.append(
                f"Missing required columns: {missing}. "
                f"Expected: {REQUIRED_COLUMNS}"
            )
            return report

        # --- 2. Non-empty check ---
        if len(df) == 0:
            report.is_valid = False
            report.errors.append("CSV contains zero data rows.")
            return report

        report.num_points = len(df)

        # --- 3. Coordinate finiteness ---
        coords = df[COORDINATE_COLUMNS].values
        if not np.all(np.isfinite(coords)):
            n_bad = np.count_nonzero(~np.isfinite(coords))
            report.is_valid = False
            report.errors.append(
                f"Coordinates contain {n_bad} non-finite values (NaN or Inf)."
            )

        for col in COORDINATE_COLUMNS:
            vals = df[col].values
            report.coordinate_bounds[col] = (float(np.nanmin(vals)), float(np.nanmax(vals)))

        # --- 4. ESS finiteness ---
        ess = df["ess_magnitude"].values
        if not np.all(np.isfinite(ess)):
            n_bad = np.count_nonzero(~np.isfinite(ess))
            report.is_valid = False
            report.errors.append(
                f"ESS magnitude contains {n_bad} non-finite values (NaN or Inf)."
            )

        report.ess_range = (float(np.nanmin(ess)), float(np.nanmax(ess)))

        # --- 5. ESS non-negativity ---
        if np.any(ess < 0.0):
            n_neg = int(np.count_nonzero(ess < 0.0))
            report.warnings.append(
                f"ESS magnitude has {n_neg} negative values. "
                f"ESS is a magnitude (||tau_w||) and should be >= 0."
            )

        # --- 6. WSS vector finiteness ---
        wss = df[WSS_COMPONENT_COLUMNS].values
        if not np.all(np.isfinite(wss)):
            n_bad = np.count_nonzero(~np.isfinite(wss))
            report.is_valid = False
            report.errors.append(
                f"WSS vector components contain {n_bad} non-finite values."
            )

        # --- 7. WSS dimensionality ---
        if wss.shape[1] != 3:
            report.is_valid = False
            report.errors.append(
                f"WSS vector has {wss.shape[1]} components; expected 3."
            )

        # --- Physical plausibility warnings ---
        if report.ess_range[1] > 100.0:
            report.warnings.append(
                f"Maximum ESS = {report.ess_range[1]:.2f} Pa exceeds "
                f"physiological coronary range (typically < 10 Pa). "
                f"Verify non-dimensionalisation recovery."
            )

        if report.ess_range[1] < 1e-6:
            report.warnings.append(
                f"Maximum ESS = {report.ess_range[1]:.2e} Pa is near zero. "
                f"The network may not have converged or dynamic viscosity "
                f"scaling may be missing."
            )

        return report


# ---------------------------------------------------------------------------
# VTP Exporter
# ---------------------------------------------------------------------------

class ESSVTPExporter:
    """
    Constructs a VTK XML PolyData (.vtp) point cloud from ESS prediction data
    with full ParaView interoperability.

    The exporter is designed to be extensible: after constructing the base ESS
    dataset, additional scalar and vector fields (velocity, pressure, PDE
    residual, wall distance, branch ID, curvature, etc.) can be attached via
    ``add_scalar_field`` and ``add_vector_field`` before writing to disk.

    Parameters
    ----------
    coords : np.ndarray
        Physical wall point coordinates of shape ``(N, 3)``.
    patient_id : str, optional
        Patient identifier embedded in VTP field data metadata.

    Examples
    --------
    >>> exporter = ESSVTPExporter(coords, patient_id="patient_001")
    >>> exporter.attach_ess_fields(ess, wss_vector)
    >>> exporter.add_scalar_field("pressure", p_wall)
    >>> exporter.write("patient_001_ess.vtp")
    """

    def __init__(
        self,
        coords: np.ndarray,
        patient_id: str = "unknown",
    ) -> None:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"Coordinates must have shape (N, 3), got {coords.shape}."
            )

        self._patient_id = patient_id
        self._num_points = coords.shape[0]

        # Construct PolyData point cloud with vertex cells
        self._polydata = pv.PolyData(coords.astype(np.float64))

        # Add a vertex cell for every point so ParaView renders them
        # as a proper point cloud (not just geometry without topology).
        verts = np.column_stack([
            np.ones(self._num_points, dtype=np.int64),
            np.arange(self._num_points, dtype=np.int64),
        ])
        self._polydata.verts = verts

        logger.info(
            "Initialised ESSVTPExporter: %d wall points, patient=%s",
            self._num_points, patient_id,
        )

    @property
    def polydata(self) -> pv.PolyData:
        """Return the underlying PyVista PolyData object."""
        return self._polydata

    @property
    def num_points(self) -> int:
        """Return the number of wall points."""
        return self._num_points

    @property
    def field_names(self) -> List[str]:
        """Return the names of all attached PointData arrays."""
        return list(self._polydata.point_data.keys())

    # ----- Core ESS attachment -----

    def attach_ess_fields(
        self,
        ess_magnitude: np.ndarray,
        wss_vector: np.ndarray,
    ) -> None:
        """
        Attach the standard ESS and WSS fields to the PolyData.

        This creates five PointData arrays:
          - ``ESS``         : scalar, shape (N,)
          - ``WSS_Vector``  : 3-component vector, shape (N, 3)
          - ``WSS_X``       : scalar, shape (N,)
          - ``WSS_Y``       : scalar, shape (N,)
          - ``WSS_Z``       : scalar, shape (N,)

        Parameters
        ----------
        ess_magnitude : np.ndarray
            ESS magnitude array of shape ``(N,)`` or ``(N, 1)``.
        wss_vector : np.ndarray
            WSS vector array of shape ``(N, 3)``.

        Raises
        ------
        ValueError
            If array shapes are inconsistent with the point cloud size.
        """
        ess = np.asarray(ess_magnitude, dtype=np.float64).ravel()
        wss = np.asarray(wss_vector, dtype=np.float64)

        if ess.shape[0] != self._num_points:
            raise ValueError(
                f"ESS array length {ess.shape[0]} != {self._num_points} points."
            )
        if wss.shape != (self._num_points, 3):
            raise ValueError(
                f"WSS array shape {wss.shape} != expected ({self._num_points}, 3)."
            )

        # Scalar: ESS magnitude
        self._polydata.point_data["ESS"] = ess

        # Vector: Full WSS vector (ParaView auto-detects 3-component arrays as vectors)
        self._polydata.point_data["WSS_Vector"] = wss

        # Individual scalar components for axis-specific analysis
        self._polydata.point_data["WSS_X"] = wss[:, 0].copy()
        self._polydata.point_data["WSS_Y"] = wss[:, 1].copy()
        self._polydata.point_data["WSS_Z"] = wss[:, 2].copy()

        # Set ESS as the active scalar for default ParaView "Color By"
        self._polydata.set_active_scalars("ESS")

        # Set WSS_Vector as the active vector for glyph rendering
        self._polydata.set_active_vectors("WSS_Vector")

        logger.info(
            "Attached ESS fields: ESS range [%.4e, %.4e] Pa, "
            "WSS magnitude range [%.4e, %.4e] Pa",
            np.min(ess), np.max(ess),
            np.min(np.linalg.norm(wss, axis=1)),
            np.max(np.linalg.norm(wss, axis=1)),
        )

    # ----- Extensible field attachment -----

    def add_scalar_field(
        self,
        name: str,
        values: np.ndarray,
        set_active: bool = False,
    ) -> None:
        """
        Attach an arbitrary scalar field to the PolyData point data.

        Designed for future-proofing: velocity magnitude, pressure, PDE
        residual, wall distance, branch ID, local vessel radius, curvature,
        etc. can all be added without modifying the exporter API.

        Parameters
        ----------
        name : str
            ParaView-visible array name (e.g., "Pressure", "WallDistance").
        values : np.ndarray
            Scalar array of shape ``(N,)`` or ``(N, 1)``.
        set_active : bool, optional
            If True, set this field as the active scalar for default colouring.

        Raises
        ------
        ValueError
            If the array length does not match the number of points.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.shape[0] != self._num_points:
            raise ValueError(
                f"Scalar field '{name}' has {arr.shape[0]} values, "
                f"expected {self._num_points}."
            )

        self._polydata.point_data[name] = arr
        if set_active:
            self._polydata.set_active_scalars(name)

        logger.debug("Added scalar field '%s': range [%.4e, %.4e]", name, np.min(arr), np.max(arr))

    def add_vector_field(
        self,
        name: str,
        values: np.ndarray,
        set_active: bool = False,
    ) -> None:
        """
        Attach an arbitrary 3-component vector field to the PolyData point data.

        Parameters
        ----------
        name : str
            ParaView-visible array name (e.g., "Velocity", "SurfaceNormal").
        values : np.ndarray
            Vector array of shape ``(N, 3)``.
        set_active : bool, optional
            If True, set this field as the active vector.

        Raises
        ------
        ValueError
            If the array shape is not ``(N, 3)`` matching the point cloud.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (self._num_points, 3):
            raise ValueError(
                f"Vector field '{name}' has shape {arr.shape}, "
                f"expected ({self._num_points}, 3)."
            )

        self._polydata.point_data[name] = arr
        if set_active:
            self._polydata.set_active_vectors(name)

        logger.debug(
            "Added vector field '%s': magnitude range [%.4e, %.4e]",
            name, np.min(np.linalg.norm(arr, axis=1)), np.max(np.linalg.norm(arr, axis=1)),
        )

    # ----- I/O -----

    def write(self, output_path: Union[str, Path]) -> Path:
        """
        Write the constructed PolyData to a VTK XML PolyData (.vtp) file.

        The output preserves full 64-bit floating-point precision and is
        directly loadable in ParaView without manual conversion or filters.

        Parameters
        ----------
        output_path : Union[str, Path]
            Destination file path. Parent directories are created automatically.

        Returns
        -------
        Path
            The absolute path to the written .vtp file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Ensure .vtp extension
        if out.suffix.lower() != ".vtp":
            out = out.with_suffix(".vtp")

        self._polydata.save(str(out))

        file_size_kb = out.stat().st_size / 1024.0
        logger.info(
            "Exported VTP: %s (%.1f KB, %d points, %d fields: %s)",
            out, file_size_kb, self._num_points,
            len(self.field_names), self.field_names,
        )

        return out

    # ----- Convenience: from CSV -----

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        patient_id: str = "unknown",
        validate: bool = True,
    ) -> "ESSVTPExporter":
        """
        Factory constructor that reads an ESS CSV, validates it, and returns
        a fully populated exporter ready for ``write()``.

        Parameters
        ----------
        csv_path : Union[str, Path]
            Path to the ESS predictions CSV.
        patient_id : str, optional
            Patient identifier for metadata.
        validate : bool, optional
            Whether to run the full validation suite (default: True).

        Returns
        -------
        ESSVTPExporter
            Exporter with ESS and WSS fields already attached.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If validation fails with critical errors.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"ESS CSV not found: {csv_path}")

        logger.info("Reading ESS CSV: %s", csv_path)
        df = pd.read_csv(csv_path)

        # Validate
        if validate:
            report = ESSDataValidator.validate(df)
            logger.info("Validation report:\n%s", report.summary())
            if not report.is_valid:
                raise ValueError(
                    f"ESS CSV validation failed with {len(report.errors)} error(s):\n"
                    + "\n".join(f"  - {e}" for e in report.errors)
                )

        # Extract arrays
        coords = df[COORDINATE_COLUMNS].values.astype(np.float64)
        wss_vector = df[WSS_COMPONENT_COLUMNS].values.astype(np.float64)
        ess_magnitude = df["ess_magnitude"].values.astype(np.float64)

        # Build exporter
        exporter = cls(coords, patient_id=patient_id)
        exporter.attach_ess_fields(ess_magnitude, wss_vector)

        return exporter


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def export_ess_csv_to_vtp(
    csv_path: Union[str, Path],
    vtp_path: Union[str, Path],
    patient_id: str = "unknown",
    validate: bool = True,
) -> Path:
    """
    One-call convenience function: read ESS CSV, validate, and write VTP.

    This is the primary entry point for orchestration integration.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to the ESS predictions CSV file.
    vtp_path : Union[str, Path]
        Destination path for the VTP output file.
    patient_id : str, optional
        Patient identifier for metadata (default: "unknown").
    validate : bool, optional
        Whether to run the full validation suite (default: True).

    Returns
    -------
    Path
        Absolute path to the written .vtp file.

    Examples
    --------
    >>> from visualization import export_ess_csv_to_vtp
    >>> export_ess_csv_to_vtp("ess_predictions.csv", "patient_001_ess.vtp", patient_id="patient_001")
    """
    exporter = ESSVTPExporter.from_csv(csv_path, patient_id=patient_id, validate=validate)
    return exporter.write(vtp_path)
