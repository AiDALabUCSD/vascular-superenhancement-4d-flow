"""Assemble auto-flow pipeline inputs from this repo's existing NIfTIs.

The auto-flow pipeline expects, per patient, two co-registered NIfTIs in a
``<base_path>/<patient_name>/`` directory:

* ``mag_4dflow.nii.gz``            shape ``(X, Y, Z, T)``    - 4D-flow magnitude.
* ``vel-corrected_4dflow.nii.gz``  shape ``(X, Y, Z, T, 3)`` - corrected velocity
  with the three components stacked on the last axis as ``[vx, vy, vz]``.

We build both from data this project already generates:

* magnitude  -> the full-FOV composite ``4d_flow_mag_<id>.nii.gz`` (padded FOV).
* velocity   -> the ground-truth phase-error-corrected components
  ``4d_flow_v{x,y,z}_corr_<id>.nii.gz`` (cropped/unpadded FOV), zero-padded back
  onto the magnitude grid using each component's affine to find its placement.

Both files are written with the *magnitude's* affine so magnitude and velocity
share one grid (auto-flow reslices both with the same sampling planes), and we
keep our RAS affine end-to-end so the resulting spline/normal/segmentation
geometry lines up with our velocity fields.

Notes
-----
* ``build_corrected_velocities`` already negates the vz component once; auto-flow
  independently negates vz once as well, so the assembled velocity matches
  auto-flow's own convention. (The plan's QA gate confirms the sign empirically.)
* Mirrors auto-flow's ``process_corrected_velocity_npy`` ``ecc_holder`` padding,
  except the placement offset is derived from the affine rather than assuming a
  symmetric centered crop, so it is correct even for asymmetric crops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import nibabel as nib
import numpy as np

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from ..data_management.patients import Patient

_VELOCITY_COMPONENTS = ("vx", "vy", "vz")

MAG_FILENAME = "mag_4dflow.nii.gz"
VEL_FILENAME = "vel-corrected_4dflow.nii.gz"


def autoflow_staging_dir(patient: "Patient") -> Path:
    """Return (creating it) the per-patient auto-flow staging directory.

    Auto-flow is run with ``base_path = patient.flow_geometry_dir`` and
    ``patient_name = patient.identifier``, so it reads/writes everything under
    ``flow_geometry_dir/<identifier>/``. This keeps the heavy auto-flow raw
    outputs (splines, segmentations, resliced volumes, GIFs) separate from the
    compact geometry cache that the in-repo evaluator consumes.
    """
    staging = patient.flow_geometry_dir / patient.identifier
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _resolve_logger(patient: "Patient", logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is not None:
        return logger
    return getattr(patient, "_logger", None) or logging.getLogger(__name__)


def _voxel_offset(mag_affine: np.ndarray, comp_affine: np.ndarray) -> tuple[int, int, int]:
    """Voxel offset at which ``comp``'s grid sits inside the magnitude grid.

    Both affines must share the same direction/spacing (the corrected velocity is
    a centered crop of the same acquisition grid). The offset is recovered by
    solving ``mag_linear @ offset = comp_origin - mag_origin``.
    """
    mag_linear = mag_affine[:3, :3]
    comp_linear = comp_affine[:3, :3]
    if not np.allclose(mag_linear, comp_linear, atol=1e-3):
        raise ValueError(
            "Magnitude and corrected-velocity affines have different "
            f"direction/spacing; cannot co-register.\nmag:\n{mag_linear}\n"
            f"vel:\n{comp_linear}"
        )

    origin_diff = comp_affine[:3, 3] - mag_affine[:3, 3]
    offset_float = np.linalg.solve(mag_linear, origin_diff)
    offset = np.round(offset_float).astype(int)

    if not np.allclose(offset_float, offset, atol=1e-2):
        raise ValueError(
            f"Corrected-velocity grid is not voxel-aligned to the magnitude grid "
            f"(non-integer offset {offset_float}); refusing to assemble."
        )
    if np.any(offset < 0):
        raise ValueError(
            f"Negative placement offset {offset}: corrected velocity extends "
            "outside the magnitude FOV."
        )
    return int(offset[0]), int(offset[1]), int(offset[2])


def _load_magnitude(patient: "Patient", logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    """Load the full-FOV 4D magnitude as ``(X, Y, Z, T)`` plus its affine.

    Prefers the composite ``4d_flow_mag_<id>.nii.gz`` (already full FOV); falls
    back to stacking the per-timepoint full-FOV volumes.
    """
    composite = patient.nifti_dir / f"4d_flow_mag_{patient.identifier}.nii.gz"
    if composite.exists():
        img = nib.load(str(composite))
        data = np.asarray(img.dataobj, dtype=np.float32)
        if data.ndim != 4:
            raise ValueError(f"Expected 4D magnitude (X,Y,Z,T), got shape {data.shape} from {composite}")
        logger.info(f"Loaded magnitude composite {composite.name} shape {data.shape}")
        return data, np.asarray(img.affine)

    # Fallback: stack per-timepoint full-FOV volumes in frame order.
    frames = sorted(patient.flow_mag_per_timepoint_full_fov_dir.glob("*.nii.gz"))
    if not frames:
        raise FileNotFoundError(
            f"No magnitude source for patient {patient.identifier}: neither "
            f"{composite} nor per-timepoint full-FOV volumes were found."
        )
    first = nib.load(str(frames[0]))
    affine = np.asarray(first.affine)
    vols = [np.asarray(nib.load(str(f)).dataobj, dtype=np.float32) for f in frames]
    data = np.stack(vols, axis=-1)
    logger.info(f"Stacked {len(frames)} per-timepoint magnitude volumes -> shape {data.shape}")
    return data, affine


def _load_corrected_component(patient: "Patient", comp: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a cropped ground-truth corrected velocity component ``(X,Y,Z,T)``."""
    path = patient.nifti_dir / f"4d_flow_{comp}_corr_{patient.identifier}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Corrected velocity component not found: {path}. "
            "Run build_corrected_velocities() first."
        )
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D corrected velocity (X,Y,Z,T), got shape {data.shape} from {path}")
    return data, np.asarray(img.affine)


def assemble_autoflow_inputs(
    patient: "Patient",
    *,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Path]:
    """Write ``mag_4dflow.nii.gz`` and ``vel-corrected_4dflow.nii.gz`` for auto-flow.

    Args:
        patient: Patient whose NIfTIs are assembled.
        overwrite: If ``False`` and both outputs already exist, skip and return them.
        logger: Optional logger; defaults to the patient's logger.

    Returns:
        Mapping ``{"mag": <path>, "vel": <path>}`` to the written files.
    """
    log = _resolve_logger(patient, logger)
    staging = autoflow_staging_dir(patient)
    out_mag = staging / MAG_FILENAME
    out_vel = staging / VEL_FILENAME

    if out_mag.exists() and out_vel.exists() and not overwrite:
        log.info(f"Auto-flow inputs already exist for {patient.identifier}, skipping assembly")
        return {"mag": out_mag, "vel": out_vel}

    log.info(f"Assembling auto-flow inputs for patient {patient.identifier}")

    mag_data, mag_affine = _load_magnitude(patient, log)
    X, Y, Z, T_mag = mag_data.shape

    # Determine the common number of timepoints across magnitude and all
    # corrected components, then build the (X, Y, Z, T, 3) velocity volume.
    corrected = {comp: _load_corrected_component(patient, comp) for comp in _VELOCITY_COMPONENTS}
    T = min([T_mag] + [arr.shape[3] for arr, _ in corrected.values()])
    if T != T_mag:
        log.warning(
            f"Timepoint mismatch for {patient.identifier}: magnitude has {T_mag}, "
            f"using common T={T} across magnitude and corrected velocities."
        )

    mag_data = mag_data[..., :T]
    vel = np.zeros((X, Y, Z, T, 3), dtype=np.float32)

    for comp_idx, comp in enumerate(_VELOCITY_COMPONENTS):
        corr_data, corr_affine = corrected[comp]
        ox, oy, oz = _voxel_offset(mag_affine, corr_affine)
        cx, cy, cz = corr_data.shape[:3]
        if ox + cx > X or oy + cy > Y or oz + cz > Z:
            raise ValueError(
                f"Corrected {comp} ({cx},{cy},{cz}) at offset ({ox},{oy},{oz}) "
                f"does not fit in magnitude FOV ({X},{Y},{Z})."
            )
        vel[ox:ox + cx, oy:oy + cy, oz:oz + cz, :, comp_idx] = corr_data[..., :T]
        log.debug(f"Placed {comp}_corr {corr_data.shape[:3]} at offset ({ox},{oy},{oz})")

    _save_nifti(mag_data, mag_affine, out_mag)
    _save_nifti(vel, mag_affine, out_vel)
    log.info(
        f"Wrote auto-flow inputs for {patient.identifier}: "
        f"{out_mag.name} {mag_data.shape}, {out_vel.name} {vel.shape}"
    )

    return {"mag": out_mag, "vel": out_vel}


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nii = nib.Nifti1Image(data, affine)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
    nib.save(nii, str(path))
