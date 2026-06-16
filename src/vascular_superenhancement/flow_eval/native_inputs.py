"""Build auto-flow's native NIfTI inputs from this project's existing NIfTIs.

The pretrained LocNet/SegNet require auto-flow's native voxel layout, but that
layout is an exact (lossless) integer transform of our RAS NIfTIs - so we can
produce ``mag_4dflow.nii.gz`` and ``vel-corrected_4dflow.nii.gz`` from artifacts
this repo already has, with *zero* DICOM reads (see :mod:`.transform`).

Outputs match auto-flow's own ``patient_to_nifti`` exactly (verified corr == 1.0
on magnitude and all three velocity components).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import nibabel as nib
import numpy as np

from .paths import autoflow_staging_dir
from .transform import native_affine_from_catalog, reindex_array, voxel_transform

if TYPE_CHECKING:  # avoid import cycle
    from ..data_management.patients import Patient

MAG_FILENAME = "mag_4dflow.nii.gz"
VEL_FILENAME = "vel-corrected_4dflow.nii.gz"

# auto-flow's native grid is (rows, cols, slices) with the in-plane axes
# transposed vs ours; the slice count matches. We take the full magnitude FOV.
_NATIVE_INPLANE_FROM = ("y", "x")  # documentation only; mapping is affine-derived


def _logger(patient: "Patient", logger: Optional[logging.Logger]) -> logging.Logger:
    return logger or getattr(patient, "_logger", None) or logging.getLogger(__name__)


def build_native_inputs(
    patient: "Patient",
    *,
    staging_dir: Optional[Path] = None,
    overwrite: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Path]:
    """Write native ``mag_4dflow.nii.gz`` + ``vel-corrected_4dflow.nii.gz``.

    Returns a dict with keys ``mag`` and ``vel`` mapping to the written paths.
    """
    log = _logger(patient, logger)
    staging = Path(staging_dir) if staging_dir is not None else autoflow_staging_dir(patient)
    staging.mkdir(parents=True, exist_ok=True)
    mag_out = staging / MAG_FILENAME
    vel_out = staging / VEL_FILENAME

    if mag_out.exists() and vel_out.exists() and not overwrite:
        log.info(f"[{patient.identifier}] native inputs already exist; skipping")
        return {"mag": mag_out, "vel": vel_out}

    catalog = patient.dicom_catalog_4d_flow
    if catalog is None:
        raise ValueError(f"[{patient.identifier}] no 4D-flow catalog available")
    A_native = native_affine_from_catalog(catalog)

    # Native grid = our magnitude FOV with in-plane axes transposed (slices match).
    mag_path = patient.nifti_dir / f"4d_flow_mag_{patient.identifier}.nii.gz"
    mag_img = nib.load(str(mag_path))
    mag = np.asanyarray(mag_img.dataobj)  # (X, Y, Z, T)
    n_slices = mag.shape[2]
    # destination in-plane size = transpose of our (X, Y); slices unchanged.
    dst_shape = (mag.shape[1], mag.shape[0], n_slices)

    T_mag = voxel_transform(mag_img.affine, A_native)
    native_mag = reindex_array(mag, T_mag, dst_shape).astype(np.int16)
    nib.save(nib.Nifti1Image(native_mag, A_native), str(mag_out))
    log.info(f"[{patient.identifier}] wrote {mag_out.name} {native_mag.shape}")

    comps = []
    for c in ("x", "y", "z"):
        vimg = nib.load(str(patient.nifti_dir / f"4d_flow_v{c}_corr_{patient.identifier}.nii.gz"))
        v = np.asanyarray(vimg.dataobj)  # (X, Yc, Z, T) cropped FOV
        T_v = voxel_transform(vimg.affine, A_native)
        comps.append(reindex_array(v, T_v, dst_shape))
    native_vel = np.stack(comps, axis=-1).astype(np.int16)  # (R, C, S, T, 3)
    nib.save(nib.Nifti1Image(native_vel, A_native), str(vel_out))
    log.info(f"[{patient.identifier}] wrote {vel_out.name} {native_vel.shape}")

    return {"mag": mag_out, "vel": vel_out}
