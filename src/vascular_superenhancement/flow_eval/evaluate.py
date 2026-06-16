"""End-to-end flow evaluation from cached auto-flow geometry + a velocity field.

Given a patient's cached auto-flow staging directory (vessel spline CSVs,
segmentations, native affine) and a native-grid velocity field, this computes
aortic / pulmonary volumetric flow (L/min) and Qp:Qs - the lightweight,
TensorFlow-free path used during validation.

The velocity field is whatever you want to measure: the cached
``vel-corrected_4dflow.nii.gz`` (to reproduce auto-flow), or a model-predicted
field mapped into the native grid (see :mod:`.transform` / :mod:`.native_inputs`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from .flow import measure_flow
from .reslice import build_velocity_rgi, reslice_through_plane

# Sampling knots along each spline; must match those used to build the cached
# segmentation/through-plane volumes (scripts/run_autoflow_geometry.py).
AORTIC_INDICES = [5, 10, 15, 20, 25]
PULMONARY_INDICES = [5, 15, 25, 35, 45]

_VESSELS = {
    "aorta": ("aortic_spline.csv", "segnet_aorta_segmentation.nii.gz", AORTIC_INDICES),
    "pulmonary": ("pulmonary_spline.csv", "segnet_pulmonary_segmentation.nii.gz", PULMONARY_INDICES),
}

MAG_FILENAME = "mag_4dflow.nii.gz"
VEL_FILENAME = "vel-corrected_4dflow.nii.gz"


def _load_segmentation(staging: Path, vessel: str) -> np.ndarray:
    return np.asarray(nib.load(str(staging / _VESSELS[vessel][1])).get_fdata())


def load_native_affine(staging: Path) -> np.ndarray:
    return nib.load(str(staging / MAG_FILENAME)).affine


def evaluate_vessel(
    staging: Path,
    vessel: str,
    vel_rgi,
    affine: np.ndarray,
    bpm: float,
    *,
    seg: Optional[np.ndarray] = None,
) -> dict:
    """Flow for a single vessel; returns measure_flow dict + through-plane array."""
    spline_csv, _, indices = _VESSELS[vessel]
    spline_df = pd.read_csv(staging / spline_csv)
    through = reslice_through_plane(spline_df, indices, vel_rgi, affine)
    if seg is None:
        seg = _load_segmentation(staging, vessel)
    result = measure_flow(through, seg, bpm)
    result["through_plane"] = through
    return result


def evaluate(
    staging_dir: Union[str, Path],
    bpm: float,
    *,
    vel_native: Optional[np.ndarray] = None,
    negate_vz: bool = True,
) -> dict:
    """Measure Ao/PA/Qp:Qs from a native-grid velocity field.

    If ``vel_native`` is None, the cached ``vel-corrected_4dflow.nii.gz`` is used
    (reproduces auto-flow). Returns per-vessel results plus summary scalars.
    """
    staging = Path(staging_dir)
    affine = load_native_affine(staging)
    if vel_native is None:
        vel_native = np.asarray(nib.load(str(staging / VEL_FILENAME)).get_fdata())

    vel_rgi = build_velocity_rgi(vel_native, negate_vz=negate_vz)

    ao = evaluate_vessel(staging, "aorta", vel_rgi, affine, bpm)
    pa = evaluate_vessel(staging, "pulmonary", vel_rgi, affine, bpm)
    Ao, PA = ao["mean"], pa["mean"]
    return {
        "aorta": ao,
        "pulmonary": pa,
        "Ao": Ao,
        "PA": PA,
        "Qp_Qs": PA / Ao if Ao != 0 else float("nan"),
    }
