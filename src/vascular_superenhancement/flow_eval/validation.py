"""Downsampled-grid flow geometry for in-loop validation.

During training the dual-task model operates on the downsampled volumes
(``downsampled_full_fov_128x128x64``) one timepoint at a time, emitting a
VENC-normalised correction field. To measure aortic / pulmonary flow from that
output we reuse the auto-flow geometry chain (splines + plane segmentations) but
build the sample-coordinate cache against the **downsampled** velocity grid
rather than the full-resolution corrected-velocity grid.

This was validated to reproduce the full-resolution reference to ~1% on
``Alernscet`` (Ao 4.379 vs 4.432, PA 5.700 vs 5.759, Qp:Qs 1.302 vs 1.299): the
downsampled velocity NIfTIs are already in mm/s (no unit rescale), TorchIO and
nibabel load them in identical array order, and the cached effective normal /
sign convention carries over unchanged.

Everything here is numpy/nibabel only; the model forward pass that produces the
model-corrected field lives in
:class:`~vascular_superenhancement.training.callbacks.flow_validation_callback.FlowValidationCallback`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd

from .geometry_cache import build_geometry_cache
from .paths import autoflow_staging_dir

DS_CACHE_FILENAME = "flow_geometry_downsampled.npz"


def localization_is_valid(staging_dir: Union[str, Path]) -> bool:
    """Return True iff the auto-flow LocNet localization is non-degenerate.

    When LocNet fails to detect a landmark, its heatmap argmax falls back to
    voxel ``(0, 0, 0)``. Any such landmark drags the vessel spline to the volume
    corner, producing geometry that is either crash-inducing (pulmonary spline
    collapses to a point) or silently wrong (spline stretched to the origin).
    A clean localization has zero landmarks at ``(r, c, s) == (0, 0, 0)``.
    """
    mp = Path(staging_dir) / "max_points.csv"
    if not mp.exists():
        return False
    df = pd.read_csv(mp)
    if not {"r", "c", "s"}.issubset(df.columns):
        return False
    missed = (df["r"] == 0) & (df["c"] == 0) & (df["s"] == 0)
    return not bool(missed.any())


def downsampled_cache_path(patient) -> Path:
    """Location of the downsampled-grid cache for ``patient``."""
    return autoflow_staging_dir(patient) / DS_CACHE_FILENAME


def _ds_root(patient, downsampled_folder: str) -> Path:
    return patient.nifti_dir / downsampled_folder


def _frame_path(patient, downsampled_folder: str, comp: str, t: int, corrected: bool) -> Path:
    """Path to one per-frame downsampled velocity component NIfTI."""
    suffix = "_corr" if corrected else ""
    name = f"4d_flow_v{comp}{suffix}"
    return _ds_root(patient, downsampled_folder) / name / f"{name}_{patient.identifier}_frame_{t:02d}.nii.gz"


def build_downsampled_cache(
    patient,
    downsampled_folder: str,
    *,
    overwrite: bool = False,
    seg_threshold: float = 0.0,
) -> Optional[Path]:
    """Build (or reuse) the downsampled-grid flow-geometry cache for ``patient``.

    Returns the cache path, or ``None`` when the prerequisites are missing (the
    auto-flow geometry chain has not produced splines for this patient yet, or
    the downsampled velocity volumes are absent).
    """
    staging = autoflow_staging_dir(patient)
    out = staging / DS_CACHE_FILENAME
    if out.exists() and not overwrite:
        return out
    if not (staging / "aortic_spline.csv").exists():
        return None
    if not localization_is_valid(staging):
        return None
    vx0 = _frame_path(patient, downsampled_folder, "x", 0, corrected=False)
    if not vx0.exists():
        return None
    img = nib.load(str(vx0))
    return build_geometry_cache(
        staging,
        float(patient.bpm),
        img.affine,
        tuple(int(s) for s in img.shape[:3]),
        seg_threshold=seg_threshold,
        out_path=out,
    )


def load_downsampled_velocity(
    patient,
    downsampled_folder: str,
    n_timepoints: int,
    *,
    corrected: bool,
    max_workers: int = 8,
) -> np.ndarray:
    """Load a downsampled velocity field as ``(X, Y, Z, T, 3)`` in mm/s.

    ``corrected=False`` loads the uncorrected acquisition (``4d_flow_v*``);
    ``corrected=True`` loads the GT phase-corrected field (``4d_flow_v*_corr``).
    The per-frame files are read concurrently (IO-bound, tiny gzip volumes).
    """
    tasks = [(comp, t) for comp in ("x", "y", "z") for t in range(n_timepoints)]

    def _load(args):
        comp, t = args
        path = _frame_path(patient, downsampled_folder, comp, t, corrected)
        return comp, t, np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

    buckets = {comp: [None] * n_timepoints for comp in ("x", "y", "z")}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for comp, t, arr in ex.map(_load, tasks):
            buckets[comp][t] = arr
    comps = [np.stack(buckets[comp], axis=-1) for comp in ("x", "y", "z")]
    return np.stack(comps, axis=-1)  # (X, Y, Z, T, 3)


def load_downsampled_mag_frames(
    patient,
    downsampled_folder: str,
    n_timepoints: int,
    *,
    max_workers: int = 8,
) -> List[np.ndarray]:
    """Load the centre-magnitude frames ``[t0, t1, ...]`` as raw ``(X, Y, Z)`` arrays.

    Returned unnormalised (the caller applies the same per-frame [0, 1] rescale
    the training transform uses). Frames are read concurrently and indexed by
    timepoint, so callers can assemble any temporal-offset window without
    re-reading from disk.
    """
    root = patient.nifti_dir / downsampled_folder / "4d_flow_mag"
    pid = patient.identifier

    def _load(t):
        path = root / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
        return t, np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

    frames: List[Optional[np.ndarray]] = [None] * n_timepoints
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for t, arr in ex.map(_load, range(n_timepoints)):
            frames[t] = arr
    return frames
