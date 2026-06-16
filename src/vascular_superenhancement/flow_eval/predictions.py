"""Map auto-flow predictions into this project's RAS NIfTI space.

auto-flow writes its localizations (``max_points.csv``) and centerline splines
(``aortic_spline.csv`` / ``pulmonary_spline.csv``) with both world ``(x, y, z)``
columns (auto-flow's LPS world) and native voxel ``(r, c, s)`` columns. This
module converts those into our RAS world and our magnitude-volume voxel indices
so they overlay directly on this project's NIfTIs.

Segmentations (``segnet_*_segmentation.nii.gz``) are stored in *resliced plane*
space (one 2-D mask per measurement plane per timepoint), not the volume grid,
so they are consumed during plane sampling in Stage B rather than remapped here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .transform import LPS_TO_RAS

_SPLINE_FILES = {"aorta": "aortic_spline.csv", "pulmonary": "pulmonary_spline.csv"}


def native_world_to_ours_world(xyz: np.ndarray) -> np.ndarray:
    """auto-flow (LPS) world -> our (RAS) world: flip x and y signs."""
    pts = np.atleast_2d(np.asarray(xyz, dtype=float))
    return pts @ LPS_TO_RAS[:3, :3].T


def native_world_to_ours_voxel(xyz: np.ndarray, A_ours: np.ndarray) -> np.ndarray:
    """auto-flow world -> our magnitude-volume voxel indices (float)."""
    world = native_world_to_ours_world(xyz)
    homog = np.c_[world, np.ones(len(world))]
    return (np.linalg.inv(A_ours) @ homog.T).T[:, :3]


def localizations_in_ours(staging_dir: Path, A_ours: np.ndarray) -> pd.DataFrame:
    """Load ``max_points.csv`` and add our-world + our-voxel columns."""
    df = pd.read_csv(Path(staging_dir) / "max_points.csv")
    w = native_world_to_ours_world(df[["x", "y", "z"]].to_numpy())
    v = native_world_to_ours_voxel(df[["x", "y", "z"]].to_numpy(), A_ours)
    df[["x_ours", "y_ours", "z_ours"]] = w
    df[["i_ours", "j_ours", "k_ours"]] = v
    return df


def spline_in_ours(staging_dir: Path, vessel: str, A_ours: np.ndarray) -> pd.DataFrame:
    """Load a vessel spline CSV and add our-world + our-voxel columns."""
    if vessel not in _SPLINE_FILES:
        raise ValueError(f"vessel must be one of {list(_SPLINE_FILES)}")
    df = pd.read_csv(Path(staging_dir) / _SPLINE_FILES[vessel])
    w = native_world_to_ours_world(df[["x", "y", "z"]].to_numpy())
    v = native_world_to_ours_voxel(df[["x", "y", "z"]].to_numpy(), A_ours)
    df[["x_ours", "y_ours", "z_ours"]] = w
    df[["i_ours", "j_ours", "k_ours"]] = v
    return df
