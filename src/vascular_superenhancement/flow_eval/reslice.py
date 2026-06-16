"""Reslice a velocity field onto cached measurement planes (numpy/scipy only).

Reproduces auto-flow's ``slice_extraction.reslice`` sampling so flow can be
recomputed for any velocity field (e.g. model-corrected) without re-running the
auto-flow geometry chain. The plane definitions come from the cached vessel
spline CSVs; only the velocity field changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator as RGI

from .geometry import compute_unit_normal, generate_sampling_plane


def build_velocity_rgi(vel_native: np.ndarray, *, negate_vz: bool = True) -> RGI:
    """RegularGridInterpolator over ``(R, C, S, T)`` for a ``(R,C,S,T,3)`` field.

    ``negate_vz`` reproduces auto-flow's ``setup_patient_rgi`` z-flip (the patch
    that un-does the negation applied at NIfTI build time). Pass the same flag
    used to build the cached geometry so through-plane signs match.
    """
    vel = np.asarray(vel_native, dtype=float).copy()
    if vel.ndim != 5 or vel.shape[-1] != 3:
        raise ValueError(f"expected (R,C,S,T,3) velocity, got {vel.shape}")
    if negate_vz:
        vel[..., 2] = -vel[..., 2]
    grid = tuple(np.arange(n) for n in vel.shape[:4])
    return RGI(grid, vel, bounds_error=False, fill_value=0)


def reslice_through_plane(
    spline_df: pd.DataFrame,
    indices: list[int],
    vel_rgi: RGI,
    affine: np.ndarray,
    *,
    plane_dims: tuple[int, int] = (256, 256),
    resolution: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Through-plane velocity ``(R, C, n_planes, T)`` for one vessel.

    For each plane: sample the velocity vector on the plane grid at every
    timepoint and dot with the (shared) vessel unit normal.
    """
    n_t = vel_rgi.grid[3].size
    n_pts = resolution[0] * resolution[1]
    unit_normal = compute_unit_normal(spline_df)
    out = np.zeros((resolution[0], resolution[1], len(indices), n_t))

    for idx, row_idx in enumerate(indices):
        plane_rcs = generate_sampling_plane(
            spline_df, row_idx, plane_dims=plane_dims, resolution=resolution, affine=affine
        )
        flat = plane_rcs.reshape(-1, 3)
        for t in range(n_t):
            pts = np.hstack([flat, np.full((n_pts, 1), t)])
            vec = vel_rgi(pts).reshape(resolution[0], resolution[1], 3)
            out[..., idx, t] = vec @ unit_normal
    return out
