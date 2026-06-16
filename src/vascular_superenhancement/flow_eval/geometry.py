"""Measurement-plane geometry from auto-flow vessel splines (numpy only).

These functions reproduce auto-flow's ``slice_extraction.reslice`` plane math
exactly so the in-repo evaluator can reslice an arbitrary velocity field onto the
same cross-section planes used to build the cached geometry.

A measurement plane is the cross-section perpendicular to the vessel tangent at a
given spline knot. The tangent (== plane normal) is the unit vector from the
first to the last spline point; each plane is a uniform ``plane_dims`` mm grid
sampled at ``resolution`` and expressed in voxel (RCS) coordinates of the native
volume via its affine.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pandas as pd


def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Rotation matrix taking ``vec1`` to ``vec2`` (Rodrigues' formula)."""
    a = np.asarray(vec1, float) / np.linalg.norm(vec1)
    b = np.asarray(vec2, float) / np.linalg.norm(vec2)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    c = float(np.dot(a, b))
    if s < 1e-8:  # parallel or anti-parallel
        return np.eye(3)
    vx = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    return np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))


def compute_unit_normal(spline_df: pd.DataFrame) -> np.ndarray:
    """Unit plane normal = vessel tangent from first to last spline point."""
    first = spline_df.loc[0, ["x", "y", "z"]].to_numpy(dtype=float)
    last = spline_df.loc[len(spline_df) - 1, ["x", "y", "z"]].to_numpy(dtype=float)
    tangent = last - first
    norm = np.linalg.norm(tangent)
    if norm == 0:
        raise ValueError("The computed vessel tangent has zero length.")
    return tangent / norm


def generate_sampling_plane(
    spline_df: pd.DataFrame,
    row: int,
    plane_dims: tuple[int, int] = (256, 256),
    resolution: tuple[int, int] = (256, 256),
    affine: np.ndarray | None = None,
) -> np.ndarray:
    """Sampling grid at spline knot ``row``, in voxel (RCS) coords if ``affine`` given.

    Returns an array of shape ``(num_rows, num_cols, 3)``.
    """
    center = spline_df.loc[row, ["x", "y", "z"]].to_numpy(dtype=float)
    unit_normal = compute_unit_normal(spline_df)

    # Rotate plane so the vessel normal aligns with axial [0,0,1]; invert to map back.
    R = rotation_matrix_from_vectors(unit_normal, np.array([0.0, 0.0, 1.0]))
    R_inv = R.T

    height, width = plane_dims
    num_rows, num_cols = resolution
    xs = np.linspace(-width / 2, width / 2, num_cols)
    ys = np.linspace(-height / 2, height / 2, num_rows)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    offsets = np.stack([X, Y, Z], axis=-1)  # (num_rows, num_cols, 3)

    plane_real_world = center + np.tensordot(offsets, R_inv, axes=([2], [1]))

    if affine is not None:
        flat = plane_real_world.reshape(-1, 3)
        plane_rcs = nib.affines.apply_affine(np.linalg.inv(affine), flat)
        return plane_rcs.reshape(num_rows, num_cols, 3)
    return plane_real_world
