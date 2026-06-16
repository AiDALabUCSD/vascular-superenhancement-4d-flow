"""Exact, lossless conversion between this project's RAS NIfTI space and
auto-flow's native voxel space.

Both representations are built from the *same* DICOM slices, so the map between
their voxel grids is a rigid integer signed-permutation (a transpose + axis
flips + an integer crop offset) - no interpolation. The only subtlety is the
world convention:

* This project's NIfTIs are true RAS (``aff2axcodes`` honest).
* auto-flow builds its affine straight from DICOM ``ImagePositionPatient`` /
  ``ImageOrientationPatient`` (LPS) and saves it as-is, so its "world" is LPS
  while NIfTI labels it RAS. The two worlds therefore differ by a fixed
  ``diag(-1, -1, 1)`` flip.

Putting these together, the voxel-to-voxel map from our grid to auto-flow's is::

    v_native = inv(A_native) @ LPS_TO_RAS @ A_ras @ v_ras

which (empirically, corr == 1.0 on magnitude and all three velocity components)
is an exact integer transform. Velocity *components* map with the identity:
both pipelines negate the SI channel, and a spatial reindex only relocates
voxels - it does not rotate the stored per-voxel components.

This module derives everything from affines, so it generalizes per patient
(different acquisition orientation / FOV / cropping) without hard-coded axes.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

# our RAS world  ==  LPS_TO_RAS @ (auto-flow LPS world); the matrix is its own inverse.
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# GE flow series tag for the magnitude volume (used to pick one geometry per slice).
_MAG_TAG = 2


def _vec(x) -> np.ndarray:
    """Parse a catalog cell like ``'[x, y, z]'`` (or a list) into a float array."""
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=float)
    return np.asarray(x, dtype=float)


def native_affine_from_catalog(catalog: pd.DataFrame, mag_tag: int = _MAG_TAG) -> np.ndarray:
    """Reconstruct auto-flow's native affine ``A_native`` from our 4D-flow catalog.

    Replicates auto-flow's ``build_affine`` (column cosines as axis 0, row cosines
    as axis 1, slice spacing from first/last ImagePositionPatient) using only
    catalog metadata - no DICOM reads. Verified to match auto-flow's saved affine
    exactly.

    The catalog must contain (lowercase) columns ``tag_0x0043_0x1030``,
    ``time_index``, ``slice_index``, ``imageorientation``, ``pixelspacing``,
    ``imagepositionpatient``, ``slicethickness``.
    """
    df = catalog[pd.to_numeric(catalog["tag_0x0043_0x1030"], errors="coerce") == mag_tag]
    if df.empty:
        df = catalog
    s_max = int(df["slice_index"].max())
    first = df[(df["time_index"] == 0) & (df["slice_index"] == 0)].iloc[0]
    last = df[(df["time_index"] == 0) & (df["slice_index"] == s_max)].iloc[0]

    dircos = _vec(first["imageorientation"])  # [row_x,row_y,row_z, col_x,col_y,col_z]
    F = np.zeros((3, 2))
    F[:, 0] = dircos[3:]   # column cosines -> axis 0
    F[:, 1] = dircos[0:3]  # row cosines    -> axis 1
    rowres, colres = _vec(first["pixelspacing"])[:2]
    ipp0 = _vec(first["imagepositionpatient"])
    ippL = _vec(last["imagepositionpatient"])
    n_slices = s_max + 1
    slice_spacing = (ippL - ipp0) / (n_slices - 1)

    A = np.eye(4)
    A[0:3, 0] = rowres * F[:, 0]
    A[0:3, 1] = colres * F[:, 1]
    A[0:3, 2] = slice_spacing
    A[0:3, 3] = ipp0
    return A


def voxel_transform(A_from: np.ndarray, A_to: np.ndarray,
                    world_correction: np.ndarray = LPS_TO_RAS) -> np.ndarray:
    """Return the 4x4 voxel->voxel map ``v_to = T @ v_from``.

    ``world_correction`` bridges the two world conventions (default RAS<->LPS).
    For grids derived from the same DICOMs this is an integer signed-permutation
    (+ integer offset); we round to exact integers.
    """
    T = np.linalg.inv(A_to) @ world_correction @ A_from
    T_int = np.rint(T)
    if not np.allclose(T, T_int, atol=1e-3):
        raise ValueError(
            "voxel_transform is not an integer signed-permutation; affines may be "
            f"inconsistent.\nT=\n{np.round(T, 4)}"
        )
    return T_int


def reindex_array(arr: np.ndarray, T: np.ndarray, dst_spatial_shape: tuple[int, int, int]) -> np.ndarray:
    """Map ``arr`` (spatial dims first, optional trailing dims) onto a destination
    grid via the integer voxel transform ``T`` (source->destination).

    Lossless gather: ``dst[d] = arr[inv(T) @ d]`` for in-bounds voxels, else 0.
    Trailing axes (e.g. time, components) are carried through unchanged.
    """
    src_shape = arr.shape[:3]
    trailing = arr.shape[3:]
    Tinv = np.rint(np.linalg.inv(T)).astype(int)
    L, off = Tinv[:3, :3], Tinv[:3, 3]

    di, dj, dk = np.meshgrid(*[np.arange(s) for s in dst_spatial_shape], indexing="ij")
    dst_vox = np.stack([di.ravel(), dj.ravel(), dk.ravel()])
    src_vox = L @ dst_vox + off[:, None]

    inb = np.ones(src_vox.shape[1], dtype=bool)
    for a in range(3):
        inb &= (src_vox[a] >= 0) & (src_vox[a] < src_shape[a])

    out = np.zeros(dst_spatial_shape + trailing, dtype=arr.dtype)
    dst_lin = tuple(v[inb] for v in (dst_vox[0], dst_vox[1], dst_vox[2]))
    src_lin = tuple(v[inb] for v in (src_vox[0], src_vox[1], src_vox[2]))
    out[dst_lin] = arr[src_lin]
    return out


def map_points(points_vox: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Map voxel coordinates ``(N, 3)`` through the voxel transform ``T``."""
    pts = np.atleast_2d(np.asarray(points_vox, dtype=float))
    return (T[:3, :3] @ pts.T + T[:3, 3:4]).T


def map_world_direction(direction: np.ndarray,
                        world_correction: np.ndarray = LPS_TO_RAS) -> np.ndarray:
    """Map a world-space direction (e.g. a plane normal) between the two world
    conventions. Translation-free: only the rotational/sign part applies.
    """
    d = np.asarray(direction, dtype=float)
    R = world_correction[:3, :3]
    return d @ R.T if d.ndim > 1 else R @ d
