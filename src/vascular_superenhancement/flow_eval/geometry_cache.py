"""Precomputed, velocity-independent flow geometry for fast on-the-fly measurement.

The measurement geometry - where each cross-section plane samples the volume, the
segmentation mask on that plane, and the plane normal - is fixed once auto-flow's
geometry chain has run. Only the velocity field changes between the GT-corrected
and model-corrected fields we want to compare.

The sample coordinates are stored **in this project's own RAS voxel space**, so
``measure()`` consumes the model's velocity field exactly as it exists in the
training loop - no per-call grid conversion. This works because:

* The spline knots are in auto-flow's LPS world; mapping them through
  ``inv(LPS_TO_RAS @ A_ours)`` lands them directly on our voxel grid (this also
  absorbs the native<->ours in-plane transpose).
* Velocity *components* are identical between the two grids (the native build only
  relocates voxels, it does not rotate per-voxel components - see
  :mod:`.transform`), so the cached effective normal (with auto-flow's ``vz``
  un-negation folded in) dots correctly against our raw ``(vx, vy, vz)``.

So we precompute (once, offline) a compact cache holding, per vessel: the our-grid
voxel coordinates of the masked plane pixels (+ their plane index), the
segmentation weight at those pixels over time, and the effective plane normal.
Measuring flow for a new velocity field is then a cheap resample at the cached
coordinates, a dot with the normal, a mask-weighted sum, and a trapezoidal
integral over the cardiac cycle - reproducing auto-flow exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .evaluate import _VESSELS
from .flow import CONVERSION_FACTOR, volumetric_flow_rate
from .geometry import compute_unit_normal, generate_sampling_plane
from .transform import LPS_TO_RAS

CACHE_FILENAME = "flow_geometry.npz"
_VESSEL_NAMES = ("aorta", "pulmonary")


def build_geometry_cache(
    staging_dir: Union[str, Path],
    bpm: float,
    ours_affine: np.ndarray,
    ours_spatial_shape: tuple[int, int, int],
    *,
    seg_threshold: float = 0.0,
    pixel_area: float = 1.0,
    plane_dims: tuple[int, int] = (256, 256),
    resolution: tuple[int, int] = (256, 256),
    out_path: Optional[Path] = None,
) -> Path:
    """Precompute the compact flow geometry cache for a patient (our RAS grid).

    Reads the cached spline CSVs + segmentations from ``staging_dir`` and writes
    ``flow_geometry.npz`` (default) into it. ``ours_affine`` / ``ours_spatial_shape``
    are this project's magnitude-volume affine and ``(X, Y, Z)`` shape, so the
    cached sample coordinates index our velocity grid directly. Pixels whose
    segmentation never exceeds ``seg_threshold`` across time are dropped (``0.0``
    keeps everything, matching auto-flow bit-for-bit).
    """
    staging = Path(staging_dir)
    # Effective affine that maps auto-flow LPS-world spline points straight to our
    # RAS voxel coordinates: ours_voxel = inv(LPS_TO_RAS @ A_ours) @ world_lps.
    affine_eff = LPS_TO_RAS @ np.asarray(ours_affine, dtype=float)

    data: dict[str, np.ndarray] = {}
    n_t = 0
    for vessel in _VESSEL_NAMES:
        spline_csv, seg_name, indices = _VESSELS[vessel]
        spline_df = pd.read_csv(staging / spline_csv)
        seg = np.asarray(nib.load(str(staging / seg_name)).get_fdata())  # (R, C, n_planes, T)
        n_t = seg.shape[-1]
        normal = compute_unit_normal(spline_df)
        normal_eff = normal * np.array([1.0, 1.0, -1.0])  # fold in auto-flow vz un-negation

        coords_list, plane_list, segw_list = [], [], []
        for plane_idx, row_idx in enumerate(indices):
            plane_vox = generate_sampling_plane(
                spline_df, row_idx, plane_dims=plane_dims, resolution=resolution, affine=affine_eff
            )  # (R, C, 3) OUR voxel coords
            seg_plane = seg[..., plane_idx, :]  # (R, C, T)
            keep = np.any(seg_plane > seg_threshold, axis=-1)
            if not keep.any():
                continue
            coords_list.append(plane_vox[keep])          # (K, 3)
            plane_list.append(np.full(int(keep.sum()), plane_idx, dtype=np.int16))
            segw_list.append(seg_plane[keep])            # (K, T)

        if coords_list:
            coords = np.concatenate(coords_list, axis=0)
            planes = np.concatenate(plane_list, axis=0)
            segw = np.concatenate(segw_list, axis=0)
        else:
            coords = np.zeros((0, 3))
            planes = np.zeros((0,), np.int16)
            segw = np.zeros((0, n_t))

        data[f"{vessel}_coords"] = coords.astype(np.float32)
        data[f"{vessel}_plane"] = planes
        data[f"{vessel}_seg"] = segw.astype(np.float32)
        data[f"{vessel}_normal"] = normal_eff.astype(np.float64)
        data[f"{vessel}_n_planes"] = np.int64(len(indices))

    data["spatial_shape"] = np.asarray(ours_spatial_shape, dtype=np.int64)
    data["bpm"] = np.float64(bpm)
    data["pixel_area"] = np.float64(pixel_area)
    data["conversion_factor"] = np.float64(CONVERSION_FACTOR)
    data["n_timepoints"] = np.int64(n_t)

    out = Path(out_path) if out_path is not None else staging / CACHE_FILENAME
    np.savez_compressed(out, **data)
    return out


def build_geometry_cache_for_patient(patient, **kwargs) -> Path:
    """Convenience: build the cache for a ``Patient`` on its velocity grid.

    The corrected-velocity NIfTIs use a cropped in-plane FOV (distinct from the
    magnitude), and that is the grid the model's velocity output lives on - so the
    cache is built against it. ``measure()`` then takes the model's velocity
    ``(vx, vy, vz)`` stacked on the last axis with no conversion. Sample points
    that fall outside the crop contribute 0, exactly as in auto-flow's
    zero-padded native volume.

    Resolves staging dir, BPM, and the velocity affine/shape from the patient's
    project NIfTIs (no DICOM pixel reads beyond the single header BPM lookup).
    """
    from .paths import autoflow_staging_dir

    staging = autoflow_staging_dir(patient)
    vel_path = patient.nifti_dir / f"4d_flow_vx_corr_{patient.identifier}.nii.gz"
    vel_img = nib.load(str(vel_path))
    return build_geometry_cache(
        staging, float(patient.bpm), vel_img.affine, tuple(vel_img.shape[:3]), **kwargs
    )


class PatientFlowGeometry:
    """Loaded flow geometry cache; measures flow for arbitrary velocity fields."""

    def __init__(self, cache_path: Union[str, Path]):
        self.path = Path(cache_path)
        self._d = dict(np.load(self.path, allow_pickle=False))
        self.bpm = float(self._d["bpm"])
        self.pixel_area = float(self._d["pixel_area"])
        self.conversion_factor = float(self._d["conversion_factor"])
        self.n_timepoints = int(self._d["n_timepoints"])
        self.spatial_shape = (
            tuple(int(x) for x in self._d["spatial_shape"]) if "spatial_shape" in self._d else None
        )

    @classmethod
    def from_staging(cls, staging_dir: Union[str, Path]) -> "PatientFlowGeometry":
        return cls(Path(staging_dir) / CACHE_FILENAME)

    def _vessel_flow(self, vessel: str, velocity: np.ndarray) -> np.ndarray:
        """Per-plane volumetric flow (L/min); ``velocity`` is our ``(X,Y,Z,T,3)``."""
        coords = self._d[f"{vessel}_coords"]            # (K, 3)
        planes = self._d[f"{vessel}_plane"]             # (K,)
        segw = self._d[f"{vessel}_seg"]                 # (K, T)
        normal = self._d[f"{vessel}_normal"]            # (3,)
        n_planes = int(self._d[f"{vessel}_n_planes"])
        K, T = segw.shape
        if K == 0:
            return np.zeros(n_planes)

        # Resample each component at the K fixed coords for all T (one trilinear
        # pass per component over the (R,C,S,T) sub-array), then dot the normal.
        rr = np.repeat(coords[:, 0], T)
        cc = np.repeat(coords[:, 1], T)
        ss = np.repeat(coords[:, 2], T)
        tt = np.tile(np.arange(T), K).astype(np.float64)
        sample_coords = np.vstack([rr, cc, ss, tt])     # (4, K*T)

        through = np.zeros(K * T)
        for c in range(3):
            samp = map_coordinates(
                velocity[..., c], sample_coords, order=1, mode="constant", cval=0.0
            )
            through += normal[c] * samp
        through = through.reshape(K, T)

        contrib = through * segw * (self.conversion_factor * self.pixel_area)
        inst = np.zeros((n_planes, T))
        np.add.at(inst, planes, contrib)
        return volumetric_flow_rate(inst, self.bpm)

    def measure(self, velocity: np.ndarray) -> dict:
        """Measure Ao/PA/Qp:Qs (L/min) from an our-grid ``(X, Y, Z, T, 3)`` field.

        ``velocity`` is this project's RAS velocity field (e.g. the model's
        corrected ``(vx, vy, vz)`` stacked on the last axis) - the same grid the
        cache coordinates were built for; no conversion is needed.
        """
        vel = np.asarray(velocity)
        if vel.ndim != 5 or vel.shape[-1] != 3:
            raise ValueError(f"expected (X,Y,Z,T,3) velocity, got {vel.shape}")
        if self.spatial_shape is not None and vel.shape[:3] != self.spatial_shape:
            raise ValueError(
                f"velocity spatial shape {vel.shape[:3]} != cached {self.spatial_shape}"
            )
        ao_pp = self._vessel_flow("aorta", vel)
        pa_pp = self._vessel_flow("pulmonary", vel)
        Ao, PA = float(np.mean(ao_pp)), float(np.mean(pa_pp))
        return {
            "aorta": {"per_plane": ao_pp, "mean": Ao, "std": float(np.std(ao_pp))},
            "pulmonary": {"per_plane": pa_pp, "mean": PA, "std": float(np.std(pa_pp))},
            "Ao": Ao,
            "PA": PA,
            "Qp_Qs": PA / Ao if Ao != 0 else float("nan"),
        }
