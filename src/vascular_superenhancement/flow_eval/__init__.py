"""Lightweight flow-rate evaluation for the dual-task velocity correction model.

This package measures aortic/pulmonary flow from a velocity field using
*precomputed* measurement geometry (vessel-centerline splines + cross-section
segmentations) produced once, offline, by the external ``auto-flow`` pipeline's
pretrained LocNet/SegNet. Once the geometry is cached, flow is recomputed on any
velocity field with plain numpy/scipy (no TensorFlow), cheap enough to run as a
validation metric comparing model-corrected vs ground-truth-corrected velocities.

Coordinate bridge
-----------------
auto-flow's pretrained models require auto-flow's *native* voxel layout, but that
layout is an exact, lossless integer transform of this project's RAS NIfTIs
(same DICOM slices; worlds differ only by a ``diag(-1,-1,1)`` LPS<->RAS flip).
So we:

* :func:`build_native_inputs` - produce auto-flow's native ``mag_4dflow`` +
  ``vel-corrected_4dflow`` from our existing NIfTIs with **zero DICOM reads**
  (bit-exact vs auto-flow's own conversion);
* run the geometry chain (``scripts/run_autoflow_geometry.py``, auto-flow env);
* :mod:`.predictions` - map the resulting localizations/splines back into our
  RAS space; :mod:`.transform` provides the underlying affine-derived mappers.
"""

from .paths import AUTOFLOW_NAME, autoflow_staging_dir
from .native_inputs import build_native_inputs
from .predictions import (
    localizations_in_ours,
    native_world_to_ours_voxel,
    native_world_to_ours_world,
    spline_in_ours,
)
from .transform import (
    native_affine_from_catalog,
    reindex_array,
    voxel_transform,
)
from .geometry import compute_unit_normal, generate_sampling_plane
from .reslice import build_velocity_rgi, reslice_through_plane
from .flow import instantaneous_flow, measure_flow, volumetric_flow_rate
from .evaluate import (
    AORTIC_INDICES,
    PULMONARY_INDICES,
    evaluate,
    evaluate_vessel,
    load_native_affine,
)
from .geometry_cache import (
    CACHE_FILENAME,
    PatientFlowGeometry,
    build_geometry_cache,
    build_geometry_cache_for_patient,
)
from .validation import (
    DS_CACHE_FILENAME,
    build_downsampled_cache,
    downsampled_cache_path,
    load_downsampled_mag_frames,
    load_downsampled_velocity,
    localization_is_valid,
)

__all__ = [
    "AUTOFLOW_NAME",
    "autoflow_staging_dir",
    "build_native_inputs",
    "native_affine_from_catalog",
    "voxel_transform",
    "reindex_array",
    "localizations_in_ours",
    "spline_in_ours",
    "native_world_to_ours_world",
    "native_world_to_ours_voxel",
    "compute_unit_normal",
    "generate_sampling_plane",
    "build_velocity_rgi",
    "reslice_through_plane",
    "instantaneous_flow",
    "measure_flow",
    "volumetric_flow_rate",
    "evaluate",
    "evaluate_vessel",
    "load_native_affine",
    "AORTIC_INDICES",
    "PULMONARY_INDICES",
    "build_geometry_cache",
    "build_geometry_cache_for_patient",
    "PatientFlowGeometry",
    "CACHE_FILENAME",
    "DS_CACHE_FILENAME",
    "build_downsampled_cache",
    "downsampled_cache_path",
    "load_downsampled_mag_frames",
    "load_downsampled_velocity",
    "localization_is_valid",
]
