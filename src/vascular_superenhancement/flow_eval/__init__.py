"""Lightweight flow-rate evaluation for the dual-task velocity correction model.

This package measures aortic/pulmonary flow from a velocity field using
*precomputed* measurement geometry (a vessel-centerline spline + a cross-section
segmentation) that is produced once, offline, by the external ``auto-flow``
pipeline. Once the geometry is cached, flow is recomputed on any velocity field
with plain numpy/scipy (no TensorFlow), which makes it cheap enough to run as a
validation metric comparing model-corrected vs ground-truth-corrected velocities.

Stage A (offline, auto-flow conda env): ``scripts/run_autoflow_geometry.py``
converts the original DICOMs + our corrected-velocity npy into auto-flow's native
NIfTIs (``patient_to_nifti``) and runs the geometry chain (LocNet -> splines ->
reslice -> SegNet). The pretrained models require auto-flow's native orientation
and intensity domain, so we do NOT feed them this repo's processed NIfTIs.

This package (Stage B, main project env) reads the cached geometry from
``flow_geometry_dir/<identifier>/`` and recomputes flow.
"""

from .paths import autoflow_staging_dir

__all__ = [
    "autoflow_staging_dir",
]
