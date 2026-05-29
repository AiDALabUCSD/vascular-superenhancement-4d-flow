"""Lightweight flow-rate evaluation for the dual-task velocity correction model.

This package measures aortic/pulmonary flow from a velocity field using
*precomputed* measurement geometry (a vessel-centerline spline + a cross-section
segmentation) that is produced once, offline, by the external ``auto-flow``
pipeline. Once the geometry is cached, flow is recomputed on any velocity field
with plain numpy/scipy (no TensorFlow), which makes it cheap enough to run as a
validation metric comparing model-corrected vs ground-truth-corrected velocities.

Stage A (this module): :func:`assemble_autoflow_inputs` builds the two NIfTIs the
auto-flow pipeline consumes (``mag_4dflow.nii.gz`` and
``vel-corrected_4dflow.nii.gz``) from artifacts this repo already produces,
keeping everything in our RAS affine frame so the cached geometry lines up with
our velocities.
"""

from .autoflow_inputs import assemble_autoflow_inputs, autoflow_staging_dir

__all__ = [
    "assemble_autoflow_inputs",
    "autoflow_staging_dir",
]
