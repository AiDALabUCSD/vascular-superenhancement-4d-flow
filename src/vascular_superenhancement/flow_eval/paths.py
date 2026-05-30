"""Path conventions for cached auto-flow geometry.

Auto-flow's native conversion + geometry chain (run offline in the auto-flow
conda env via ``scripts/run_autoflow_geometry.py``) writes everything for a
patient under ``flow_geometry_dir/<identifier>/``. The in-repo flow evaluator
(Stage B) reads the cached geometry (splines, segmentations, resliced volumes)
from that same directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from ..data_management.patients import Patient


def autoflow_staging_dir(patient: "Patient", *, create: bool = True) -> Path:
    """Return the per-patient auto-flow staging directory.

    Auto-flow is run with ``base_output_folder = patient.flow_geometry_dir`` and
    ``patient_name = patient.identifier``, so it reads/writes everything under
    ``flow_geometry_dir/<identifier>/``.
    """
    staging = patient.flow_geometry_dir / patient.identifier
    if create:
        staging.mkdir(parents=True, exist_ok=True)
    return staging
