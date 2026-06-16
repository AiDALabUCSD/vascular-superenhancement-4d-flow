"""Path conventions for cached auto-flow geometry.

Auto-flow's native conversion + geometry chain (run offline in the auto-flow
conda env via ``scripts/run_autoflow_geometry.py``) writes everything for a
patient directly under ``flow_geometry_dir/`` (i.e. ``<working_dir>/flow_measurement/``).
The in-repo flow evaluator (Stage B) reads the cached geometry (splines,
segmentations, resliced volumes) from that same directory.

Auto-flow's API always joins ``base_folderpath / patient_name``. To avoid a
redundant per-patient subfolder (the patient is already identified by
``working_dir``), we run it with ``base_folderpath = patient.working_dir`` and
``patient_name = AUTOFLOW_NAME`` so the staging dir is exactly
``working_dir/flow_measurement`` rather than ``.../flow_measurement/<identifier>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from ..data_management.patients import Patient

# Auto-flow ``patient_name`` for our runs. Must equal ``flow_geometry_dir.name``
# so that ``base_folderpath (= working_dir) / AUTOFLOW_NAME == flow_geometry_dir``.
AUTOFLOW_NAME = "flow_measurement"


def autoflow_staging_dir(patient: "Patient", *, create: bool = True) -> Path:
    """Return the per-patient auto-flow staging directory.

    Auto-flow is run with ``base_output_folder = patient.working_dir`` and
    ``patient_name = AUTOFLOW_NAME``, so it reads/writes everything under
    ``working_dir/flow_measurement/`` (== ``patient.flow_geometry_dir``).
    """
    staging = patient.flow_geometry_dir
    if create:
        staging.mkdir(parents=True, exist_ok=True)
    return staging
