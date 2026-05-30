"""Stage A, part 1 (runs in the *main* project env).

Resolves, for each selected patient, the paths the auto-flow native conversion
needs and writes a manifest CSV that ``run_autoflow_geometry.py`` (which runs in
the auto-flow conda env) consumes. We no longer assemble NIfTIs here: the
pretrained LocNet/SegNet require auto-flow's *native* DICOM->NIfTI output, so the
runner rebuilds inputs from the original DICOMs + our corrected-velocity npy.

Manifest columns:
    patient_id            - auto-flow ``patient_name`` (== corrected npy stem)
    base_dicom_folder     - dir such that ``base_dicom_folder/patient_id`` holds unzipped DICOMs
    base_velocity_folder  - dir holding ``patient_id.npy`` corrected velocities
    base_output_folder    - staging base; outputs go to ``base_output_folder/patient_id/``

Usage:
  # all `validation` patients in the configured splits file
  python scripts/prepare_autoflow_inputs.py

  # explicit patients
  python scripts/prepare_autoflow_inputs.py --patients Alernscet Achelney

  # different split label / custom manifest path
  python scripts/prepare_autoflow_inputs.py --split sagittal_validation \
      --manifest working_dir/all_patients/autoflow_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _select_patients(args, pc) -> list[str]:
    if args.patients:
        return list(args.patients)
    splits_path = Path(args.splits or pc.splits_path)
    df = pd.read_csv(splits_path)
    selected = df.loc[df["split"] == args.split, "patient_id"].astype(str).tolist()
    logger.info(f"Selected {len(selected)} patients with split='{args.split}' from {splits_path}")
    return selected


def _resolve_paths(patient: Patient, pid: str) -> dict[str, str]:
    """Resolve the DICOM / velocity / output bases for one patient and validate layout."""
    dicom_dir = Path(patient.unzipped_dir)
    npy_path = Path(patient.corrected_velocity_numpy_path)
    staging_base = Path(patient.flow_geometry_dir)

    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"Unzipped DICOM dir missing: {dicom_dir}")
    if not npy_path.exists():
        raise FileNotFoundError(f"Corrected velocity npy missing: {npy_path}")
    if dicom_dir.name != pid:
        raise ValueError(f"DICOM dir name {dicom_dir.name!r} != patient_id {pid!r}")
    if npy_path.stem != pid:
        raise ValueError(f"Corrected npy stem {npy_path.stem!r} != patient_id {pid!r}")

    return {
        "patient_id": pid,
        "base_dicom_folder": str(dicom_dir.parent),
        "base_velocity_folder": str(npy_path.parent),
        "base_output_folder": str(staging_base),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="all_patients", help="path_config name (default: all_patients)")
    parser.add_argument("--patients", nargs="+", help="Explicit patient ids (overrides --split)")
    parser.add_argument("--split", default="validation", help="Split label to select (default: validation)")
    parser.add_argument("--splits", help="Override path to splits CSV (default: from path_config)")
    parser.add_argument(
        "--manifest",
        help="Output manifest CSV path (default: <dataset working dir>/autoflow_manifest.csv)",
    )
    args = parser.parse_args()

    pc = load_path_config(args.config)
    patient_ids = _select_patients(args, pc)
    if not patient_ids:
        logger.warning("No patients selected; nothing to do.")
        return

    rows: list[dict[str, str]] = []
    failures: list[str] = []
    for pid in patient_ids:
        try:
            patient = Patient(path_config=pc, phonetic_id=pid, debug=False, config=args.config)
            rows.append(_resolve_paths(patient, pid))
            logger.info(f"[{pid}] resolved -> {rows[-1]['base_output_folder']}/{pid}")
        except Exception as exc:
            logger.error(f"[{pid}] failed to resolve paths: {exc}")
            failures.append(pid)

    if not rows:
        logger.error("No patients resolved successfully; manifest not written.")
        return

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        # base_output_folder == patient_data/<id>/flow_measurement; parents[2] == dataset working dir
        dataset_dir = Path(rows[0]["base_output_folder"]).parents[2]
        manifest_path = dataset_dir / "autoflow_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["patient_id", "base_dicom_folder", "base_velocity_folder", "base_output_folder"]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote manifest with {len(rows)} patients to {manifest_path}")
    if failures:
        logger.warning(f"{len(failures)} patient(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
