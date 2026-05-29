"""Stage A, part 1 (runs in the *main* project env).

For each selected patient, assemble the two NIfTIs the auto-flow pipeline
consumes (``mag_4dflow.nii.gz`` + ``vel-corrected_4dflow.nii.gz``) into that
patient's auto-flow staging directory, and write a manifest CSV that
``run_autoflow_geometry.py`` (which runs in the auto-flow conda env) reads.

The manifest has two columns:
    patient_id        - the auto-flow ``patient_name``
    base_folderpath   - the directory such that
                        ``base_folderpath/patient_id/mag_4dflow.nii.gz`` exists

Usage:
  # all `validation` patients in the configured splits file
  python scripts/prepare_autoflow_inputs.py

  # explicit patients
  python scripts/prepare_autoflow_inputs.py --patients Alernscet Achelney

  # a different split label, custom manifest path, force re-assembly
  python scripts/prepare_autoflow_inputs.py --split sagittal_validation \
      --manifest working_dir/all_patients/autoflow_manifest.csv --overwrite
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.flow_eval import assemble_autoflow_inputs, autoflow_staging_dir

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
    parser.add_argument("--overwrite", action="store_true", help="Re-assemble even if inputs exist")
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
            assemble_autoflow_inputs(patient, overwrite=args.overwrite)
            staging = autoflow_staging_dir(patient)
            rows.append({"patient_id": pid, "base_folderpath": str(staging.parent)})
            logger.info(f"[{pid}] staged -> {staging}")
        except Exception as exc:  # keep going; report at the end
            logger.error(f"[{pid}] failed to assemble auto-flow inputs: {exc}")
            failures.append(pid)

    if not rows:
        logger.error("No patients were staged successfully; manifest not written.")
        return

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        # base_folderpath == patient_data/<id>/flow_measurement; parents[2] == dataset working dir
        dataset_dir = Path(rows[0]["base_folderpath"]).parents[2]
        manifest_path = dataset_dir / "autoflow_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patient_id", "base_folderpath"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote manifest with {len(rows)} patients to {manifest_path}")
    if failures:
        logger.warning(f"{len(failures)} patient(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
