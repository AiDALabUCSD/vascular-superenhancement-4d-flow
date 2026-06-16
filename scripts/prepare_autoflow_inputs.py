"""Stage A, part 1 (runs in the *main* project env).

Builds auto-flow's native NIfTI inputs (``mag_4dflow.nii.gz`` +
``vel-corrected_4dflow.nii.gz``) for each selected patient directly from this
project's existing NIfTIs, with **zero DICOM reads** (see
``flow_eval.build_native_inputs``). The outputs are bit-exact vs auto-flow's own
``patient_to_nifti`` conversion, but require no DICOM parsing/pixel reads.

Then writes a manifest CSV that ``run_autoflow_geometry.py`` (auto-flow conda
env) consumes to run the geometry chain.

Manifest columns:
    patient_id           - this project's patient identifier (logging + DICOM fallback)
    autoflow_name        - auto-flow ``patient_name`` (= ``flow_measurement``)
    base_output_folder   - staging base; inputs live in ``<base>/<autoflow_name>/``
    base_dicom_folder    - (fallback only) dir with ``<patient_id>/`` unzipped DICOMs
    base_velocity_folder - (fallback only) dir with ``<patient_id>.npy``

Usage:
  python scripts/prepare_autoflow_inputs.py                       # validation split
  python scripts/prepare_autoflow_inputs.py --patients Alernscet
  python scripts/prepare_autoflow_inputs.py --split sagittal_validation --overwrite
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import pandas as pd

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.flow_eval import autoflow_staging_dir, build_native_inputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _default_splits_csv() -> Path:
    """Most recently modified ``splits/splits_*.csv`` under the repo root.

    Filenames are ``MM-DD-YY`` coded, so lexical order isn't chronological; we use
    mtime. Pass ``--splits`` explicitly to override.
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = list((repo_root / "splits").glob("splits_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No splits_*.csv found in {repo_root / 'splits'}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _select_patients(args, pc) -> list[str]:
    if args.patients:
        return list(args.patients)
    # PathConfig has no splits concept; splits live in the repo's splits/ folder.
    splits_path = Path(args.splits) if args.splits else _default_splits_csv()
    df = pd.read_csv(splits_path)
    selected = df.loc[df["split"] == args.split, "patient_id"].astype(str).tolist()
    logger.info(f"Selected {len(selected)} patients with split='{args.split}' from {splits_path}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="all_patients", help="path_config name (default: all_patients)")
    parser.add_argument("--patients", nargs="+", help="Explicit patient ids (overrides --split)")
    parser.add_argument("--split", default="validation", help="Split label to select (default: validation)")
    parser.add_argument("--splits", help="Path to splits CSV (default: newest splits/splits_*.csv by mtime)")
    parser.add_argument("--manifest", help="Output manifest CSV path (default: <dataset working dir>/autoflow_manifest.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild native inputs even if present")
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
            build_native_inputs(patient, overwrite=args.overwrite, logger=logger)
            staging = autoflow_staging_dir(patient)
            npy_path = Path(patient.corrected_velocity_numpy_path)
            rows.append({
                "patient_id": pid,
                "autoflow_name": staging.name,
                "base_output_folder": str(staging.parent),
                "base_dicom_folder": str(Path(patient.unzipped_dir).parent),
                "base_velocity_folder": str(npy_path.parent),
            })
            logger.info(f"[{pid}] native inputs ready -> {staging}")
        except Exception as exc:
            logger.error(f"[{pid}] failed: {exc}")
            failures.append(pid)

    if not rows:
        logger.error("No patients prepared successfully; manifest not written.")
        return

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        # base_output_folder == patient_data/<id> (working_dir); parents[1] == dataset working dir
        manifest_path = Path(rows[0]["base_output_folder"]).parents[1] / "autoflow_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["patient_id", "autoflow_name", "base_output_folder", "base_dicom_folder", "base_velocity_folder"]
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote manifest with {len(rows)} patients to {manifest_path}")
    if failures:
        logger.warning(f"{len(failures)} patient(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
