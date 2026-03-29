"""
Create PEC-only DICOMs: original magnitude + CNN-corrected velocities.

Unlike hdf5_to_dicoms.py (which uses VSE-predicted magnitude), this script
pairs the *original* magnitude NIfTIs with CNN-corrected velocity NIfTIs,
so PEC can be demonstrated independently of VSE.

Usage:
  python scripts/pec_only_dicoms.py --patients Diequipi
"""

import argparse
import zipfile
import logging
from pathlib import Path

from pydicom.uid import generate_uid

from vascular_superenhancement.utils.path_config import load_path_config, _PROJECT_ROOT
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.data_management.nifti_to_dicom import NiftiToDicomConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def write_pec_only_dicoms(
    patient_ids: list[str],
    nifti_vel_root: Path,
    dicom_output_root: Path,
    pc,
) -> None:
    """Write DICOMs from original magnitude + CNN-corrected velocities."""
    for pid in patient_ids:
        logger.info(f"{'='*60}")
        logger.info(f"Writing PEC-only DICOMs for {pid}")
        logger.info(f"{'='*60}")

        patient = Patient(path_config=pc, phonetic_id=pid, debug=False, config="all_patients")
        converter = NiftiToDicomConverter.from_patient(patient)
        num_tp = patient.num_timepoints

        mag_dir = patient.flow_mag_per_timepoint_dir
        vel_dir = nifti_vel_root / pid

        if not mag_dir.exists():
            logger.warning(f"  Original mag dir not found: {mag_dir}, skipping")
            continue
        if not vel_dir.exists():
            logger.warning(f"  Vel dir not found: {vel_dir}, skipping")
            continue

        study_uid = generate_uid()
        series_uids = {
            2: generate_uid(),
            3: generate_uid(),
            4: generate_uid(),
            5: generate_uid(),
        }

        out_dir = dicom_output_root / pid / "pec_only"

        for t in range(num_tp):
            mag_path = mag_dir / f"4d_flow_mag_{pid}_frame_{t:02d}.nii.gz"
            if not mag_path.exists():
                logger.warning(f"  Missing mag frame {t} for {pid}, skipping timepoint")
                continue

            vel_paths = {
                "vx": vel_dir / "4d_flow_vx_cnn_corr" / f"4d_flow_vx_cnn_corr_{pid}_frame_{t:02d}.nii.gz",
                "vy": vel_dir / "4d_flow_vy_cnn_corr" / f"4d_flow_vy_cnn_corr_{pid}_frame_{t:02d}.nii.gz",
                "vz": vel_dir / "4d_flow_vz_cnn_corr" / f"4d_flow_vz_cnn_corr_{pid}_frame_{t:02d}.nii.gz",
            }
            missing = [k for k, p in vel_paths.items() if not p.exists()]
            if missing:
                logger.warning(f"  Missing velocity components {missing} at t={t} for {pid}, skipping")
                continue

            converter.write_timepoint_with_velocities_to_dicoms(
                mag_prediction_path=mag_path,
                velocity_paths=vel_paths,
                output_dir=out_dir,
                timepoint=t,
                study_uid=study_uid,
                series_uids=series_uids,
                overwrite=True,
                series_descriptions={
                    2: "Original Mag",
                    3: "CNN Corrected Vx",
                    4: "CNN Corrected Vy",
                    5: "CNN Corrected Vz",
                },
            )
            logger.info(f"  t={t}: done")

        zip_path = dicom_output_root / pid / f"{pid}_pec_only.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(out_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, arcname=fp.relative_to(out_dir))
        logger.info(f"  Zipped to {zip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create PEC-only DICOMs (original mag + CNN-corrected velocities)",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        required=True,
        help="Patient IDs to process",
    )
    args = parser.parse_args()

    pc = load_path_config("all_patients")

    nifti_vel_root = _PROJECT_ROOT / "working_dir" / "cnn_corrected_niftis"
    dicom_output_root = _PROJECT_ROOT / "working_dir" / "cnn_corrected_dicoms"

    logger.info(f"Velocity NIfTIs root: {nifti_vel_root}")
    logger.info(f"DICOM output root: {dicom_output_root}")
    logger.info(f"Patients: {args.patients}")

    write_pec_only_dicoms(args.patients, nifti_vel_root, dicom_output_root, pc)

    logger.info(f"\nAll done. DICOMs in: {dicom_output_root}")


if __name__ == "__main__":
    main()
