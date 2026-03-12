"""
Convert CNN-corrected velocity data from HDF5 to DICOMs.

Pipeline:
  1. Load HDF5 velocity data (D0, D1, Z, T, 3)
  2. Transpose axes (1,0,2) to match NIfTI convention
  3. Flip z-axis for S_to_I patients
  4. Save per-component, per-timepoint NIfTIs with padded FOV affine
  5. Write DICOMs using NiftiToDicomConverter with blend_50 mag from inference run
  6. Zip the output DICOMs

Usage:
  python scripts/hdf5_to_dicoms.py
"""

import argparse
import zipfile
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import h5py
from pydicom.uid import generate_uid

from vascular_superenhancement.utils.path_config import load_path_config, _PROJECT_ROOT
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.data_management.nifti_to_dicom import NiftiToDicomConverter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VZ_FLOW_TAG = 5
COMP_NAMES = ["vx", "vy", "vz"]


def get_orig_direction(patient_data_dir: Path, pid: str) -> str:
    catalog_path = patient_data_dir / pid / f"dicom_catalog_{pid}.csv"
    if not catalog_path.exists():
        return "UNKNOWN"
    catalog = pd.read_csv(catalog_path)
    vz_cat = catalog[catalog["tag_0x0043_0x1030"] == VZ_FLOW_TAG].copy()
    if len(vz_cat) == 0:
        return "UNKNOWN"
    vz_cat["time_index"] = (vz_cat["instancenumber"] - 1) % vz_cat["cardiacnumberofimages"]
    vz_cat["slice_index"] = (vz_cat["instancenumber"] - 1) // vz_cat["cardiacnumberofimages"]
    vz_cat["z"] = vz_cat["imagepositionpatient"].apply(lambda x: np.array(eval(x))[2])
    t0 = vz_cat[vz_cat["time_index"] == 0].sort_values("slice_index")
    z_diff = np.diff(t0["z"].values)
    if np.sum(z_diff > 0) > np.sum(z_diff < 0):
        return "I_to_S"
    elif np.sum(z_diff < 0) > np.sum(z_diff > 0):
        return "S_to_I"
    return "AMBIGUOUS"


def convert_hdf5_to_niftis(
    hdf5_path: Path,
    patient_data_dir: Path,
    output_root: Path,
    patient_ids: list[str],
) -> None:
    """Convert HDF5 velocity data to per-timepoint NIfTIs."""
    with h5py.File(hdf5_path, "r") as f:
        for pi, pid in enumerate(patient_ids):
            orig_dir = get_orig_direction(patient_data_dir, pid)
            flip_z = orig_dir == "S_to_I"
            logger.info(
                f"[{pi+1}/{len(patient_ids)}] {pid} ({orig_dir})"
                f"{' [flip z]' if flip_z else ''}"
            )

            if pid not in f:
                logger.warning(f"  {pid} not in HDF5, skipping")
                continue

            ref_nii_path = (
                patient_data_dir / pid / "nifti"
                / f"4d_flow_vx_{pid}_per_timepoint_full_fov"
                / f"4d_flow_vx_{pid}_frame_00.nii.gz"
            )
            if not ref_nii_path.exists():
                logger.warning(f"  Reference NIfTI not found for {pid}, skipping")
                continue
            ref_affine = nib.load(str(ref_nii_path)).affine

            hdf5_data = f[pid][:].astype(np.float32)
            hdf5_data = np.transpose(hdf5_data, (1, 0, 2, 3, 4))
            if flip_z:
                hdf5_data = hdf5_data[:, :, ::-1, :, :]

            n_t = hdf5_data.shape[3]
            pid_out_dir = output_root / pid
            pid_out_dir.mkdir(parents=True, exist_ok=True)

            for c_idx, comp in enumerate(COMP_NAMES):
                comp_dir = pid_out_dir / f"4d_flow_{comp}_cnn_corr"
                comp_dir.mkdir(exist_ok=True)
                for t in range(n_t):
                    vol = hdf5_data[:, :, :, t, c_idx]
                    nii = nib.Nifti1Image(vol, ref_affine)
                    out_path = comp_dir / f"4d_flow_{comp}_cnn_corr_{pid}_frame_{t:02d}.nii.gz"
                    nib.save(nii, out_path)

            logger.info(f"  Saved {n_t} timepoints × 3 components to {pid_out_dir}")


def write_dicoms(
    patient_ids: list[str],
    nifti_root: Path,
    inference_root: Path,
    mag_method: str,
    dicom_output_root: Path,
    pc,
) -> None:
    """Write DICOMs from CNN-corrected velocity NIfTIs + predicted mag."""
    for pid in patient_ids:
        logger.info(f"{'='*60}")
        logger.info(f"Writing DICOMs for {pid}")
        logger.info(f"{'='*60}")

        patient = Patient(path_config=pc, phonetic_id=pid, debug=False, config="all_patients")
        converter = NiftiToDicomConverter.from_patient(patient)
        num_tp = patient.num_timepoints

        mag_dir = inference_root / pid / "predicted_mag" / mag_method
        vel_dir = nifti_root / pid

        if not mag_dir.exists():
            logger.warning(f"  Mag dir not found: {mag_dir}, skipping")
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

        out_dir = dicom_output_root / pid / "cnn_corrected"

        for t in range(num_tp):
            mag_path = mag_dir / f"pred_mag_{pid}_frame_{t:02d}.nii.gz"
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
                    2: f"CNN Mag ({mag_method})",
                    3: "CNN Corrected Vx",
                    4: "CNN Corrected Vy",
                    5: "CNN Corrected Vz",
                },
            )
            logger.info(f"  t={t}: done")

        zip_path = dicom_output_root / pid / f"{pid}_cnn_corrected.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(out_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, arcname=fp.relative_to(out_dir))
        logger.info(f"  Zipped to {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 CNN-corrected velocities to DICOMs")
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=_PROJECT_ROOT / "working_dir" / "cardiac_processed_cnn_corrected.hdf5",
        help="Path to HDF5 file with CNN-corrected velocity data",
    )
    parser.add_argument(
        "--inference-run",
        type=str,
        default="glowing-microwave_epoch-69",
        help="Name of the inference run directory",
    )
    parser.add_argument(
        "--mag-method",
        type=str,
        default="blend_50",
        help="Magnitude reconstruction method",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Patient IDs to process (default: all patients in HDF5)",
    )
    args = parser.parse_args()

    pc = load_path_config("all_patients")
    patient_data_dir = pc.working_dir / "patient_data"

    if args.patients:
        patient_ids = args.patients
    else:
        with h5py.File(args.hdf5_path, "r") as f:
            patient_ids = sorted(list(f.keys()))
        logger.info(f"Found {len(patient_ids)} patients in HDF5")

    nifti_root = _PROJECT_ROOT / "working_dir" / "cnn_corrected_niftis"
    inference_root = _PROJECT_ROOT / "working_dir" / "inference" / args.inference_run
    dicom_output_root = _PROJECT_ROOT / "working_dir" / "cnn_corrected_dicoms"

    logger.info(f"HDF5: {args.hdf5_path}")
    logger.info(f"Inference run: {args.inference_run}")
    logger.info(f"Mag method: {args.mag_method}")
    logger.info(f"Patients: {patient_ids}")

    logger.info("\n=== Step 1: Converting HDF5 → NIfTIs ===")
    convert_hdf5_to_niftis(args.hdf5_path, patient_data_dir, nifti_root, patient_ids)

    logger.info("\n=== Step 2: Writing DICOMs ===")
    write_dicoms(patient_ids, nifti_root, inference_root, args.mag_method, dicom_output_root, pc)

    logger.info(f"\nAll done. DICOMs in: {dicom_output_root}")


if __name__ == "__main__":
    main()
