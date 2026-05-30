"""Stage A (runs in the *auto-flow* conda env, e.g. ``auto-flow_3-9``).

Reads the manifest produced by ``prepare_autoflow_inputs.py`` and, per patient,
(1) converts the original DICOMs + our corrected-velocity npy into auto-flow's
*native* NIfTIs, then (2) runs the auto-flow geometry chain through SegNet's
reverse preprocessing. We stop before ``compute_flow`` - the in-repo evaluator
does the flow math.

Why native conversion (not our project's NIfTIs): the pretrained LocNet/SegNet
were trained exclusively on auto-flow's native DICOM->NIfTI output (the exact
[col, row, slice] LPS layout + intensity domain + slice-flip). Feeding our
RAS/padded NIfTIs collapses localization. ``patient_to_nifti`` reproduces the
training distribution and consumes the same corrected npy our pipeline uses.

Conversion detail: our GE studies include an extra ``Tag_0043_1030 == 7`` series
that breaks auto-flow's strict ``identify_4_real_series`` (it requires exactly
``{2,3,4,5}``). We therefore build ``flow_info.csv`` directly with a ``{2,3,4,5}``
tag filter (tag 7 is ignored by ``fill_volume_arrays`` anyway) and skip
auto-flow's ``parse_patient``/``filter_and_save_4d_flow``.

Chain:
    [convert] -> prepare_for_locnet -> run_locnet -> reverse_preprocessing(locnet)
        -> extract_from_locnet -> generate_spline -> reslice
        -> prepare_for_segnet -> run_segnet -> reverse_segmentation

Usage (in the auto-flow env):
  python scripts/run_autoflow_geometry.py --manifest working_dir/all_patients/autoflow_manifest.csv

  # single patient without a manifest
  python scripts/run_autoflow_geometry.py \
      --patient-id Alernscet \
      --base-dicom-folder /.../all_patients/unzipped_files \
      --base-velocity-folder /.../all_patients/corrected_velocities \
      --base-output-folder /.../patient_data/Alernscet/flow_measurement
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_autoflow_geometry")

# Sampling indices along each spline (auto-flow defaults; see reslice_driver).
AORTIC_INDICES = [5, 10, 15, 20, 25]
PULMONARY_INDICES = [5, 15, 25, 35, 45]

# GE flow series tags: 2=magnitude, 3=vx(RL), 4=vy(AP), 5=vz(SI). Tag 7 is dropped.
FLOW_TAGS = [2, 3, 4, 5]

MANIFEST_FIELDS = ["patient_id", "base_dicom_folder", "base_velocity_folder", "base_output_folder"]


def _read_manifest(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _convert_patient(
    patient_id: str,
    base_dicom_folder: str,
    base_velocity_folder: str,
    base_output_folder: str,
    *,
    overwrite: bool,
) -> None:
    """Build native ``mag_4dflow.nii.gz`` + ``vel-corrected_4dflow.nii.gz``.

    Parses the patient's DICOMs, writes a ``flow_info.csv`` restricted to the 4
    flow series, then runs auto-flow's ``patient_to_nifti``.
    """
    import numpy as np
    import pandas as pd

    from auto_flow_pipeline.data_io.parse_dicom_files import parse_dicom_folder
    from auto_flow_pipeline.data_io.dicom_to_nifti import patient_to_nifti

    patient_out = os.path.join(base_output_folder, patient_id)
    os.makedirs(patient_out, exist_ok=True)

    mag_path = os.path.join(patient_out, "mag_4dflow.nii.gz")
    cor_path = os.path.join(patient_out, "vel-corrected_4dflow.nii.gz")
    if os.path.exists(mag_path) and os.path.exists(cor_path) and not overwrite:
        logger.info(f"[{patient_id}] native NIfTIs already exist; skipping conversion")
        return

    npy_path = os.path.join(base_velocity_folder, f"{patient_id}.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Corrected velocity npy not found: {npy_path}")
    vel_shape = np.load(npy_path, mmap_mode="r").shape  # (T, 3, Z, Y, X)
    tdim = int(vel_shape[0])

    logger.info(f"[{patient_id}] (convert 1/2) parsing DICOMs in {base_dicom_folder}/{patient_id}")
    dicom_folder = os.path.join(base_dicom_folder, patient_id)
    df = parse_dicom_folder(dicom_folder, logger)

    df = df[df["Tag_0043_1030"].astype(float).isin(FLOW_TAGS)].copy()
    if df.empty:
        raise ValueError(f"[{patient_id}] no flow-series DICOMs (tags {FLOW_TAGS}) found")
    inst = df["InstanceNumber"].astype(int) - 1
    df["time_index"] = inst % tdim
    df["slice_index"] = inst // tdim
    df["vel_npy_shape"] = str(tuple(vel_shape))
    df.sort_values(by=["time_index", "slice_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n_series = df["SeriesInstanceUID"].nunique()
    logger.info(
        f"[{patient_id}] flow_info: {len(df)} rows, {n_series} series, "
        f"tags {sorted(df['Tag_0043_1030'].unique())}, "
        f"{df['slice_index'].max() + 1} slices x {df['time_index'].max() + 1} tp"
    )
    df.to_csv(os.path.join(patient_out, "flow_info.csv"), index=False)
    df.to_pickle(os.path.join(patient_out, "flow_info.pkl"))

    logger.info(f"[{patient_id}] (convert 2/2) patient_to_nifti")
    patient_to_nifti(patient_id, base_dicom_folder, base_output_folder, base_velocity_folder, overwrite=True)


def _run_geometry(patient_id: str, base_folderpath: str, locnet, segnet, *, make_gifs: bool) -> None:
    import nibabel as nib

    from auto_flow_pipeline.preprocessing.locnet.prepare_for_locnet import preprocess_nifti_for_inference
    from auto_flow_pipeline.inference.locnet.run_locnet import run_locnet_inference
    from auto_flow_pipeline.postprocessing.locnet.reverse_preprocessing import (
        reverse_preprocessing_for_patient,
    )
    from auto_flow_pipeline.slice_extraction.extract_from_locnet import find_all_max_locations
    from auto_flow_pipeline.slice_extraction.generate_spline import (
        generate_aortic_spline,
        generate_pulmonary_spline,
    )
    from auto_flow_pipeline.slice_extraction.reslice import (
        setup_patient_rgi,
        sample_aortic_spline,
        sample_pulmonary_spline,
    )
    from auto_flow_pipeline.preprocessing.segnet.prepare_for_segnet import compose_and_save_splines
    from auto_flow_pipeline.inference.segnet.run_segnet import run_a_and_p_segnet_inference
    from auto_flow_pipeline.postprocessing.segnet.reverse_preprocessing import reverse_segmentation

    mag_path = os.path.join(base_folderpath, patient_id, "mag_4dflow.nii.gz")
    if not os.path.exists(mag_path):
        raise FileNotFoundError(f"Missing converted input: {mag_path}")

    logger.info(f"[{patient_id}] (1/9) prepare_for_locnet")
    preprocess_nifti_for_inference(patient_id, base_folderpath, overwrite=False)

    logger.info(f"[{patient_id}] (2/9) run_locnet")
    run_locnet_inference(locnet, base_folderpath, patient_id)

    logger.info(f"[{patient_id}] (3/9) reverse_preprocessing (locnet)")
    reverse_preprocessing_for_patient(
        patient_id, base_folderpath, should_generate_gif=make_gifs, logger=logger
    )

    logger.info(f"[{patient_id}] (4/9) extract_from_locnet")
    find_all_max_locations(patient_name=patient_id, base_folderpath=base_folderpath, timepoint=3)

    logger.info(f"[{patient_id}] (5/9) generate_spline")
    generate_aortic_spline(patient_id, base_folderpath)
    generate_pulmonary_spline(patient_id, base_folderpath)

    logger.info(f"[{patient_id}] (6/9) reslice")
    mag_rgi, flow_rgi = setup_patient_rgi(patient_id, base_folderpath)
    affine = nib.load(mag_path).affine
    sample_aortic_spline(patient_id, base_folderpath, AORTIC_INDICES, mag_rgi, flow_rgi, affine)
    sample_pulmonary_spline(patient_id, base_folderpath, PULMONARY_INDICES, mag_rgi, flow_rgi, affine)

    logger.info(f"[{patient_id}] (7/9) prepare_for_segnet")
    compose_and_save_splines(patient_id, base_folderpath)

    logger.info(f"[{patient_id}] (8/9) run_segnet")
    run_a_and_p_segnet_inference(segnet, base_folderpath, patient_id)

    logger.info(f"[{patient_id}] (9/9) reverse_segmentation")
    reverse_segmentation(patient_id, base_folderpath, logger=logger)

    logger.info(f"[{patient_id}] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="CSV with columns: " + ", ".join(MANIFEST_FIELDS))
    parser.add_argument("--patient-id", help="Single patient id (alternative to --manifest)")
    parser.add_argument("--base-dicom-folder", help="Folder containing <patient_id>/ unzipped DICOMs")
    parser.add_argument("--base-velocity-folder", help="Folder containing <patient_id>.npy corrected velocities")
    parser.add_argument("--base-output-folder", help="Staging base; outputs go to <base>/<patient_id>/")
    parser.add_argument("--skip-convert", action="store_true", help="Assume native NIfTIs already exist")
    parser.add_argument("--overwrite-convert", action="store_true", help="Re-run DICOM->NIfTI conversion")
    parser.add_argument("--make-gifs", action="store_true", help="Also generate auto-flow QA GIFs (slower)")
    args = parser.parse_args()

    if args.manifest:
        rows = _read_manifest(args.manifest)
    elif args.patient_id and args.base_dicom_folder and args.base_velocity_folder and args.base_output_folder:
        rows = [{
            "patient_id": args.patient_id,
            "base_dicom_folder": args.base_dicom_folder,
            "base_velocity_folder": args.base_velocity_folder,
            "base_output_folder": args.base_output_folder,
        }]
    else:
        parser.error("Provide --manifest, or all of --patient-id/--base-dicom-folder/"
                     "--base-velocity-folder/--base-output-folder.")

    if not rows:
        logger.warning("Manifest is empty; nothing to do.")
        return

    from auto_flow_pipeline.inference.locnet.model_loader import load_locnet
    from auto_flow_pipeline.inference.segnet.model_loader import load_segnet

    logger.info("Loading LocNet and SegNet (once)...")
    locnet = load_locnet()
    segnet = load_segnet()
    logger.info("Models loaded.")

    failures: list[str] = []
    for row in rows:
        pid = row["patient_id"]
        base_out = row["base_output_folder"]
        logger.info("=" * 60)
        logger.info(f"Processing {pid}")
        try:
            if not args.skip_convert:
                _convert_patient(
                    pid,
                    row["base_dicom_folder"],
                    row["base_velocity_folder"],
                    base_out,
                    overwrite=args.overwrite_convert,
                )
            _run_geometry(pid, base_out, locnet, segnet, make_gifs=args.make_gifs)
        except Exception as exc:
            logger.exception(f"[{pid}] FAILED: {exc}")
            failures.append(pid)

    logger.info("=" * 60)
    logger.info(f"Completed {len(rows) - len(failures)}/{len(rows)} patients.")
    if failures:
        logger.warning(f"Failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
