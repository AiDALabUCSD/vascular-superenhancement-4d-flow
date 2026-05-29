"""Stage A, part 2 (runs in the *auto-flow* conda env, e.g. ``auto-flow_3-9``).

Reads the manifest produced by ``prepare_autoflow_inputs.py`` and runs the
auto-flow geometry chain for each patient, stopping after SegNet (we do NOT run
auto-flow's ``compute_flow`` - the in-repo evaluator does the flow math):

    prepare_for_locnet -> run_locnet -> extract_from_locnet
        -> generate_spline -> reslice -> prepare_for_segnet -> run_segnet

LocNet and SegNet are loaded once and reused across all patients. The model
paths come from auto-flow's own config (already pointed at the mounted
``final_models/{locnet,segnet}.hdf5``), so nothing here needs editing.

This script imports ``auto_flow_pipeline`` and TensorFlow, so it must run in the
auto-flow env - NOT the main project env.

Usage (in the auto-flow env):
  python scripts/run_autoflow_geometry.py --manifest working_dir/all_patients/autoflow_manifest.csv

  # single patient without a manifest
  python scripts/run_autoflow_geometry.py \
      --base-folderpath /path/patient_data/Alernscet/flow_measurement \
      --patient-id Alernscet
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


def _read_manifest(path: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pairs.append((row["patient_id"], row["base_folderpath"]))
    return pairs


def _run_patient(patient_id: str, base_folderpath: str, locnet, segnet, *, make_gifs: bool) -> None:
    import nibabel as nib

    from auto_flow_pipeline.preprocessing.locnet.prepare_for_locnet import preprocess_nifti_for_inference
    from auto_flow_pipeline.inference.locnet.run_locnet import run_locnet_inference
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

    mag_path = os.path.join(base_folderpath, patient_id, "mag_4dflow.nii.gz")
    if not os.path.exists(mag_path):
        raise FileNotFoundError(f"Missing assembled input: {mag_path}")

    logger.info(f"[{patient_id}] (1/7) prepare_for_locnet")
    preprocess_nifti_for_inference(patient_id, base_folderpath, overwrite=False)

    logger.info(f"[{patient_id}] (2/7) run_locnet")
    run_locnet_inference(locnet, base_folderpath, patient_id)

    logger.info(f"[{patient_id}] (3/7) extract_from_locnet")
    find_all_max_locations(patient_name=patient_id, base_folderpath=base_folderpath, timepoint=3)

    logger.info(f"[{patient_id}] (4/7) generate_spline")
    generate_aortic_spline(patient_id, base_folderpath)
    generate_pulmonary_spline(patient_id, base_folderpath)

    logger.info(f"[{patient_id}] (5/7) reslice")
    mag_rgi, flow_rgi = setup_patient_rgi(patient_id, base_folderpath)
    affine = nib.load(mag_path).affine
    sample_aortic_spline(patient_id, base_folderpath, AORTIC_INDICES, mag_rgi, flow_rgi, affine)
    sample_pulmonary_spline(patient_id, base_folderpath, PULMONARY_INDICES, mag_rgi, flow_rgi, affine)

    logger.info(f"[{patient_id}] (6/7) prepare_for_segnet")
    compose_and_save_splines(patient_id, base_folderpath)

    logger.info(f"[{patient_id}] (7/7) run_segnet")
    run_a_and_p_segnet_inference(segnet, base_folderpath, patient_id)

    logger.info(f"[{patient_id}] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", help="CSV with columns patient_id,base_folderpath")
    parser.add_argument("--base-folderpath", help="Single-patient base folder (alternative to --manifest)")
    parser.add_argument("--patient-id", help="Single patient id (use with --base-folderpath)")
    parser.add_argument("--make-gifs", action="store_true", help="Also generate auto-flow QA GIFs (slower)")
    args = parser.parse_args()

    if args.manifest:
        pairs = _read_manifest(args.manifest)
    elif args.base_folderpath and args.patient_id:
        pairs = [(args.patient_id, args.base_folderpath)]
    else:
        parser.error("Provide either --manifest, or both --base-folderpath and --patient-id.")

    if not pairs:
        logger.warning("Manifest is empty; nothing to do.")
        return

    from auto_flow_pipeline.inference.locnet.model_loader import load_locnet
    from auto_flow_pipeline.inference.segnet.model_loader import load_segnet

    logger.info("Loading LocNet and SegNet (once)...")
    locnet = load_locnet()
    segnet = load_segnet()
    logger.info("Models loaded.")

    failures: list[str] = []
    for patient_id, base_folderpath in pairs:
        logger.info("=" * 60)
        logger.info(f"Processing {patient_id} (base={base_folderpath})")
        try:
            _run_patient(patient_id, base_folderpath, locnet, segnet, make_gifs=args.make_gifs)
        except Exception as exc:
            logger.exception(f"[{patient_id}] FAILED: {exc}")
            failures.append(patient_id)

    logger.info("=" * 60)
    logger.info(f"Completed {len(pairs) - len(failures)}/{len(pairs)} patients.")
    if failures:
        logger.warning(f"Failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
