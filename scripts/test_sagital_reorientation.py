#!/usr/bin/env python3
"""
Test runner for the sagittal/coronal axial-aligned reorientation pipeline.

Builds the downsampled per-timepoint volumes for a small set of representative
patients (one or more axial, sagittal, coronal each) into a separate output
folder so the experimental output does NOT overwrite production
``downsampled_full_fov_<size>/`` directories.

Default output folder per patient:
    <working_dir>/patient_data/<identifier>/nifti/test_sagital_reorientation/

Use --no-reorient for an ablation that runs the same pipeline but with the
existing ``create_downsampled_reference_grid`` path on every patient (legacy
behavior). Useful to prove the new code path is bit-identical to today on
axial patients.

Examples:
    # Default test set (axial, sagittal, coronal regression)
    python scripts/test_sagital_reorientation.py \
        --config all_patients Achelney Ackdradum Epcayit

    # Force overwrite of an existing test_sagital_reorientation folder
    python scripts/test_sagital_reorientation.py \
        --config all_patients --overwrite Ackdradum

    # Ablation: run the experimental folder with the legacy grid (axial path
    # for everyone, no reorientation). Use to compare visual / numeric output.
    python scripts/test_sagital_reorientation.py \
        --config all_patients --no-reorient \
        --output-folder-name test_sagital_reorientation_legacy_grid \
        Achelney Ackdradum Epcayit
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import SimpleITK as sitk

from vascular_superenhancement.data_management.dicom_to_nifti import (
    DicomToNiftiConverter,
)
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.logger import (
    setup_dataset_logger,
    setup_patient_logger,
)
from vascular_superenhancement.utils.path_config import load_path_config


@dataclass
class PatientReport:
    patient_id: str
    success: bool
    orientation: str | None = None
    source_size: tuple[int, int, int] | None = None
    source_spacing: tuple[float, float, float] | None = None
    target_size: tuple[int, int, int] | None = None
    target_spacing: tuple[float, float, float] | None = None
    target_direction_is_identity: bool | None = None
    padding_mask_coverage: float | None = None
    output_root: Path | None = None
    error: str | None = None


def _is_identity_direction(direction: tuple[float, ...], tol: float = 1e-6) -> bool:
    expected = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return all(abs(direction[i] - expected[i]) < tol for i in range(9))


def process_patient(
    patient_id: str,
    *,
    config: str,
    target_size: tuple[int, int, int],
    output_folder_name: str,
    reorient_non_axial: bool,
    overwrite: bool,
    dataset_logger: logging.Logger,
) -> PatientReport:
    logger = setup_patient_logger(patient_id, config=config, level=logging.INFO)
    dataset_logger.info(f"Starting patient {patient_id}")
    report = PatientReport(patient_id=patient_id, success=False)

    try:
        path_config = load_path_config(config)
        patient = Patient(
            path_config=path_config,
            phonetic_id=patient_id,
            debug=False,
            overwrite_images=False,
            overwrite_catalogs=False,
            overwrite_corrected=False,
            overwrite_downsampled=overwrite,
            config=config,
            dataset_logger=dataset_logger,
        )

        ref_path = (
            patient.flow_vx_corr_per_timepoint_dir
            / f"4d_flow_vx_corr_{patient.identifier}_frame_00.nii.gz"
        )
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Corrected velocity reference not found at {ref_path}; "
                f"skip this patient or run the corrected velocity pipeline first."
            )

        source_img = sitk.ReadImage(str(ref_path))
        report.orientation = DicomToNiftiConverter.classify_orientation(source_img)
        report.source_size = tuple(source_img.GetSize())
        report.source_spacing = tuple(source_img.GetSpacing())

        patient.build_downsampled_full_fov_per_timepoint(
            target_size=target_size,
            output_folder_name=output_folder_name,
            reorient_non_axial=reorient_non_axial,
        )

        output_root = patient.nifti_dir / output_folder_name
        report.output_root = output_root

        ref_grid_path = output_root / "reference.nii.gz"
        if ref_grid_path.exists():
            ref_img = sitk.ReadImage(str(ref_grid_path))
            report.target_size = tuple(ref_img.GetSize())
            report.target_spacing = tuple(ref_img.GetSpacing())
            report.target_direction_is_identity = _is_identity_direction(
                ref_img.GetDirection()
            )

        mask_path = output_root / f"padding_support_mask_{patient.identifier}.nii.gz"
        if mask_path.exists():
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            report.padding_mask_coverage = float(mask_arr.mean())

        report.success = True
        dataset_logger.info(f"Patient {patient_id} complete")
    except Exception as exc:
        logger.exception(f"Failed patient {patient_id}")
        dataset_logger.error(f"Patient {patient_id} failed: {exc}")
        report.error = str(exc)

    return report


def _format_tuple(t: tuple[float, ...] | tuple[int, ...] | None, fmt: str) -> str:
    if t is None:
        return "-"
    return "(" + ", ".join(format(v, fmt) for v in t) + ")"


def print_summary(reports: list[PatientReport]) -> None:
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (
        f"{'Patient':<14}{'OK':<4}{'orient':<10}"
        f"{'src size':<22}{'src spacing':<28}"
        f"{'tgt size':<18}{'tgt spacing':<28}"
        f"{'identity?':<10}{'mask cov':<10}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        ok_str = "OK" if r.success else "FAIL"
        identity_str = (
            "yes" if r.target_direction_is_identity else "no"
        ) if r.target_direction_is_identity is not None else "-"
        cov_str = f"{r.padding_mask_coverage:.3f}" if r.padding_mask_coverage is not None else "-"
        print(
            f"{r.patient_id:<14}{ok_str:<4}{(r.orientation or '-'):<10}"
            f"{_format_tuple(r.source_size, 'd'):<22}"
            f"{_format_tuple(r.source_spacing, '.3f'):<28}"
            f"{_format_tuple(r.target_size, 'd'):<18}"
            f"{_format_tuple(r.target_spacing, '.3f'):<28}"
            f"{identity_str:<10}{cov_str:<10}"
        )
        if r.error:
            print(f"  -> error: {r.error}")
        if r.output_root:
            print(f"  -> output: {r.output_root}")
    print("=" * 100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "patient_ids",
        nargs="*",
        default=["Achelney", "Ackdradum", "Epcayit"],
        help="Patient IDs to process. Default: Achelney (axial), Ackdradum (sagittal), Epcayit (coronal).",
    )
    parser.add_argument("--config", default="all_patients", help="Path config name (default: all_patients)")
    parser.add_argument(
        "--output-folder-name",
        default="test_sagital_reorientation",
        help="Output folder name under each patient's nifti/ dir (default: test_sagital_reorientation)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=3,
        default=(128, 128, 64),
        metavar=("LR", "AP", "SI"),
        help="Target voxel dimensions (default: 128 128 64)",
    )
    parser.add_argument(
        "--no-reorient",
        dest="reorient",
        action="store_false",
        help="Disable axial-aligned reorientation (use legacy grid for everyone). Default: reorient enabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in the experimental folder.",
    )
    parser.set_defaults(reorient=True)
    args = parser.parse_args()

    target_size = tuple(args.target_size)

    dataset_logger = setup_dataset_logger(
        "test_sagital_reorientation",
        config=args.config,
        level=logging.INFO,
    )

    dataset_logger.info(
        f"Test build: patients={args.patient_ids}, "
        f"output_folder_name={args.output_folder_name}, "
        f"target_size={target_size}, reorient_non_axial={args.reorient}, "
        f"overwrite={args.overwrite}"
    )

    reports: list[PatientReport] = []
    for pid in args.patient_ids:
        report = process_patient(
            pid,
            config=args.config,
            target_size=target_size,
            output_folder_name=args.output_folder_name,
            reorient_non_axial=args.reorient,
            overwrite=args.overwrite,
            dataset_logger=dataset_logger,
        )
        reports.append(report)

    print_summary(reports)
    failures = [r for r in reports if not r.success]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
