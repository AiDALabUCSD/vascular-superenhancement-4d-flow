#!/usr/bin/env python3
"""One-off migration: add ``correction_tissue_mask_<pid>.nii.gz`` to legacy
``downsampled_full_fov_128x128x64/`` folders.

Background:
-----------
The legacy downsampled folders were written before the tissue-mask /
fit-mask split was introduced. They contain everything the trainer needs
(mag, velocities, corrected velocities, direct diffs, cine, cine mask,
reference grid) EXCEPT ``correction_tissue_mask_<pid>.nii.gz``, which the
trainer uses as a loss-weighting / inside-tissue indicator.

This script computes and writes only that one missing file so we can avoid
a full Phase 4 + Phase 5 rebuild for every patient.

Methodology (verified bit-identical to a full rebuild on Achelney; see
conversation history):

  1. Load the full-resolution corrected-FOV magnitude time series from
     ``4d_flow_mag_<pid>_per_timepoint_corr_fov/`` and average across time.
  2. Run ``Patient._create_magnitude_mask(..., shrink_margin=0,
     shrink_fraction=None)`` to obtain the unshrunk tissue mask at full
     resolution.
  3. Nearest-neighbor resample that mask onto the existing
     ``reference.nii.gz`` grid in the downsampled folder.
  4. Write the resampled mask as ``correction_tissue_mask_<pid>.nii.gz``
     into the existing downsampled folder.

The script is idempotent and skips patients that already have the mask
(unless ``--overwrite`` is passed). Sagittal/skip splits are excluded by
default; sagittals need a fresh reorientation rebuild and shouldn't have
their legacy stretched downsampled folder migrated.

Usage::

    # Dry run on everything in train/validation/test:
    python scripts/add_correction_tissue_mask.py --config all_patients --dry-run

    # Real migration for the same set:
    python scripts/add_correction_tissue_mask.py --config all_patients

    # Just a couple of patients:
    python scripts/add_correction_tissue_mask.py --config all_patients \
        --patient-ids Achelney Adbankad
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import _PROJECT_ROOT, load_path_config


DOWNSAMPLED_FOLDER = "downsampled_full_fov_128x128x64"
DEFAULT_ACTIVE_SPLITS = {"train", "validation", "test"}

logger = logging.getLogger("add_tissue_mask")


def _migrate_one(
    pid: str,
    cfg,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """Write the correction_tissue_mask for one patient.

    Returns a short status string describing the outcome.
    """
    try:
        patient = Patient(path_config=cfg, phonetic_id=pid)
    except Exception as exc:
        logger.warning(f"  [{pid}] Patient load failed: {exc}")
        return "skip:load_failed"

    ds_root = patient.nifti_dir / DOWNSAMPLED_FOLDER
    if not ds_root.exists():
        return "skip:no_downsampled_folder"

    mask_path = ds_root / f"correction_tissue_mask_{pid}.nii.gz"
    if mask_path.exists() and not overwrite:
        return "skip:already_exists"

    ref_path = ds_root / "reference.nii.gz"
    if not ref_path.exists():
        return "skip:no_reference"

    full_res_dir = patient.nifti_dir / f"4d_flow_mag_{pid}_per_timepoint_corr_fov"
    if not full_res_dir.exists():
        return "skip:no_fullres_mag_dir"

    mag_files = sorted(full_res_dir.glob(f"4d_flow_mag_{pid}_frame_*.nii.gz"))
    if not mag_files:
        return "skip:no_mag_frames"

    if dry_run:
        return "would_write"

    # Full-resolution mean-across-time magnitude.
    mags = np.stack([nib.load(str(p)).get_fdata() for p in mag_files], axis=-1)
    mag_mean = mags.mean(axis=-1)

    # Use the function's defaults for threshold / sigma / rethreshold so the
    # migration always matches whatever a fresh Phase 4 build would produce.
    # Pass shrink_margin=0 explicitly to opt out of edge cropping (we want
    # the unshrunk tissue mask, not the fit mask).
    tissue, _ = Patient._create_magnitude_mask(
        mag_mean,
        shrink_margin=0,
        shrink_fraction=None,
    )

    # Wrap as SimpleITK with the source mag's affine, then resample to the
    # existing downsampled reference grid via nearest-neighbor.
    # Use float32 to match the dtype the proper Phase 4 + Phase 5 pipeline
    # writes (see Patient._create_magnitude_mask -> sitk.WriteImage of the
    # full-res tissue mask, which round-trips as float32).
    src_img = sitk.ReadImage(str(mag_files[0]))
    ref_img = sitk.ReadImage(str(ref_path))

    mask_img = sitk.GetImageFromArray(
        np.transpose(tissue.astype(np.float32), (2, 1, 0))
    )
    mask_img.CopyInformation(src_img)

    down = sitk.Resample(
        mask_img, ref_img, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0.0, mask_img.GetPixelID(),
    )
    sitk.WriteImage(down, str(mask_path))
    return "wrote"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", default="all_patients",
                   help="Path-config name (default: all_patients)")
    p.add_argument("--splits-file", default="splits_05-05-26.csv",
                   help="Splits CSV under splits/ (default: splits_05-05-26.csv)")
    p.add_argument("--patient-ids", nargs="+", default=None,
                   help="Explicit patient IDs (overrides --active-splits).")
    p.add_argument("--active-splits", nargs="+", default=sorted(DEFAULT_ACTIVE_SPLITS),
                   help="Split values to migrate when --patient-ids is not "
                        "given. Sagittal/skip splits are excluded by default.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-write the mask even if it already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would happen without writing files.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_path_config(args.config)
    splits_csv = _PROJECT_ROOT / "splits" / args.splits_file
    if not splits_csv.exists():
        logger.error(f"Splits file not found: {splits_csv}")
        return 1

    df = pd.read_csv(splits_csv)
    if args.patient_ids:
        df = df[df.patient_id.isin(args.patient_ids)]
    else:
        df = df[df.split.isin(set(args.active_splits))]
    todo = list(zip(df.patient_id.tolist(), df.split.tolist()))

    logger.info(f"Patients to consider: {len(todo)}")
    logger.info(f"Dry run: {args.dry_run}    Overwrite: {args.overwrite}")
    logger.info("")

    stats: dict[str, int] = {}
    t0 = time.time()
    for i, (pid, split) in enumerate(todo, 1):
        status = _migrate_one(pid, cfg, overwrite=args.overwrite, dry_run=args.dry_run)
        stats[status] = stats.get(status, 0) + 1
        logger.info(f"  [{i:4d}/{len(todo)}] {pid:24s} ({split:20s}) {status}")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Done in {time.time() - t0:.1f}s")
    logger.info("Summary:")
    for s, n in sorted(stats.items()):
        logger.info(f"  {s:30s}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
