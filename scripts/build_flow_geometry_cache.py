"""Stage B precompute (runs in the *main* project env).

After the auto-flow geometry chain has produced splines + segmentations for each
patient (``run_autoflow_geometry.py``), this builds the compact, velocity-grid
``flow_geometry.npz`` cache that the lightweight evaluator uses on the fly:

    flow_geometry.npz  (per patient, in <id>/flow_measurement/)

The cache stores the masked plane sample coordinates in this project's RAS
velocity grid plus per-time segmentation weights and the effective plane normal,
so ``PatientFlowGeometry.measure(velocity)`` consumes the model's velocity output
directly (no grid conversion).

Usage:
  python scripts/build_flow_geometry_cache.py                       # validation split
  python scripts/build_flow_geometry_cache.py --patients Alernscet Beborep
  python scripts/build_flow_geometry_cache.py --split validation --overwrite
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.flow_eval import (
    CACHE_FILENAME,
    autoflow_staging_dir,
    build_geometry_cache_for_patient,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _default_splits_csv() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = list((repo_root / "splits").glob("splits_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No splits_*.csv found in {repo_root / 'splits'}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _select_patients(args) -> list[str]:
    if args.patients:
        return list(args.patients)
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
    parser.add_argument("--overwrite", action="store_true", help="Rebuild caches that already exist")
    parser.add_argument("--seg-threshold", type=float, default=0.0,
                        help="Drop plane pixels whose seg never exceeds this (0.0 = exact)")
    args = parser.parse_args()

    pc = load_path_config(args.config)
    patient_ids = _select_patients(args)
    if not patient_ids:
        logger.warning("No patients selected; nothing to do.")
        return

    built, skipped, failures = 0, 0, []
    for pid in patient_ids:
        try:
            patient = Patient(path_config=pc, phonetic_id=pid, debug=False, config=args.config)
            staging = autoflow_staging_dir(patient)
            if (staging / CACHE_FILENAME).exists() and not args.overwrite:
                logger.info(f"[{pid}] cache exists; skipping")
                skipped += 1
                continue
            if not (staging / "aortic_spline.csv").exists():
                raise FileNotFoundError(f"geometry missing in {staging}; run the geometry chain first")
            out = build_geometry_cache_for_patient(patient, seg_threshold=args.seg_threshold)
            logger.info(f"[{pid}] wrote {out}")
            built += 1
        except Exception as exc:
            logger.error(f"[{pid}] failed: {exc}")
            failures.append(pid)

    logger.info(f"Done: {built} built, {skipped} skipped, {len(failures)} failed.")
    if failures:
        logger.warning(f"Failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
