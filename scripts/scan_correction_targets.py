"""Scan the training correction targets (``4d_flow_diff_*``) for data-quality
problems: VENC overflow, isolated high-magnitude "speckle" voxels, large
corrections in air (outside the tissue mask), and NaN/inf.

The correction target the model is supervised on is the per-timepoint diff
(GT-corrected minus uncorrected velocity). Noisy voxel constellations in this
field directly poison training, so this scan quantifies how prevalent they are
across patients.

All magnitudes are reported in VENC units (correction / venc), so 1.0 == one
full VENC of correction at a voxel, which for a *correction* field is already
implausibly large.

Usage (main project env):
  python scripts/scan_correction_targets.py --split train --sample 30
  python scripts/scan_correction_targets.py --patients Achelney Adompel --frames all
  python scripts/scan_correction_targets.py --split train          # all train patients
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import median_filter

from vascular_superenhancement.utils.path_config import load_path_config
from vascular_superenhancement.data_management.patients import Patient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# A correction this large (in VENC units) is physically implausible for a phase
# error correction; treat as a candidate artifact.
SPECKLE_T = 1.0
# Isolated == voxel above SPECKLE_T but its 3x3x3 median is below this fraction
# of it (i.e. its neighbourhood is quiet -> it stands alone).
ISOLATION_FRAC = 0.5


def _load_frame(root: Path, pid: str, t: int) -> np.ndarray | None:
    comps = []
    for comp in ("vx", "vy", "vz"):
        p = root / f"4d_flow_diff_{comp}" / f"4d_flow_diff_{comp}_{pid}_frame_{t:02d}.nii.gz"
        if not p.exists():
            return None
        comps.append(np.asarray(nib.load(str(p)).dataobj, dtype=np.float32))
    return np.stack(comps, axis=-1)  # (X, Y, Z, 3)


def _scan_patient(patient: Patient, downsampled_folder: str, frames: str) -> dict | None:
    pid = patient.identifier
    root = patient.nifti_dir / downsampled_folder
    venc = float(patient.venc)
    n_tp = int(patient.num_timepoints)

    mask_path = root / f"correction_tissue_mask_{pid}.nii.gz"
    tissue = None
    if mask_path.exists():
        tissue = np.asarray(nib.load(str(mask_path)).dataobj) > 0.5

    if frames == "all":
        frame_idxs = list(range(n_tp))
    else:
        frame_idxs = [n_tp // 2]

    n_vox = 0
    n_nan = n_inf = 0
    n_overflow = 0          # |corr|/venc > 1 (vector norm)
    n_speckle = 0           # isolated high voxels
    n_air_large = 0         # |corr|/venc > 1 outside tissue
    n_air_vox = 0
    max_mag = 0.0
    seen = 0
    for t in frame_idxs:
        arr = _load_frame(root, pid, t)
        if arr is None:
            continue
        seen += 1
        n_nan += int(np.isnan(arr).sum())
        n_inf += int(np.isinf(arr).sum())
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        mag = np.linalg.norm(arr, axis=-1) / venc   # (X, Y, Z) in VENC units
        n_vox += mag.size
        max_mag = max(max_mag, float(mag.max()))
        high = mag > SPECKLE_T
        n_overflow += int(high.sum())
        if high.any():
            med = median_filter(mag, size=3)
            n_speckle += int((high & (med < ISOLATION_FRAC * mag)).sum())
        if tissue is not None:
            air = ~tissue
            n_air_vox += int(air.sum())
            n_air_large += int((high & air).sum())

    if seen == 0:
        return None
    return {
        "patient_id": pid,
        "venc": round(venc, 2),
        "frames_scanned": seen,
        "max_corr_venc": round(max_mag, 3),
        "n_overflow": n_overflow,
        "frac_overflow": n_overflow / max(n_vox, 1),
        "n_speckle": n_speckle,
        "frac_speckle": n_speckle / max(n_vox, 1),
        "n_air_large": n_air_large,
        "frac_air_large": (n_air_large / n_air_vox) if n_air_vox else float("nan"),
        "n_nan": n_nan,
        "n_inf": n_inf,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="all_patients")
    ap.add_argument("--splits", default="splits/splits_05-05-26.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--patients", nargs="+", help="Explicit ids (overrides --split)")
    ap.add_argument("--sample", type=int, default=0, help="Random subset size (0 = all)")
    ap.add_argument("--frames", choices=["center", "all"], default="center")
    ap.add_argument("--downsampled-folder", default="downsampled_full_fov_128x128x64")
    ap.add_argument("--out", default="working_dir/correction_target_scan.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pc = load_path_config(args.config)
    if args.patients:
        pids = list(args.patients)
    else:
        df = pd.read_csv(args.splits)
        pids = df.loc[df["split"] == args.split, "patient_id"].astype(str).tolist()
    if args.sample and args.sample < len(pids):
        rng = np.random.default_rng(args.seed)
        pids = sorted(rng.choice(pids, size=args.sample, replace=False).tolist())
    logger.info(f"Scanning {len(pids)} patients (frames={args.frames})")

    rows = []
    for i, pid in enumerate(pids):
        try:
            patient = Patient(path_config=pc, phonetic_id=pid, debug=False, config=args.config)
            res = _scan_patient(patient, args.downsampled_folder, args.frames)
        except Exception as exc:
            logger.warning(f"[{pid}] failed: {exc}")
            continue
        if res is None:
            logger.warning(f"[{pid}] no diff frames found; skipping")
            continue
        rows.append(res)
        logger.info(
            f"[{i+1}/{len(pids)}] {pid}: max={res['max_corr_venc']} "
            f"overflow={res['frac_overflow']:.2e} speckle={res['n_speckle']} "
            f"air_large={res['frac_air_large']:.2e} nan={res['n_nan']}"
        )

    if not rows:
        logger.error("No patients scanned.")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows -> {out}")

    # Quick cohort summary
    arr = pd.DataFrame(rows)
    logger.info("=== cohort summary ===")
    logger.info(f"max_corr_venc: median={arr.max_corr_venc.median():.2f} max={arr.max_corr_venc.max():.2f}")
    logger.info(f"total speckle voxels: {int(arr.n_speckle.sum())} across {len(arr)} patients")
    logger.info(f"patients with any NaN/inf: {int(((arr.n_nan+arr.n_inf)>0).sum())}")
    worst = arr.sort_values('n_speckle', ascending=False).head(5)
    logger.info("worst speckle patients:\n" + worst[['patient_id','max_corr_venc','n_speckle','frac_air_large']].to_string(index=False))


if __name__ == "__main__":
    main()
