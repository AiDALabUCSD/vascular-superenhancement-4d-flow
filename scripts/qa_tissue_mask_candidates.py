#!/usr/bin/env python3
"""Render side-by-side candidate tissue masks for one patient.

Use this to pick mask hyperparameters when the production mask is too
aggressive (e.g. bleeding into the lungs) or too conservative. Generates
one PNG per requested patient, one row per axial slab (top/mid/bottom),
one column per candidate parameter set. Each panel shows the magnitude
image with the candidate mask outlined in green.

The candidate set is hardcoded for now (current production + 4 tighter
variants); edit ``CANDIDATES`` below to try other combos.

Usage::

    python scripts/qa_tissue_mask_candidates.py --config all_patients \
        --patient-ids Achelney
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    gaussian_filter,
    label as cc_label,
    generate_binary_structure,
)

from vascular_superenhancement.utils.path_config import _PROJECT_ROOT, load_path_config
from vascular_superenhancement.data_management.patients import Patient


DOWNSAMPLED_FOLDER = "downsampled_full_fov_128x128x64"
QA_SUBFOLDER = "tissue_mask_candidates"


@dataclass
class Candidate:
    name: str  # short label for figure
    threshold: float
    sigma: float
    rethreshold: float | None  # None means: use morphological closing + LCC instead
    closing_radius: int = 0  # only used when rethreshold is None
    keep_largest_cc: bool = False  # post-filter step

    def short(self) -> str:
        if self.rethreshold is not None:
            return (
                f"{self.name}\n"
                f"thr={self.threshold:.2f}  σ={self.sigma:.1f}  rethr={self.rethreshold:.2f}"
                f"{'  +LCC' if self.keep_largest_cc else ''}"
            )
        return (
            f"{self.name}\n"
            f"thr={self.threshold:.2f}  close r={self.closing_radius}"
            f"{'  +LCC' if self.keep_largest_cc else ''}"
        )


CANDIDATES: list[Candidate] = [
    Candidate("PROD (new)", threshold=0.10, sigma=1.5, rethreshold=0.50),
    Candidate("LEGACY", threshold=0.05, sigma=3.0, rethreshold=0.333),
    Candidate("A: lower thr", threshold=0.05, sigma=1.5, rethreshold=0.50),
    Candidate("B: tighter sigma", threshold=0.10, sigma=1.0, rethreshold=0.50),
    Candidate("C: closing+LCC", threshold=0.10, sigma=0.0, rethreshold=None,
              closing_radius=2, keep_largest_cc=True),
    Candidate("D: tightest", threshold=0.15, sigma=1.0, rethreshold=0.60,
              keep_largest_cc=True),
]


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(3, 3)  # 26-connectivity
    labels, n = cc_label(mask > 0.5, structure=structure)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_label = counts.argmax()
    return (labels == keep_label).astype(np.float32)


def _make_mask(mag_mean: np.ndarray, c: Candidate) -> np.ndarray:
    """Compute one candidate mask from mean magnitude."""
    norm = float(np.percentile(mag_mean, 99.0))
    if norm <= 0:
        norm = 1.0
    mag_n = mag_mean / norm
    m = (mag_n > c.threshold).astype(np.float32)

    if c.rethreshold is not None:
        if c.sigma > 0:
            m = gaussian_filter(m, sigma=c.sigma)
        m = (m > c.rethreshold).astype(np.float32)
    else:
        struct = generate_binary_structure(3, 3)
        if c.closing_radius > 0:
            m = binary_closing(m > 0.5, structure=struct,
                               iterations=c.closing_radius).astype(np.float32)

    if c.keep_largest_cc:
        m = _largest_cc(m)
    return m


def _row_indices(z_dim: int) -> list[tuple[int, str]]:
    return [
        (int(z_dim * 0.30), "upper third (apex / lung)"),
        (z_dim // 2, "mid (heart)"),
        (int(z_dim * 0.70), "lower third (diaphragm / liver)"),
    ]


def render_one(
    pid: str,
    cfg,
    out_root: Path,
) -> None:
    patient = Patient(path_config=cfg, phonetic_id=pid)
    ds_root = patient.nifti_dir / DOWNSAMPLED_FOLDER
    if not ds_root.exists():
        logging.warning(f"  [{pid}] downsampled folder missing: {ds_root}")
        return

    # Mean across time at downsampled resolution (matches what training sees).
    mag_dir = ds_root / "4d_flow_mag"
    mag_files = sorted(mag_dir.glob(f"4d_flow_mag_{pid}_frame_*.nii.gz"))
    if not mag_files:
        logging.warning(f"  [{pid}] no mag frames in {mag_dir}")
        return
    mags = np.stack([nib.load(str(p)).get_fdata() for p in mag_files], axis=-1)
    mag_mean = mags.mean(axis=-1)
    z_dim = mag_mean.shape[2]

    # Compute every candidate up-front so we know voxel counts for the labels.
    masks = [(c, _make_mask(mag_mean, c)) for c in CANDIDATES]

    # Layout: rows = slabs, cols = candidates.
    n_rows = 3
    n_cols = len(CANDIDATES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        f"{pid} — tissue mask candidate sweep\n"
        f"(mean-across-time mag, downsampled FOV, axial slices)",
        fontsize=12, fontweight="bold",
    )

    slabs = _row_indices(z_dim)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # Top row gets candidate labels above each column.
    for j, (c, m) in enumerate(masks):
        voxel_pct = m.mean() * 100.0
        axes[0, j].set_title(
            c.short() + f"\nvoxels in mask: {voxel_pct:.1f}% of FOV",
            fontsize=8,
        )

    for i, (z, slab_label) in enumerate(slabs):
        axes[i, 0].set_ylabel(slab_label, fontsize=9)
        for j, (c, m) in enumerate(masks):
            ax = axes[i, j]
            ax.imshow(mag_mean[:, :, z].T, origin="upper", cmap="gray")
            ax.contour(
                m[:, :, z].T.astype(float), levels=[0.5],
                colors=["#00ff7f"], linewidths=0.8,
            )

    out_path = out_root / f"{pid}_tissue_mask_candidates.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logging.info(f"  [{pid}] -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="all_patients")
    p.add_argument("--patient-ids", nargs="+", required=True,
                   help="Patient IDs to render.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_path_config(args.config)

    out_root = Path(cfg.working_dir) / "qa" / QA_SUBFOLDER
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"QA output: {out_root}")

    for pid in args.patient_ids:
        render_one(pid, cfg, out_root)
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
