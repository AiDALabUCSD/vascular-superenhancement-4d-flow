#!/usr/bin/env python3
"""Generate per-patient QA figures for the phase-error-correction (PEC) pipeline.

NOTE on what the polynomial reconstruction actually represents:
    The "ground-truth corrected" velocity (``4d_flow_*_corr``) is the output of a
    PACS-side correction in which a human manually segments static tissue and a
    *piecewise-linear* model is fit to those voxels. The polynomial reconstruction
    rendered in this figure is OUR 3rd-order global cubic fit on a magnitude-only
    tissue mask, which is a different basis on a different mask. The two will
    disagree most strongly in the heart/great vessels and that's expected, not a
    bug. The trainer is supervised on the *direct PACS diff*
    (``4d_flow_diff_*``), so the polynomial is purely a QA / visualization
    artifact and the per-crop ``captured%`` numbers below have no impact on
    training.

For each requested patient this script loads the downsampled training-time
volumes plus the precomputed polynomial outputs and writes an 11-row x 3-column
diagnostic PNG. The figure lets you see in one glance:

- The data the polynomial fit consumed (rows 1-2: 4D flow magnitude, cine,
  tissue mask overlay, fit-pixels overlay).
- The direct per-timepoint diffs and the polynomial reconstruction, both
  unmasked and air-masked (rows 3-6).
- Two "did it help?" maps per component on a shared color scale (rows 7-8):
  row 7 is the polynomial delta-delta (improvement of poly over no
  correction) and row 8 is the *ceiling* delta-delta (improvement of the
  perfect direct-diff correction over no correction). Comparing the two
  rows answers "what fraction of the available improvement did the
  polynomial actually capture?".
- The ground-truth corrected, polynomial-corrected, and uncorrected velocities
  side-by-side, tissue-masked (rows 9-11).

By default the figure is rendered on the mid-axial slice of one cardiac frame.

Output path:
``<working_dir>/qa/pec_polynomial_fits/<variant>/<split>_<pid>_<variant>.png``,
where ``<variant>`` is auto-derived from ``--downsampled-folder`` (e.g.
``crop-10``).

Typical invocation::

    python scripts/qa_phase_error_correction.py \
        --config all_patients \
        --splits-file splits_05-05-26.csv \
        --downsampled-folder downsampled_full_fov_128x128x64_crop-17.5 \
        --patient-ids Achelney

Without ``--patient-ids`` the script processes every patient whose split is in
{train, validation, test} (skipping sagittal/coronal/skip rows).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from vascular_superenhancement.data_management.patients import Patient
from vascular_superenhancement.utils.path_config import _PROJECT_ROOT, load_path_config


# Active splits whose patients should be QA'd by default. Sagittal/coronal/skip
# rows are excluded.
DEFAULT_QA_SPLITS = {"train", "validation", "test"}

# Cardiac frame used for the QA snapshot.
DEFAULT_FRAME = 4

# Output sub-folder under ``working_dir/qa/``.
QA_SUBFOLDER = "pec_polynomial_fits"

logger = logging.getLogger("qa_pec")


# ---------------------------------------------------------------------------
# Volume IO
# ---------------------------------------------------------------------------


def _load(path: Path) -> np.ndarray | None:
    """Load a NIfTI as float32 ndarray, or None if missing."""
    if not path.exists():
        return None
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _mid_axial_slice(vol: np.ndarray) -> np.ndarray:
    """Return the mid-axial (z = Z/2) slice of an ``(X, Y, Z)`` volume."""
    z = vol.shape[2] // 2
    return vol[:, :, z]


def _load_slice(path: Path) -> np.ndarray | None:
    vol = _load(path)
    return None if vol is None else _mid_axial_slice(vol)


# ---------------------------------------------------------------------------
# Per-patient data assembly
# ---------------------------------------------------------------------------


def _patient_data(
    pid: str,
    ds_root: Path,
    frame: int,
    venc: float,
) -> dict:
    """Assemble all 2-D slices and per-voxel quantities needed by the QA layout.

    Returns a dict keyed by name; values are 2-D ndarrays (or None when the
    underlying file is missing). The QA renderer downstream uses ``None`` as
    a "data not available" sentinel and substitutes a placeholder panel.
    """
    out: dict = {}

    # Row 1: structural context
    out["mag"] = _load_slice(
        ds_root / "4d_flow_mag" / f"4d_flow_mag_{pid}_frame_{frame:02d}.nii.gz"
    )
    out["cine"] = _load_slice(
        ds_root / "3d_cine" / f"3d_cine_{pid}_frame_{frame:02d}.nii.gz"
    )

    # Masks used by rows 2, 5, 6, 8-10
    out["tissue_mask"] = _load_slice(
        ds_root / f"correction_tissue_mask_{pid}.nii.gz"
    )
    out["fit_mask"] = _load_slice(
        ds_root / f"correction_fit_mask_{pid}.nii.gz"
    )

    # Per-component velocity components: uncorrected, externally-corrected,
    # direct diff (corrected - uncorrected), and polynomial correction.
    for comp in ("vx", "vy", "vz"):
        out[f"uncorr_{comp}"] = _load_slice(
            ds_root / f"4d_flow_{comp}" / f"4d_flow_{comp}_{pid}_frame_{frame:02d}.nii.gz"
        )
        out[f"corr_{comp}"] = _load_slice(
            ds_root / f"4d_flow_{comp}_corr" / f"4d_flow_{comp}_corr_{pid}_frame_{frame:02d}.nii.gz"
        )
        out[f"direct_diff_{comp}"] = _load_slice(
            ds_root / f"4d_flow_diff_{comp}" / f"4d_flow_diff_{comp}_{pid}_frame_{frame:02d}.nii.gz"
        )
        poly = _load_slice(
            ds_root / f"ground_truth_correction_{comp}_{pid}.nii.gz"
        )
        # Polynomial GT is stored VENC-normalized; de-normalize to cm/s so all
        # rows of the figure share a consistent physical unit.
        out[f"poly_{comp}"] = None if poly is None else poly * venc

    return out


# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------


def _placeholder(ax: plt.Axes, title: str, message: str = "missing") -> None:
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", color="white", fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_title(title, fontsize=9)


def _imshow(
    ax: plt.Axes,
    arr: np.ndarray,
    title: str,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
):
    if mask is not None:
        arr = np.ma.masked_where(~mask, arr)
    im = ax.imshow(arr.T, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    return im


def _percentile_symmetric(
    arrays: list[np.ndarray | None],
    pct: float = 99.0,
    mask: np.ndarray | None = None,
) -> float:
    """Return ``percentile(|arr|, pct)`` over the concatenation of non-None inputs.

    When ``mask`` is given, only voxels where ``mask`` is True contribute to
    the percentile. This is useful for color-scaling diff/improvement panels
    using the in-tissue dynamic range only.
    """
    stack = []
    for a in arrays:
        if a is None:
            continue
        if mask is not None:
            stack.append(a[mask].ravel())
        else:
            stack.append(a.ravel())
    if not stack:
        return 1.0
    v = float(np.percentile(np.abs(np.concatenate(stack)), pct))
    return v if v > 0 else 1.0


def _row_colorbar(fig: plt.Figure, axes_row: list[plt.Axes], im, label: str) -> None:
    """Attach a thin colorbar to the right of a row of panels."""
    cb = fig.colorbar(im, ax=axes_row, fraction=0.025, pad=0.015)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def make_qa_figure(
    pid: str,
    split: str,
    frame: int,
    ds_root: Path,
    venc: float,
) -> plt.Figure:
    """Render the 11-row x 3-column QA figure described above for ``pid``."""
    data = _patient_data(pid, ds_root, frame, venc)

    components = ("vx", "vy", "vz")

    # Tissue / fit masks (binary). Compose the optional masked-array overlay
    # selector once for reuse.
    tissue_bool = None if data["tissue_mask"] is None else data["tissue_mask"] > 0.5
    fit_bool = None if data["fit_mask"] is None else data["fit_mask"] > 0.5

    # Color limits
    vel_max = 300.0  # cm/s clip for velocity panels (rows 8-10)
    # Correction limits use joint percentile across direct diff + polynomial,
    # restricted to tissue voxels so air/edge extrapolation doesn't compress
    # the in-tissue dynamic range. Falls back to whole-FOV when no tissue mask.
    corr_arrays: list[np.ndarray | None] = []
    for c in components:
        corr_arrays.append(data[f"direct_diff_{c}"])
        corr_arrays.append(data[f"poly_{c}"])
    corr_max = _percentile_symmetric(corr_arrays, pct=99.0, mask=tissue_bool)

    # Colormaps
    bg_color = "0.20"
    gray = plt.cm.gray.copy()
    gray.set_bad(bg_color)
    jet = plt.cm.jet.copy()
    jet.set_bad(bg_color)
    rdbu = plt.cm.RdBu_r.copy()
    rdbu.set_bad(bg_color)

    # Figure
    n_rows = 11
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 3.4), constrained_layout=True)
    z = data["mag"].shape[1] // 2 if data["mag"] is not None else "?"
    fig.suptitle(
        f"{pid}  [{split}]  frame {frame:02d}  mid-axial slice  "
        f"VENC={venc:.0f} cm/s",
        fontsize=14, fontweight="bold",
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(bg_color)

    # ------------------ Row 1: structural context ------------------
    # vx slot: 4D-flow magnitude. vz slot: cine. vy slot intentionally empty.
    if data["mag"] is not None:
        _imshow(axes[0, 0], data["mag"], "4D-flow magnitude", gray)
    else:
        _placeholder(axes[0, 0], "4D-flow magnitude")
    axes[0, 1].set_title("")  # intentionally empty
    axes[0, 1].axis("off")
    if data["cine"] is not None:
        _imshow(axes[0, 2], data["cine"], "3D cine", gray)
    else:
        _placeholder(axes[0, 2], "3D cine")

    # ------------------ Row 2: mask overlays ------------------
    # vx slot: magnitude + tissue mask outline (1 = tissue).
    # vy slot: magnitude + fit-pixels mask outline (what the polynomial fit
    #          consumed, i.e. tissue ∩ from-edge crop).
    # vz slot: intentionally empty.
    for col, mask, title in (
        (0, tissue_bool, "mag + tissue mask"),
        (1, fit_bool, "mag + fit pixels (used in polyfit)"),
    ):
        ax = axes[1, col]
        if data["mag"] is None or mask is None:
            _placeholder(ax, title)
            continue
        ax.imshow(data["mag"].T, origin="upper", cmap=gray)
        ax.contour(mask.T.astype(float), levels=[0.5], colors=["#00ff7f"], linewidths=1.0)
        ax.set_title(title, fontsize=9)
    axes[1, 2].set_title("")
    axes[1, 2].axis("off")

    # ------------------ Row 3: direct diff (cm/s) ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"direct_diff_{c}"]
        if arr is None:
            _placeholder(axes[2, j], f"direct diff {c}")
        else:
            last_im = _imshow(
                axes[2, j], arr, f"direct diff {c} (corr − uncorr)",
                jet, -corr_max, corr_max,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[2, :].tolist(), last_im, "cm/s")

    # ------------------ Row 4: polynomial correction (cm/s) ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"poly_{c}"]
        if arr is None:
            _placeholder(axes[3, j], f"poly correction {c}")
        else:
            last_im = _imshow(
                axes[3, j], arr, f"poly correction {c}",
                jet, -corr_max, corr_max,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[3, :].tolist(), last_im, "cm/s")

    # ------------------ Row 5: direct diff x tissue mask ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"direct_diff_{c}"]
        if arr is None or tissue_bool is None:
            _placeholder(axes[4, j], f"direct diff {c} x tissue")
        else:
            last_im = _imshow(
                axes[4, j], arr, f"direct diff {c} x tissue",
                jet, -corr_max, corr_max, mask=tissue_bool,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[4, :].tolist(), last_im, "cm/s")

    # ------------------ Row 6: poly correction x tissue mask ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"poly_{c}"]
        if arr is None or tissue_bool is None:
            _placeholder(axes[5, j], f"poly {c} x tissue")
        else:
            last_im = _imshow(
                axes[5, j], arr, f"poly {c} x tissue",
                jet, -corr_max, corr_max, mask=tissue_bool,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[5, :].tolist(), last_im, "cm/s")

    # ------------------ Rows 7-8: delta-delta plots ------------------
    # Row 7: polynomial delta-delta = |gt-uncorr| - |gt-poly_corr|.
    # Row 8: ceiling delta-delta    = |gt-uncorr| - |gt-diff_corr|, where
    #        diff_corr = uncorr + direct_diff = gt. The second term is zero
    #        by construction, so this row reduces to |gt - uncorr| and
    #        represents the *maximum possible improvement* from any
    #        correction. Plotting on the same RdBu scale as row 7 makes
    #        "what fraction of the available improvement did the polynomial
    #        capture?" visually obvious.
    # Both rows are scaled jointly on in-tissue values so air/edge
    # extrapolation doesn't compress the in-tissue dynamic range.
    poly_dd_arrays: list[np.ndarray | None] = []
    ceil_dd_arrays: list[np.ndarray | None] = []
    for c in components:
        uncorr = data[f"uncorr_{c}"]
        corr = data[f"corr_{c}"]
        poly = data[f"poly_{c}"]
        if uncorr is None or corr is None or poly is None:
            poly_dd_arrays.append(None)
            ceil_dd_arrays.append(None)
            continue
        err_uncorr = np.abs(corr - uncorr)
        err_poly = np.abs(corr - (uncorr + poly))
        poly_dd_arrays.append(err_uncorr - err_poly)
        ceil_dd_arrays.append(err_uncorr)  # ceiling: err_diff_corr ≡ 0
    dd_max = _percentile_symmetric(
        poly_dd_arrays + ceil_dd_arrays, pct=99.0, mask=tissue_bool
    )

    # Row 7: polynomial delta-delta.
    summary_lines: list[str] = []
    last_im = None
    for j, (c, dd, ceil_dd) in enumerate(
        zip(components, poly_dd_arrays, ceil_dd_arrays)
    ):
        title = f"{c}: |gt − uncorr| − |gt − poly_corr|"
        if dd is None:
            _placeholder(axes[6, j], title)
            summary_lines.append(f"{c}: n/a")
            continue
        last_im = _imshow(
            axes[6, j], dd, title,
            rdbu, -dd_max, dd_max,
        )
        if tissue_bool is not None:
            dd_t = dd[tissue_bool]
            ceil_t = ceil_dd[tissue_bool]
            mean_dd = float(dd_t.mean())
            mean_ceil = float(ceil_t.mean())
            pct_helped = float((dd_t > 0).mean()) * 100.0
            # Fraction of available improvement captured by the polynomial.
            # Negative if the polynomial on average makes things worse.
            captured = (mean_dd / mean_ceil * 100.0) if mean_ceil > 0 else float("nan")
            summary_lines.append(
                f"{c}: Δ|err|={mean_dd:+.2f} cm/s of {mean_ceil:.2f} ceiling "
                f"({captured:+.0f}% captured, {pct_helped:.0f}% tissue helped)"
            )
        else:
            summary_lines.append(f"{c}: (no tissue mask)")
    if last_im is not None:
        _row_colorbar(
            fig, axes[6, :].tolist(), last_im,
            "cm/s  (red = poly helped, blue = poly hurt)",
        )

    # Row 8: ceiling delta-delta (perfect direct-diff correction).
    last_im = None
    for j, (c, ceil_dd) in enumerate(zip(components, ceil_dd_arrays)):
        title = f"{c}: |gt − uncorr| − |gt − diff_corr|  (ceiling)"
        if ceil_dd is None:
            _placeholder(axes[7, j], title)
            continue
        last_im = _imshow(
            axes[7, j], ceil_dd, title,
            rdbu, -dd_max, dd_max,
        )
    if last_im is not None:
        _row_colorbar(
            fig, axes[7, :].tolist(), last_im,
            "cm/s  (max improvement available; diff_corr ≡ gt)",
        )

    # ------------------ Row 9: gt corrected x tissue ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"corr_{c}"]
        if arr is None or tissue_bool is None:
            _placeholder(axes[8, j], f"gt corrected {c} x tissue")
        else:
            last_im = _imshow(
                axes[8, j], arr, f"gt corrected {c} x tissue",
                rdbu, -vel_max, vel_max, mask=tissue_bool,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[8, :].tolist(), last_im, "cm/s")

    # ------------------ Row 10: poly corrected x tissue ------------------
    last_im = None
    for j, c in enumerate(components):
        uncorr = data[f"uncorr_{c}"]
        poly = data[f"poly_{c}"]
        if uncorr is None or poly is None or tissue_bool is None:
            _placeholder(axes[9, j], f"poly corrected {c} x tissue")
        else:
            poly_corr = uncorr + poly
            last_im = _imshow(
                axes[9, j], poly_corr, f"poly corrected {c} x tissue",
                rdbu, -vel_max, vel_max, mask=tissue_bool,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[9, :].tolist(), last_im, "cm/s")

    # ------------------ Row 11: uncorrected x tissue ------------------
    last_im = None
    for j, c in enumerate(components):
        arr = data[f"uncorr_{c}"]
        if arr is None or tissue_bool is None:
            _placeholder(axes[10, j], f"uncorrected {c} x tissue")
        else:
            last_im = _imshow(
                axes[10, j], arr, f"uncorrected {c} x tissue",
                rdbu, -vel_max, vel_max, mask=tissue_bool,
            )
    if last_im is not None:
        _row_colorbar(fig, axes[10, :].tolist(), last_im, "cm/s")

    # Numerical summary box (footer). "captured" = mean(poly Δ|err|) /
    # mean(ceiling Δ|err|), i.e. fraction of the available improvement the
    # polynomial captured in-tissue.
    if summary_lines:
        fig.text(
            0.01, 0.005,
            "Per-component summary (in tissue):\n  " + "\n  ".join(summary_lines),
            fontsize=8, family="monospace", color="0.15",
        )

    return fig


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------


def _resolve_patient_list(
    splits_csv: Path,
    explicit_ids: list[str] | None,
    active_splits: set[str],
) -> list[tuple[str, str]]:
    """Return ordered list of (pid, split) tuples to process."""
    df = pd.read_csv(splits_csv)
    if explicit_ids:
        keep = df[df.patient_id.isin(explicit_ids)]
        # Preserve user-supplied order.
        order = {pid: i for i, pid in enumerate(explicit_ids)}
        keep = keep.sort_values(by="patient_id", key=lambda s: s.map(order))
    else:
        keep = df[df.split.isin(active_splits)]
    return list(zip(keep.patient_id.tolist(), keep.split.tolist()))


def _auto_variant_from_folder(downsampled_folder: str) -> str:
    """Derive a short subfolder tag (e.g. ``crop-10``) from the downsampled folder.

    For ``downsampled_full_fov_128x128x64_crop-10`` -> ``crop-10``.
    For ``downsampled_full_fov_128x128x64`` (no crop) -> ``default``.
    Unknown patterns fall back to the literal folder name.
    """
    m = re.search(r"_(crop-[0-9.]+)$", downsampled_folder)
    if m:
        return m.group(1)
    if downsampled_folder.startswith("downsampled_full_fov_"):
        return "default"
    return downsampled_folder


def _make_one(
    pid: str,
    split: str,
    cfg,
    downsampled_folder: str,
    frame: int,
    out_root: Path,
    variant: str,
    overwrite: bool,
) -> None:
    out_path = out_root / f"{split}_{pid}_{variant}.png"
    if out_path.exists() and not overwrite:
        logger.info(f"  [{pid}] skipping (exists; pass --overwrite to redo)")
        return

    try:
        patient = Patient(path_config=cfg, phonetic_id=pid)
    except Exception as exc:
        logger.warning(f"  [{pid}] Patient load failed: {exc}")
        return

    ds_root = patient.nifti_dir / downsampled_folder
    if not ds_root.exists():
        logger.warning(f"  [{pid}] downsampled folder missing: {ds_root}")
        return

    fig = make_qa_figure(
        pid=pid,
        split=split,
        frame=frame,
        ds_root=ds_root,
        venc=float(patient.venc),
    )
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    logger.info(f"  [{pid}] -> {out_path.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="all_patients",
                   help="Path-config name (default: all_patients)")
    p.add_argument("--splits-file", default="splits_05-05-26.csv",
                   help="Splits CSV under splits/ (default: splits_05-05-26.csv)")
    p.add_argument("--downsampled-folder", default="downsampled_full_fov_128x128x64_crop-17.5",
                   help="Subfolder under each patient's nifti_dir")
    p.add_argument("--frame", type=int, default=DEFAULT_FRAME,
                   help=f"Cardiac frame to render (default: {DEFAULT_FRAME})")
    p.add_argument("--patient-ids", nargs="+", default=None,
                   help="Optional explicit patient IDs (default: all active-split patients)")
    p.add_argument("--active-splits", nargs="+", default=sorted(DEFAULT_QA_SPLITS),
                   help="Split values to include when --patient-ids is not given")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-render QA images that already exist on disk")
    p.add_argument("--variant", default=None,
                   help="Subfolder name under qa/pec_polynomial_fits/ for "
                        "this run's outputs. Defaults to auto-derived from "
                        "--downsampled-folder (e.g. 'crop-10').")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_path_config(args.config)
    # Splits live alongside the codebase, not on the NAS.
    splits_csv = _PROJECT_ROOT / "splits" / args.splits_file
    if not splits_csv.exists():
        logger.error(f"Splits file not found: {splits_csv}")
        return 1

    variant = args.variant or _auto_variant_from_folder(args.downsampled_folder)
    out_root = Path(cfg.working_dir) / "qa" / QA_SUBFOLDER / variant
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"QA output: {out_root}")
    logger.info(f"Variant subfolder: {variant}")
    logger.info(f"Downsampled folder: {args.downsampled_folder}")
    logger.info(f"Frame: {args.frame}")

    todo = _resolve_patient_list(
        splits_csv=splits_csv,
        explicit_ids=args.patient_ids,
        active_splits=set(args.active_splits),
    )
    logger.info(f"Patients to QA: {len(todo)}")

    for pid, split in todo:
        _make_one(
            pid=pid,
            split=split,
            cfg=cfg,
            downsampled_folder=args.downsampled_folder,
            frame=args.frame,
            out_root=out_root,
            variant=variant,
            overwrite=args.overwrite,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
