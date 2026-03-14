#!/usr/bin/env python
"""
Extract ROI statistics (mean, std) from screen recordings of a medical image viewer.

Processes a video of a user scrolling through cardiac timepoints, detects stable
display periods via frame differencing, and extracts mean/std values from the
stats panel via OCR.

Segmentation strategy
---------------------
The video is analyzed by computing frame-to-frame mean absolute pixel differences
within the user-selected ROI (the stats panel). During scroll transitions the
displayed numbers change rapidly, producing large inter-frame differences. During
stable display of a timepoint the panel pixels are nearly identical between frames.
We threshold these differences (Otsu's method by default) to identify contiguous
"stable plateaus", select one representative frame from each plateau, and OCR only
those frames. If more plateaus are found than expected, near-duplicate OCR results
are merged; if fewer are found, the script warns.

Usage
-----
  # Full frame (no ROI selection — video already cropped to stats panel):
  python scripts/extract_roi_stats.py --video input.mov --out output.txt --full-frame

  # Interactive ROI selection on first frame:
  python scripts/extract_roi_stats.py --video input.mov --out output.txt

  # Hardcoded ROI (x, y, width, height in pixels):
  python scripts/extract_roi_stats.py --video input.mov --out output.txt --roi 100,200,300,50

  # Batch mode — process all .mov/.mp4 in a directory:
  python scripts/extract_roi_stats.py --video-dir recordings/ --full-frame

  # Full debug output with plots, cropped ROI images, and OCR log:
  python scripts/extract_roi_stats.py --video input.mov --out output.txt --debug-dir debug/

Dependencies
------------
  opencv-python, numpy, easyocr (or pytesseract + Tesseract), matplotlib (optional)
  See scripts/extract_roi_stats_requirements.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Measurement(NamedTuple):
    timepoint: int
    frame_index: int
    mean: float | None
    std: float | None
    raw_ocr: str
    confidence: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract mean/std statistics from a screen recording of a medical image viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", type=Path, default=None,
                   help="Input video file (.mov or .mp4)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output TSV file (one 'mean<TAB>std' per line)")
    p.add_argument("--csv", type=Path, default=None,
                   help="Optional CSV output with columns: timepoint, mean, std")
    p.add_argument("--roi", type=str, default="15,75,625,110",
                   help="ROI as x,y,w,h in pixels (default: 15,75,625,110 — Mean+Std lines)")
    p.add_argument("--full-frame", action="store_true",
                   help="Use the entire frame as the ROI (skip interactive selection)")
    p.add_argument("--sample-every", type=int, default=2,
                   help="Sample every Nth frame for stability analysis (default: 2)")
    p.add_argument("--min-plateau-frames", type=int, default=5,
                   help="Minimum consecutive stable sampled frames to form a plateau (default: 5)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Manual threshold for frame-diff stability detection. "
                        "Auto-detected via Otsu if omitted.")
    p.add_argument("--expected-timepoints", type=int, default=20,
                   help="Expected number of cardiac timepoints (default: 20)")
    p.add_argument("--ocr-engine", choices=["easyocr", "tesseract"], default="easyocr",
                   help="OCR backend (default: easyocr)")
    p.add_argument("--ocr-scale", type=int, default=3,
                   help="Upscale factor applied to ROI crop before OCR (default: 3)")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="Process all .mov/.mp4 files in this directory (batch mode)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for batch mode (one subfolder per video)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process videos even if output already exists")
    p.add_argument("--debug-dir", type=Path, default=None,
                   help="Directory for debug output (difference plot, ROI crops, OCR log)")
    p.add_argument("--interactive", action="store_true",
                   help="Review and optionally edit extracted values before saving")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug-level logging")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def open_video(path: Path) -> cv2.VideoCapture:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %s | %d frames | %.1f fps | %dx%d", path.name, n, fps, w, h)
    return cap


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def get_roi(
    frame: np.ndarray,
    roi_arg: str | None = None,
    full_frame: bool = False,
) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) from CLI string, full-frame flag, or interactive selection."""
    if full_frame:
        h, w = frame.shape[:2]
        logger.info("Using full frame as ROI: 0,0,%d,%d", w, h)
        return 0, 0, w, h

    if roi_arg is not None:
        parts = [int(v.strip()) for v in roi_arg.split(",")]
        if len(parts) != 4:
            raise ValueError(f"--roi must be x,y,w,h — got: {roi_arg}")
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            raise ValueError(f"ROI width/height must be positive — got w={w}, h={h}")
        logger.info("Using CLI ROI: x=%d y=%d w=%d h=%d", x, y, w, h)
        return x, y, w, h

    logger.info("Opening interactive ROI selector — draw a rectangle around the stats panel.")
    win = "Select Stats Panel ROI (ENTER to confirm, C to cancel)"
    roi = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, w, h = (int(v) for v in roi)
    if w == 0 or h == 0:
        raise ValueError("No ROI selected (zero area). Aborting.")
    logger.info("Selected ROI: x=%d y=%d w=%d h=%d", x, y, w, h)
    return x, y, w, h


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frame[y : y + h, x : x + w]


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def compute_frame_differences(
    cap: cv2.VideoCapture,
    roi: tuple[int, int, int, int],
    sample_every: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Compute per-frame mean absolute difference within *roi*.

    Returns
    -------
    frame_indices : int array of sampled frame numbers
    diffs         : float array of length ``len(frame_indices) - 1``
    roi_crops     : list of grayscale ROI crops (one per sampled frame)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices: list[int] = []
    roi_crops: list[np.ndarray] = []
    diffs: list[float] = []
    prev_gray: np.ndarray | None = None
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray = cv2.cvtColor(crop_roi(frame, roi), cv2.COLOR_BGR2GRAY)
            frame_indices.append(idx)
            roi_crops.append(gray.copy())
            if prev_gray is not None:
                diffs.append(float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))))
            prev_gray = gray
        idx += 1

    logger.info("Sampled %d / %d frames (every %d)", len(frame_indices), n_total, sample_every)
    return np.array(frame_indices), np.array(diffs, dtype=np.float64), roi_crops


def auto_threshold(diffs: np.ndarray) -> float:
    """Otsu's method on the distribution of ROI frame-differences.

    The differences are quantised to 256 bins, Otsu finds the threshold that
    minimises intra-class variance, then we map back to the original scale.
    """
    d_min, d_max = float(diffs.min()), float(diffs.max())
    if d_max - d_min < 1e-6:
        return d_max + 1.0

    normed = ((diffs - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(normed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(thresh_val) / 255.0 * (d_max - d_min) + d_min
    logger.info("Auto threshold (Otsu): %.4f  (diff range %.4f – %.4f)", threshold, d_min, d_max)
    return threshold


def find_stable_plateaus(
    diffs: np.ndarray,
    threshold: float,
    min_plateau_frames: int = 5,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs (inclusive) into the sampled-frames array.

    ``diffs[i]`` is the difference between sampled frame *i* and *i+1*.
    Frame *i+1* is marked stable when ``diffs[i] < threshold``.  Frame 0 is
    assumed stable (no prior diff).  Contiguous stable runs shorter than
    *min_plateau_frames* are discarded.
    """
    stable = np.concatenate([[True], diffs < threshold])
    plateaus: list[tuple[int, int]] = []
    start: int | None = None

    for i, s in enumerate(stable):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= min_plateau_frames:
                plateaus.append((start, i - 1))
            start = None
    if start is not None and len(stable) - start >= min_plateau_frames:
        plateaus.append((start, len(stable) - 1))

    logger.info("Found %d stable plateaus (threshold=%.4f, min_frames=%d)",
                len(plateaus), threshold, min_plateau_frames)
    for i, (s, e) in enumerate(plateaus):
        logger.debug("  Plateau %d: sampled frames [%d, %d] (%d frames)", i, s, e, e - s + 1)
    return plateaus


def select_representative_frames(plateaus: list[tuple[int, int]]) -> list[int]:
    """Pick one sampled-frame index per plateau (60 % into the run to dodge residual transitions)."""
    return [s + int(0.6 * (e - s)) for s, e in plateaus]


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def init_ocr(engine: str):
    if engine == "easyocr":
        import easyocr  # noqa: F811
        return easyocr.Reader(["en"], gpu=False, verbose=False)
    if engine == "tesseract":
        import pytesseract  # noqa: F811
        return pytesseract
    raise ValueError(f"Unknown OCR engine: {engine}")


def preprocess_for_ocr(crop_gray: np.ndarray, scale: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Return (upscaled_gray, binarised) versions of the crop for OCR."""
    h, w = crop_gray.shape
    up = cv2.resize(crop_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    binary = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 10)
    return up, binary


def ocr_roi_crop(
    crop_gray: np.ndarray,
    reader,
    engine: str = "easyocr",
    scale: int = 3,
) -> tuple[str, float]:
    """Run OCR and return ``(full_text, avg_confidence)``."""
    up, binary = preprocess_for_ocr(crop_gray, scale)

    if engine == "easyocr":
        results = reader.readtext(up, detail=1)
        if not results:
            results = reader.readtext(binary, detail=1)
        if not results:
            return "", 0.0
        texts = [t for _, t, _ in results]
        confs = [c for _, _, c in results]
        return " ".join(texts), float(np.mean(confs))

    if engine == "tesseract":
        cfg = "--psm 6"
        text = reader.image_to_string(up, config=cfg)
        return text.strip(), 0.0

    return "", 0.0


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

_CHAR_SUBS = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    ",": ".",
})

_LABEL_WORDS = {"mean", "avg", "average", "std", "sd", "dev", "deviation",
                "stdev", "signal", "intensity", "standard"}


def sanitize_ocr_text(text: str) -> str:
    """Fix common OCR substitution errors in numeric tokens."""
    tokens: list[str] = []
    for tok in text.split():
        if tok.lower().rstrip(":") in _LABEL_WORDS:
            tokens.append(tok)
            continue
        cleaned = tok.translate(_CHAR_SUBS)
        cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
        if cleaned:
            tokens.append(cleaned)
    return " ".join(tokens)


def extract_numbers(text: str) -> list[float]:
    """Pull all float-like tokens from *text*."""
    nums: list[float] = []
    for m in re.finditer(r"-?\d+\.?\d*", text):
        try:
            nums.append(float(m.group()))
        except ValueError:
            pass
    return nums


def _fix_broken_decimals(text: str) -> str:
    """Rejoin decimals split by OCR artifacts like ``86_ 11`` → ``86.11``."""
    return re.sub(r"(\d+)[_\s]+(\d{2})\b", r"\1.\2", text)


def parse_mean_std(raw_text: str) -> tuple[float | None, float | None]:
    """Extract a (mean, std) pair from OCR text.

    Strategy:
      1. Fix broken decimals (``86_ 11`` → ``86.11``).
      2. Look for labelled values (``Mean … <number>``, ``Std … <number>``).
      3. Fall back to taking the first two numbers from the sanitised text.
    """
    fixed_text = _fix_broken_decimals(raw_text)
    sanitized = sanitize_ocr_text(fixed_text)
    logger.debug("  Raw OCR : %r", raw_text)
    logger.debug("  Fixed   : %r", fixed_text)
    logger.debug("  Sanitized: %r", sanitized)

    mean_m = re.search(r"(?:mean|avg|average)[^0-9\-]*(-?\d+\.?\d*)", fixed_text, re.I)
    std_m = re.search(r"(?:std|sd|dev(?:iation)?|stdev)[^0-9\-]*(-?\d+\.?\d*)", fixed_text, re.I)

    mean_val = float(mean_m.group(1)) if mean_m else None
    std_val = float(std_m.group(1)) if std_m else None

    if mean_val is not None and std_val is not None:
        return mean_val, std_val

    numbers = extract_numbers(sanitized)
    if len(numbers) >= 2:
        if mean_val is not None:
            remaining = [n for n in numbers if n != mean_val]
            return mean_val, remaining[0] if remaining else numbers[-1]
        if std_val is not None:
            remaining = [n for n in numbers if n != std_val]
            return remaining[0] if remaining else numbers[0], std_val
        return numbers[0], numbers[1]

    if len(numbers) == 1:
        return (mean_val or numbers[0]), (std_val or None)

    logger.warning("  Could not parse numbers from: %r", raw_text)
    return None, None


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def merge_to_target(measurements: list[Measurement], target: int) -> list[Measurement]:
    """Iteratively merge the most-similar consecutive pair until we have *target* items."""
    merged = list(measurements)
    while len(merged) > target:
        best_idx, best_diff = 0, float("inf")
        for i in range(len(merged) - 1):
            a, b = merged[i], merged[i + 1]
            d = _pair_distance(a, b)
            if d < best_diff:
                best_diff = d
                best_idx = i
        keep = merged[best_idx] if merged[best_idx].confidence >= merged[best_idx + 1].confidence else merged[best_idx + 1]
        merged[best_idx] = keep
        del merged[best_idx + 1]

    return [m._replace(timepoint=i + 1) for i, m in enumerate(merged)]


def _pair_distance(a: Measurement, b: Measurement) -> float:
    if a.mean is None or b.mean is None:
        return 0.0  # merge unknowns first
    return abs(a.mean - b.mean) + abs((a.std or 0) - (b.std or 0))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(
    measurements: list[Measurement],
    out_path: Path,
    csv_path: Path | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for m in measurements:
            fh.write(f"{_fmt(m.mean)}\t{_fmt(m.std)}\n")
    logger.info("Saved %d rows → %s", len(measurements), out_path)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timepoint", "mean", "std"])
            for m in measurements:
                w.writerow([m.timepoint, _fmt(m.mean), _fmt(m.std)])
        logger.info("Saved CSV → %s", csv_path)


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def save_debug_output(
    frame_indices: np.ndarray,
    diffs: np.ndarray,
    threshold: float,
    plateaus: list[tuple[int, int]],
    rep_indices: list[int],
    roi_crops: list[np.ndarray],
    measurements: list[Measurement],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    # --- difference plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(frame_indices[1:], diffs, linewidth=0.5, alpha=0.8, label="ROI frame diff")
        ax.axhline(threshold, color="r", ls="--", alpha=0.7, label=f"Threshold ({threshold:.2f})")
        for s, e in plateaus:
            ax.axvspan(frame_indices[s], frame_indices[min(e, len(frame_indices) - 1)],
                       alpha=0.12, color="green")
        for ri in rep_indices:
            if ri < len(frame_indices):
                ax.axvline(frame_indices[ri], color="blue", alpha=0.4, lw=1)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Mean |Δpixel| in ROI")
        ax.set_title("Frame-to-frame ROI difference with detected plateaus")
        ax.legend()
        fig.tight_layout()
        fig.savefig(debug_dir / "frame_differences.png", dpi=150)
        plt.close(fig)
        logger.info("Debug plot → %s", debug_dir / "frame_differences.png")
    except ImportError:
        logger.warning("matplotlib not available — skipping debug plot")

    # --- cropped ROI images ---
    crops_dir = debug_dir / "roi_crops"
    crops_dir.mkdir(exist_ok=True)
    for i, ri in enumerate(rep_indices):
        if ri < len(roi_crops):
            fname = crops_dir / f"tp{i + 1:02d}_frame{frame_indices[ri]:06d}.png"
            cv2.imwrite(str(fname), roi_crops[ri])
    logger.info("Saved %d ROI crops → %s", len(rep_indices), crops_dir)

    # --- OCR log ---
    log_path = debug_dir / "ocr_log.json"
    records = [
        {
            "timepoint": m.timepoint,
            "frame_index": m.frame_index,
            "mean": m.mean,
            "std": m.std,
            "raw_ocr": m.raw_ocr,
            "confidence": round(m.confidence, 4),
        }
        for m in measurements
    ]
    with open(log_path, "w") as fh:
        json.dump(records, fh, indent=2)
    logger.info("OCR log → %s", log_path)


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def interactive_review(measurements: list[Measurement]) -> list[Measurement]:
    hdr = f"{'TP':>3}  {'Mean':>12}  {'Std':>12}  {'Conf':>6}  OCR (truncated)"
    print(f"\n{'=' * 64}\nINTERACTIVE REVIEW\n{'=' * 64}")
    print(hdr)
    print("-" * 64)
    for m in measurements:
        flag = " *" if m.mean is None or m.std is None or m.confidence < 0.3 else ""
        ocr_short = (m.raw_ocr[:30] + "…") if len(m.raw_ocr) > 30 else m.raw_ocr
        print(f"{m.timepoint:>3}  {_fmt(m.mean):>12}  {_fmt(m.std):>12}  {m.confidence:>6.2f}  {ocr_short}{flag}")

    print("\nType 'ok' to accept, or 'edit N mean std' to fix timepoint N.")
    edited = list(measurements)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAccepted as-is.")
            break
        if line.lower() in {"ok", "done", "accept", "q", "quit", ""}:
            break
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "edit":
            try:
                tp = int(parts[1])
                nm, ns = float(parts[2]), float(parts[3])
                idx = tp - 1
                if 0 <= idx < len(edited):
                    edited[idx] = edited[idx]._replace(mean=nm, std=ns)
                    print(f"  Updated timepoint {tp}: mean={nm}, std={ns}")
                else:
                    print(f"  Timepoint {tp} out of range")
            except ValueError:
                print("  Usage: edit <tp> <mean> <std>")
        else:
            print("  Commands: ok | edit N mean std")
    return edited


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_video(
    video_path: Path,
    out_path: Path,
    csv_path: Path | None,
    debug_dir: Path | None,
    reader,
    ocr_engine: str,
    ocr_scale: int,
    roi_arg: str | None,
    full_frame: bool,
    sample_every: int,
    min_plateau_frames: int,
    threshold_override: float | None,
    expected_timepoints: int,
    interactive: bool,
) -> list[Measurement]:
    """Process one video file end-to-end. *reader* is a pre-initialised OCR engine."""

    cap = open_video(video_path)
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Cannot read the first frame of %s — skipping.", video_path.name)
        cap.release()
        return []

    roi = get_roi(first_frame, roi_arg, full_frame=full_frame)

    logger.info("Computing frame-to-frame differences in ROI …")
    frame_indices, diffs, roi_crops = compute_frame_differences(cap, roi, sample_every)
    cap.release()

    if len(diffs) == 0:
        logger.error("No frame differences computed for %s — video may be too short.", video_path.name)
        return []

    threshold = threshold_override if threshold_override is not None else auto_threshold(diffs)

    plateaus = find_stable_plateaus(diffs, threshold, min_plateau_frames)
    if not plateaus:
        logger.error("No stable plateaus in %s. Try lowering --threshold (%.4f) "
                      "or --min-plateau-frames (%d).", video_path.name, threshold, min_plateau_frames)
        return []
    if len(plateaus) < expected_timepoints:
        logger.warning("[%s] Found %d plateaus but expected %d.",
                        video_path.name, len(plateaus), expected_timepoints)
    elif len(plateaus) > expected_timepoints:
        logger.info("[%s] Found %d plateaus (expected %d) — will merge after OCR.",
                     video_path.name, len(plateaus), expected_timepoints)

    rep_indices = select_representative_frames(plateaus)
    logger.info("Selected %d representative frames for OCR.", len(rep_indices))

    measurements: list[Measurement] = []
    for i, ri in enumerate(rep_indices):
        logger.info("OCR  %d/%d  (video frame %d) …", i + 1, len(rep_indices), frame_indices[ri])
        raw_text, conf = ocr_roi_crop(roi_crops[ri], reader, ocr_engine, ocr_scale)
        mean_val, std_val = parse_mean_std(raw_text)
        measurements.append(Measurement(
            timepoint=i + 1,
            frame_index=int(frame_indices[ri]),
            mean=mean_val,
            std=std_val,
            raw_ocr=raw_text,
            confidence=conf,
        ))
        logger.info("  → mean=%s  std=%s  conf=%.2f", _fmt(mean_val), _fmt(std_val), conf)

    if len(measurements) > expected_timepoints:
        measurements = merge_to_target(measurements, expected_timepoints)

    if len(measurements) != expected_timepoints:
        logger.warning("[%s] Final count: %d (expected %d)",
                        video_path.name, len(measurements), expected_timepoints)

    if interactive:
        measurements = interactive_review(measurements)

    save_outputs(measurements, out_path, csv_path)

    if debug_dir:
        save_debug_output(frame_indices, diffs, threshold, plateaus, rep_indices,
                          roi_crops, measurements, debug_dir)

    return measurements


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Collect video files ────────────────────────────────────────────────
    _VID_EXTS = {".mov", ".mp4"}

    if args.video_dir is not None:
        direct_videos = sorted(
            p for p in args.video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _VID_EXTS
        )
        sub_dirs = sorted(
            d for d in args.video_dir.iterdir()
            if d.is_dir() and any(f.suffix.lower() in _VID_EXTS for f in d.iterdir() if f.is_file())
        )

        if direct_videos:
            # Flat directory with videos (single patient)
            patient_groups = [(args.video_dir.name, direct_videos)]
        elif sub_dirs:
            # Parent directory with patient sub-folders
            patient_groups = [
                (d.name, sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in _VID_EXTS))
                for d in sub_dirs
            ]
            total = sum(len(vids) for _, vids in patient_groups)
            logger.info("Found %d patient folders with %d total videos in %s",
                        len(patient_groups), total, args.video_dir)
        else:
            logger.error("No .mov/.mp4 files found in %s (or its subdirectories)", args.video_dir)
            sys.exit(1)

        if args.out_dir is None:
            args.out_dir = args.video_dir / "roi_stats_output"

    elif args.video is not None:
        if args.out is None:
            logger.error("--out is required in single-video mode.")
            sys.exit(1)
        patient_groups = [("single", [args.video])]
    else:
        logger.error("Provide --video or --video-dir.")
        sys.exit(1)

    # ── Process each video ─────────────────────────────────────────────────
    all_results: dict[str, dict[str, list[Measurement]]] = {}
    video_counter = 0
    skipped = 0
    total_videos = sum(len(vids) for _, vids in patient_groups)
    reader = None  # lazy-init OCR only when needed

    for patient_name, video_files in patient_groups:
        all_results[patient_name] = {}

        for vpath in video_files:
            video_counter += 1
            label = f"{patient_name}/{vpath.name}" if len(patient_groups) > 1 else vpath.name

            if args.out_dir is not None:
                if len(patient_groups) > 1:
                    vid_out_dir = args.out_dir / patient_name / vpath.stem
                else:
                    vid_out_dir = args.out_dir / vpath.stem
                out_path = vid_out_dir / f"{vpath.stem}.txt"
                csv_path = vid_out_dir / f"{vpath.stem}.csv"
                debug_dir = vid_out_dir / "debug" if args.debug_dir is not None or args.out_dir is not None else None
            else:
                out_path = args.out
                csv_path = args.csv
                debug_dir = args.debug_dir

            if not args.overwrite and out_path.exists():
                logger.info("[%d/%d] %s — already processed, skipping (use --overwrite to redo)",
                            video_counter, total_videos, label)
                skipped += 1
                continue

            if args.out_dir is not None:
                vid_out_dir.mkdir(parents=True, exist_ok=True)

            if reader is None:
                logger.info("Initialising OCR engine: %s …", args.ocr_engine)
                reader = init_ocr(args.ocr_engine)

            sep = "─" * 60
            logger.info("%s\n  [%d/%d] %s\n%s", sep, video_counter, total_videos, label, sep)

            measurements = process_single_video(
                video_path=vpath,
                out_path=out_path,
                csv_path=csv_path,
                debug_dir=debug_dir,
                reader=reader,
                ocr_engine=args.ocr_engine,
                ocr_scale=args.ocr_scale,
                roi_arg=args.roi,
                full_frame=args.full_frame,
                sample_every=args.sample_every,
                min_plateau_frames=args.min_plateau_frames,
                threshold_override=args.threshold,
                expected_timepoints=args.expected_timepoints,
                interactive=args.interactive,
            )
            all_results[patient_name][vpath.name] = measurements

            print(f"\n{'=' * 44}")
            print(f"  [{video_counter}/{total_videos}] {label}")
            print(f"  Extracted {len(measurements)} timepoints → {out_path}")
            print(f"{'=' * 44}\n")

            if measurements:
                print("mean\tstd")
                for m in measurements:
                    print(f"{_fmt(m.mean)}\t{_fmt(m.std)}")
                print()

    # ── Batch summary ──────────────────────────────────────────────────────
    if total_videos > 1:
        processed = total_videos - skipped
        print(f"\n{'═' * 60}")
        print(f"  BATCH COMPLETE: {processed} processed, {skipped} skipped, {total_videos} total")
        for patient, vids in all_results.items():
            if len(patient_groups) > 1 and vids:
                print(f"  {patient}/")
            for name, ms in vids.items():
                status = f"{len(ms)} timepoints" if ms else "FAILED"
                prefix = "    " if len(patient_groups) > 1 else "  "
                print(f"{prefix}{name}: {status}")
        if args.out_dir:
            print(f"  Output: {args.out_dir}")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
