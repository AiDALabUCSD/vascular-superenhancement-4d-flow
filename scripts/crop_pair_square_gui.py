#!/usr/bin/env python
"""
Interactive crop tool for paired videos (with/without overlay).

Use case:
- Two videos show the same content, but one includes a color overlay.
- You want both exports cropped identically for slide-to-slide reveal in PowerPoint.
- Output should be 1:1 (square), with bottom-right anchor preserved for logo stability.

Workflow:
1) Open first frame from the reference video.
2) Adjust only one parameter: TOP_CUT (pixels removed from top).
3) Script computes a square crop anchored to bottom-right and previews it.
4) Press Enter to export both videos with the exact same crop rule.

Example:
  python scripts/crop_pair_square_gui.py \
      --without-overlay input_without_overlay.mp4 \
      --with-overlay input_with_overlay.mp4 \
      --out-dir cropped/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively choose TOP_CUT for a pair of videos, then export both "
            "with identical bottom-right anchored square crops."
        )
    )
    parser.add_argument(
        "--without-overlay",
        type=Path,
        required=False,
        help="Path to the base video (without overlay).",
    )
    parser.add_argument(
        "--with-overlay",
        type=Path,
        required=False,
        help="Path to the overlay video (with overlay).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for cropped files (default: current directory).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_square",
        help="Suffix for output filenames before extension (default: _square).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for H.264 encoding (default: 18). Lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        help="x264 preset (default: medium).",
    )
    parser.add_argument(
        "--use-file-dialog",
        action="store_true",
        help=(
            "Open file-picker dialogs to choose both videos. "
            "If no input paths are provided, dialogs are used automatically."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def ensure_ffmpeg_tools_available() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run(
            [tool, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool} not found. Install ffmpeg first (includes ffprobe)."
            )


def ffprobe_dimensions(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video_path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    width, height = map(int, out.split("x"))
    return width, height


def read_first_frame(video_path: Path) -> cv2.typing.MatLike:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")
    return frame


def output_path(in_path: Path, out_dir: Path, suffix: str) -> Path:
    return out_dir / f"{in_path.stem}{suffix}{in_path.suffix}"


def run_ffmpeg_crop(
    input_path: Path,
    output_file: Path,
    top_cut: int,
    crf: int,
    preset: str,
) -> None:
    # Square crop with bottom-right anchor after removing TOP_CUT from top.
    crop_expr = (
        "crop="
        f"'min(iw,ih-{top_cut})':'min(iw,ih-{top_cut})':"
        "'iw-min(iw,ih-{top_cut})':'ih-min(iw,ih-{top_cut})'"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        crop_expr,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_file),
    ]
    logger.debug("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def choose_top_cut_interactively(frame: cv2.typing.MatLike) -> int:
    height, width = frame.shape[:2]
    max_top = max(0, height - 2)

    window_name = "TOP_CUT Preview (Enter=Export, Esc=Cancel)"
    trackbar_name = "TOP_CUT"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar_name, window_name, 0, max_top, lambda _v: None)

    chosen_top = 0
    while True:
        top_cut = cv2.getTrackbarPos(trackbar_name, window_name)
        # Maintain 1:1 crop anchored to bottom-right.
        available_height = height - top_cut
        square_size = min(width, available_height)
        x = width - square_size
        y = height - square_size

        preview = frame.copy()
        cv2.rectangle(
            preview,
            (x, y),
            (x + square_size, y + square_size),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            preview,
            f"TOP_CUT={top_cut} | crop={square_size}:{square_size}:{x}:{y}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):  # Enter
            chosen_top = top_cut
            break
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Crop selection cancelled by user.")

    cv2.destroyAllWindows()
    return chosen_top


def validate_inputs(without_overlay: Path, with_overlay: Path) -> None:
    for p in (without_overlay, with_overlay):
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")


def select_file_pair_with_dialog() -> tuple[Path, Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "tkinter is unavailable. Provide --without-overlay and --with-overlay via CLI."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.update()

    filetypes = [
        ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"),
        ("All files", "*.*"),
    ]

    without = filedialog.askopenfilename(
        title="Select video WITHOUT overlay",
        filetypes=filetypes,
    )
    if not without:
        raise KeyboardInterrupt("No file selected for without-overlay video.")

    with_overlay = filedialog.askopenfilename(
        title="Select video WITH overlay",
        filetypes=filetypes,
    )
    if not with_overlay:
        raise KeyboardInterrupt("No file selected for with-overlay video.")

    root.destroy()
    return Path(without), Path(with_overlay)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        ensure_ffmpeg_tools_available()
        if args.use_file_dialog or not (args.without_overlay and args.with_overlay):
            logger.info("Opening file picker dialogs for input videos...")
            args.without_overlay, args.with_overlay = select_file_pair_with_dialog()
        validate_inputs(args.without_overlay, args.with_overlay)
    except KeyboardInterrupt as exc:
        logger.info("%s", exc)
        return 130
    except Exception as exc:  # pragma: no cover - startup validation
        logger.error("%s", exc)
        return 1

    dims_without = ffprobe_dimensions(args.without_overlay)
    dims_with = ffprobe_dimensions(args.with_overlay)
    logger.info("without-overlay dimensions: %dx%d", *dims_without)
    logger.info("with-overlay dimensions:    %dx%d", *dims_with)
    if dims_without != dims_with:
        logger.warning(
            "Input dimensions differ. TOP_CUT will still be applied to both; "
            "crop results may not align perfectly."
        )

    try:
        frame = read_first_frame(args.without_overlay)
        top_cut = choose_top_cut_interactively(frame)
    except KeyboardInterrupt as exc:
        logger.info("%s", exc)
        return 130
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_without = output_path(args.without_overlay, out_dir, args.suffix)
    out_with = output_path(args.with_overlay, out_dir, args.suffix)

    logger.info("Selected TOP_CUT=%d", top_cut)
    logger.info("Exporting: %s", out_without)
    run_ffmpeg_crop(args.without_overlay, out_without, top_cut, args.crf, args.preset)
    logger.info("Exporting: %s", out_with)
    run_ffmpeg_crop(args.with_overlay, out_with, top_cut, args.crf, args.preset)

    logger.info("Done.")
    logger.info("Use these two outputs on adjacent slides for clean reveal transitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
