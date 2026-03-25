#!/usr/bin/env python
"""
Interactive crop tool using one reference video and one-or-more target videos.

Use case:
- Pick crop interactively from a single reference video.
- Apply that exact crop rule to any number of target videos.
- Output should be 1:1 (square), with bottom-right anchor preserved for logo stability.

Workflow:
1) Open first frame from the reference video.
2) Adjust only one parameter: TOP_CUT (pixels removed from top).
3) Script computes a square crop anchored to bottom-right and previews it.
4) Press Enter to export all target videos with the exact same crop rule.

Example:
  python scripts/crop_pair_square_gui.py \
      --reference input_reference.mp4 \
      --targets clip_a.mp4 clip_b.mp4 clip_c.mp4 \
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
            "Interactively choose TOP_CUT from a reference video, then export one "
            "or more target videos with identical bottom-right anchored square crops."
        )
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=False,
        help="Path to the reference video used only for interactive crop selection.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        nargs="+",
        required=False,
        help="One or more videos to crop using the reference crop settings.",
    )
    parser.add_argument(
        "--without-overlay",
        type=Path,
        required=False,
        help="Legacy: path to base video (without overlay).",
    )
    parser.add_argument(
        "--with-overlay",
        type=Path,
        required=False,
        help="Legacy: path to overlay video (with overlay).",
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
            "Open file-picker dialogs. If you multi-select videos at once, you can "
            "pick the reference from that set and all selected files are cropped. "
            "If you pick only one file first, it is treated as the reference and "
            "you will be asked for target videos next. "
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
    # ffmpeg filtergraphs treat commas as separators, so they must be escaped
    # inside expressions when passed as a single -vf argument.
    size_expr = f"min(iw\\,ih-{top_cut})"
    crop_expr = f"crop={size_expr}:{size_expr}:iw-{size_expr}:ih-{size_expr}"
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


def validate_inputs(paths: list[Path]) -> None:
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")


def choose_reference_from_candidates(candidates: list[Path]) -> Path:
    if len(candidates) == 1:
        return candidates[0]

    logger.info("Selected videos:")
    for idx, path in enumerate(candidates, start=1):
        logger.info("  [%d] %s", idx, path)
    logger.info("Choose which one is the REFERENCE video.")

    while True:
        raw = input(f"Reference index (1-{len(candidates)}): ").strip()
        if not raw:
            continue
        try:
            chosen = int(raw)
        except ValueError:
            logger.info("Please enter a number.")
            continue
        if 1 <= chosen <= len(candidates):
            return candidates[chosen - 1]
        logger.info("Index out of range.")


def select_reference_and_targets_with_dialog() -> tuple[Path, list[Path]]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "tkinter is unavailable. Provide --reference and --targets via CLI."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.update()

    filetypes = [
        ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"),
        ("All files", "*.*"),
    ]

    selected = filedialog.askopenfilenames(
        title="Select one or more videos (reference + outputs)",
        filetypes=filetypes,
    )
    if not selected:
        raise KeyboardInterrupt("No videos selected.")

    selected_paths = [Path(p) for p in selected]
    if len(selected_paths) > 1:
        root.destroy()
        reference = choose_reference_from_candidates(selected_paths)
        # When users select multiple files up front, crop all selected files.
        return reference, selected_paths

    reference = selected_paths[0]
    targets = filedialog.askopenfilenames(
        title="Select one or more TARGET videos to crop",
        filetypes=filetypes,
    )
    if not targets:
        raise KeyboardInterrupt("No target videos selected.")

    root.destroy()
    return reference, [Path(p) for p in targets]


def resolve_input_videos(args: argparse.Namespace) -> tuple[Path, list[Path]]:
    has_new = args.reference is not None or bool(args.targets)
    has_legacy = args.without_overlay is not None or args.with_overlay is not None

    if has_new and has_legacy:
        raise ValueError(
            "Use either --reference/--targets or legacy --without-overlay/--with-overlay, not both."
        )

    if args.use_file_dialog or (not has_new and not has_legacy):
        logger.info("Opening file picker dialogs for input videos...")
        return select_reference_and_targets_with_dialog()

    if has_new:
        if args.reference is None or not args.targets:
            raise ValueError("Provide both --reference and at least one --targets path.")
        return args.reference, list(args.targets)

    if args.without_overlay and args.with_overlay:
        logger.info(
            "Using legacy pair args. Prefer --reference + --targets for multi-video crop."
        )
        return args.without_overlay, [args.without_overlay, args.with_overlay]

    raise ValueError(
        "Missing inputs. Provide --reference and --targets (or use --use-file-dialog)."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        ensure_ffmpeg_tools_available()
        reference_video, target_videos = resolve_input_videos(args)
        validate_inputs([reference_video, *target_videos])
    except KeyboardInterrupt as exc:
        logger.info("%s", exc)
        return 130
    except Exception as exc:  # pragma: no cover - startup validation
        logger.error("%s", exc)
        return 1

    dims_reference = ffprobe_dimensions(reference_video)
    logger.info("reference dimensions: %dx%d", *dims_reference)
    for target_video in target_videos:
        dims_target = ffprobe_dimensions(target_video)
        logger.info("target dimensions (%s): %dx%d", target_video.name, *dims_target)
        if dims_target != dims_reference:
            logger.warning(
                "Target dimensions differ from reference for %s. "
                "TOP_CUT will still be applied; framing may not align perfectly.",
                target_video.name,
            )

    try:
        frame = read_first_frame(reference_video)
        top_cut = choose_top_cut_interactively(frame)
    except KeyboardInterrupt as exc:
        logger.info("%s", exc)
        return 130
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Selected TOP_CUT=%d", top_cut)
    for i, target_video in enumerate(target_videos, start=1):
        out_target = output_path(target_video, out_dir, args.suffix)
        logger.info("Exporting (%d/%d): %s", i, len(target_videos), out_target)
        run_ffmpeg_crop(target_video, out_target, top_cut, args.crf, args.preset)

    logger.info("Done.")
    logger.info("Cropped %d target video(s).", len(target_videos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
