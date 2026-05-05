#!/usr/bin/env python3
"""
CLI to free disk space by deleting NIfTI artifacts for post-2023 patients.

The "keep" set is the set of phonetic IDs in
``<repository_root>/full_pre-2023_cohort.csv`` (column ``Phonetic ID``).

For every patient directory under
``<working_dir>/all_patients/patient_data/<id>/`` whose ID is NOT in the keep
set, this tool removes the ``nifti/`` subdirectory. DICOM catalog CSVs are
left in place as a small record. Source DICOMs on the NFS mount are never
touched, so any deleted patient can be rebuilt later via ``build-patients``.

By default, runs in dry-run mode and only prints what would be deleted.
Pass ``--apply`` (and ``--yes`` to skip the interactive confirmation) to
actually delete.

Examples
--------
Dry-run with the all_patients config::

    prune-post2023-niftis --config all_patients

Actually delete after reviewing::

    prune-post2023-niftis --config all_patients --apply

Treat truly undated patients (in neither cohort source) as deletable too::

    prune-post2023-niftis --config all_patients --apply --include-undated

Remove the entire patient folder (catalogs included) instead of just nifti::

    prune-post2023-niftis --config all_patients --apply --whole-patient-dir
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ..utils.logger import setup_dataset_logger
from ..utils.path_config import load_path_config

PRE_2023_COHORT_FILENAME = "full_pre-2023_cohort.csv"
PHONETIC_ID_COLUMN = "Phonetic ID"


def load_keep_set(cohort_csv: Path) -> set[str]:
    """Read phonetic IDs from the pre-2023 cohort CSV."""
    if not cohort_csv.exists():
        raise FileNotFoundError(
            f"Pre-2023 cohort CSV not found at {cohort_csv}. "
            f"Expected column '{PHONETIC_ID_COLUMN}'."
        )
    df = pd.read_csv(cohort_csv, dtype=str)
    if PHONETIC_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{PHONETIC_ID_COLUMN}' not found in {cohort_csv}. "
            f"Found columns: {list(df.columns)}"
        )
    keep = {pid.strip() for pid in df[PHONETIC_ID_COLUMN].dropna() if pid.strip()}
    return keep


def load_main_db_ids(db_csv: Path) -> set[str]:
    """Read phonetic IDs from the main patients database (post-2023 cohort)."""
    if not db_csv.exists():
        return set()
    df = pd.read_csv(db_csv, dtype=str)
    ids: set[str] = set()
    for col in ("Phonetic ID_x", "Phonetic ID_y", "Phonetic ID"):
        if col in df.columns:
            ids.update(pid.strip() for pid in df[col].dropna() if pid.strip())
    return ids


def directory_size_bytes(path: Path) -> int:
    """Sum file sizes under ``path`` (follows no symlinks)."""
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return ans in {"y", "yes"}


def collect_targets(
    patient_data_dir: Path,
    keep_ids: set[str],
    main_db_ids: set[str],
    include_undated: bool,
    whole_patient_dir: bool,
    logger: logging.Logger,
) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """Return ``(targets, kept_ids, undated_skipped)``.

    ``targets`` is a list of ``(patient_id, path_to_delete)``.
    ``kept_ids`` are patients on disk that are in the keep set.
    ``undated_skipped`` are patients in neither source (kept by default).
    """
    on_disk = sorted(d.name for d in patient_data_dir.iterdir() if d.is_dir())
    targets: list[tuple[str, Path]] = []
    kept: list[str] = []
    undated_skipped: list[str] = []

    for pid in on_disk:
        patient_dir = patient_data_dir / pid
        if pid in keep_ids:
            kept.append(pid)
            continue

        in_main_db = pid in main_db_ids
        if not in_main_db and not include_undated:
            undated_skipped.append(pid)
            logger.warning(
                f"Skipping '{pid}': not in pre-2023 cohort and not in main DB. "
                f"Pass --include-undated to delete."
            )
            continue

        if whole_patient_dir:
            target = patient_dir
        else:
            target = patient_dir / "nifti"
            if not target.exists():
                logger.info(
                    f"Skipping '{pid}': no nifti/ subdir to delete "
                    f"(patient may already be cleaned)."
                )
                continue

        targets.append((pid, target))

    return targets, kept, undated_skipped


def write_manifest(
    manifest_path: Path,
    targets: list[tuple[str, Path]],
    sizes: dict[str, int],
    applied: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "target_path", "size_bytes", "size_human", "applied"])
        for pid, target in targets:
            sz = sizes.get(pid, 0)
            writer.writerow([pid, str(target), sz, human_bytes(sz).strip(), int(applied)])


def load_targets_from_manifest(
    manifest_path: Path,
    patient_data_dir: Path,
    logger: logging.Logger,
) -> list[tuple[str, Path]]:
    """Read targets from a previously written manifest CSV.

    Each row must have ``patient_id`` and ``target_path`` columns. Rows with
    ``applied`` set to a truthy value (1/true/yes) are skipped, so it's safe to
    re-feed a partially-applied manifest.

    Safety: every ``target_path`` must resolve to a directory under
    ``patient_data_dir``. Anything that doesn't is rejected with an error.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    pdd_resolved = patient_data_dir.resolve()
    targets: list[tuple[str, Path]] = []
    seen: set[str] = set()
    n_skipped_applied = 0
    n_skipped_missing = 0

    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "patient_id" not in (reader.fieldnames or []) or "target_path" not in (
            reader.fieldnames or []
        ):
            raise ValueError(
                f"Manifest {manifest_path} missing required columns "
                f"'patient_id' and/or 'target_path'. Got: {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader, start=2):  # start=2 for header offset
            pid = (row.get("patient_id") or "").strip()
            target_str = (row.get("target_path") or "").strip()
            applied_val = (row.get("applied") or "").strip().lower()
            if not pid or not target_str:
                continue
            if applied_val in {"1", "true", "yes", "y"}:
                n_skipped_applied += 1
                continue
            if pid in seen:
                logger.warning(
                    f"manifest line {row_idx}: duplicate patient_id '{pid}', skipping"
                )
                continue
            target = Path(target_str)
            try:
                target_resolved = target.resolve()
            except OSError as e:
                logger.error(
                    f"manifest line {row_idx}: cannot resolve '{target}': {e}; skipping"
                )
                continue
            try:
                target_resolved.relative_to(pdd_resolved)
            except ValueError:
                raise ValueError(
                    f"manifest line {row_idx}: target_path '{target}' is not under "
                    f"patient_data dir '{patient_data_dir}'. Refusing to proceed."
                )
            if not target.exists():
                logger.warning(
                    f"manifest line {row_idx}: target '{target}' does not exist; "
                    f"skipping (already deleted?)"
                )
                n_skipped_missing += 1
                continue
            if not target.is_dir():
                logger.error(
                    f"manifest line {row_idx}: target '{target}' is not a directory; "
                    f"skipping"
                )
                continue
            targets.append((pid, target))
            seen.add(pid)

    if n_skipped_applied:
        logger.info(
            f"manifest: skipped {n_skipped_applied} rows already marked applied=1"
        )
    if n_skipped_missing:
        logger.info(
            f"manifest: skipped {n_skipped_missing} rows whose target no longer exists"
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete NIfTI artifacts for post-2023 patients to free disk space.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="path_config name (without .yaml extension)",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help=f"Override path to pre-2023 cohort CSV "
             f"(default: <repository_root>/{PRE_2023_COHORT_FILENAME})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation when --apply is set.",
    )
    parser.add_argument(
        "--include-undated",
        action="store_true",
        help="Also delete patients that appear in neither the pre-2023 cohort "
             "CSV nor the main patients_database.csv (default: keep them).",
    )
    parser.add_argument(
        "--whole-patient-dir",
        action="store_true",
        help="Delete the entire patient folder (catalogs included) instead of "
             "just the nifti/ subdir.",
    )
    parser.add_argument(
        "--patient-ids",
        nargs="*",
        default=None,
        help="Optional explicit allowlist of patient IDs to consider as targets. "
             "Useful for testing on a small subset.",
    )
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Read targets directly from a previously written manifest CSV "
             "(columns: patient_id, target_path, ...). "
             "When set, the cohort/DB CSVs are NOT consulted -- the manifest is the "
             "source of truth, so you can hand-edit it (delete rows for patients you "
             "want to spare) and rerun. Rows with applied=1 are skipped. "
             "All target_path entries must be under <working_dir>/patient_data/.",
    )
    args = parser.parse_args()

    path_config = load_path_config(args.config)
    cohort_csv = args.cohort_csv or (path_config.repository_root / PRE_2023_COHORT_FILENAME)
    db_csv = path_config.database_path
    patient_data_dir = path_config.working_dir / "patient_data"

    if not patient_data_dir.exists():
        print(f"ERROR: patient_data dir not found: {patient_data_dir}", file=sys.stderr)
        return 2

    logger = setup_dataset_logger(
        "prune_post2023_niftis",
        level=logging.INFO,
        config=args.config,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = path_config.working_dir / "logs"
    manifest_path = log_dir / f"prune_post2023_niftis_{timestamp}.csv"

    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(f"=== prune-post2023-niftis [{mode}] ===")
    logger.info(f"config:           {args.config}")
    logger.info(f"patient_data dir: {patient_data_dir}")
    if args.from_manifest:
        logger.info(f"from manifest:    {args.from_manifest} "
                    f"(cohort/DB CSVs ignored)")
    else:
        logger.info(f"cohort CSV:       {cohort_csv}")
        logger.info(f"main DB CSV:      {db_csv}")
        logger.info(f"include_undated:  {args.include_undated}")
    logger.info(f"whole_patient:    {args.whole_patient_dir}")
    logger.info(f"manifest:         {manifest_path}")

    if args.from_manifest:
        targets = load_targets_from_manifest(
            args.from_manifest, patient_data_dir, logger
        )
        kept = []
        undated_skipped = []
    else:
        keep_ids = load_keep_set(cohort_csv)
        main_db_ids = load_main_db_ids(db_csv)
        logger.info(
            f"keep set size: {len(keep_ids)} (from pre-2023 cohort), "
            f"main DB size: {len(main_db_ids)}"
        )

        targets, kept, undated_skipped = collect_targets(
            patient_data_dir=patient_data_dir,
            keep_ids=keep_ids,
            main_db_ids=main_db_ids,
            include_undated=args.include_undated,
            whole_patient_dir=args.whole_patient_dir,
            logger=logger,
        )

    if args.patient_ids:
        allowed = set(args.patient_ids)
        before = len(targets)
        targets = [(p, t) for (p, t) in targets if p in allowed]
        logger.info(
            f"--patient-ids provided: filtered targets from {before} to {len(targets)}"
        )

    if args.from_manifest:
        logger.info(f"targets from manifest:   {len(targets)}")
    else:
        on_disk_total = len(kept) + len(targets) + len(undated_skipped)
        logger.info(f"on-disk patients:        {on_disk_total}")
        logger.info(f"  kept (pre-2023):       {len(kept)}")
        logger.info(f"  targeted for delete:   {len(targets)}")
        logger.info(f"  skipped (undated):     {len(undated_skipped)}")

    if not targets:
        logger.info("Nothing to do. Exiting.")
        return 0

    logger.info("Computing sizes for targets (this may take a moment)...")
    sizes: dict[str, int] = {}
    total_bytes = 0
    for pid, target in tqdm(targets, desc="Sizing", unit="patient"):
        sz = directory_size_bytes(target)
        sizes[pid] = sz
        total_bytes += sz

    target_by_pid = dict(targets)
    largest = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    logger.info(f"All {len(largest)} targets (sorted by size, descending):")
    for i, (pid, sz) in enumerate(largest, start=1):
        logger.info(
            f"  [{i:3d}/{len(largest)}] {pid:30s} {human_bytes(sz)}  "
            f"-> {target_by_pid[pid]}"
        )
    logger.info(f"TOTAL reclaimable: {human_bytes(total_bytes)}  "
                f"across {len(targets)} patients")

    write_manifest(manifest_path, targets, sizes, applied=False)
    logger.info(f"Wrote manifest of planned deletions to {manifest_path}")

    if not args.apply:
        logger.info("Dry-run complete. Re-run with --apply to actually delete.")
        return 0

    if not args.yes:
        prompt = (
            f"\nAbout to permanently delete {len(targets)} paths "
            f"({human_bytes(total_bytes).strip()}).\n"
            f"Type 'yes' to proceed: "
        )
        if not confirm(prompt):
            logger.info("Aborted by user.")
            return 1

    logger.info("Beginning deletion...")
    n_ok = 0
    n_fail = 0
    bytes_freed = 0
    t0 = time.time()
    pbar = tqdm(targets, desc="Deleting", unit="patient")
    for pid, target in pbar:
        try:
            shutil.rmtree(target)
            n_ok += 1
            bytes_freed += sizes.get(pid, 0)
            logger.info(f"deleted: {pid} -> {target} ({human_bytes(sizes.get(pid, 0)).strip()})")
        except Exception as e:
            n_fail += 1
            logger.error(f"FAILED to delete {target}: {e}")
        pbar.set_postfix(freed=human_bytes(bytes_freed).strip(), failed=n_fail)
    elapsed = time.time() - t0

    logger.info("=== summary ===")
    logger.info(f"deleted: {n_ok} / {len(targets)}")
    logger.info(f"failed:  {n_fail}")
    logger.info(f"freed:   {human_bytes(bytes_freed)}")
    logger.info(f"elapsed: {elapsed:.1f}s")

    write_manifest(manifest_path, targets, sizes, applied=True)
    logger.info(f"Updated manifest with applied=1 at {manifest_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
