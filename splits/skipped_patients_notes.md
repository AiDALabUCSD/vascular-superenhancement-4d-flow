# Skipped Patients — Diagnostic Notes

Maintained alongside `splits/splits_05-05-26.csv`. Each entry documents WHY a patient is set to `skip` and what would be needed to recover them. Use this as a debug worklist after re-`rsync`ing raw data from the source / re-running upstream pipelines.

When a patient is moved out of `skip`, **delete its entry here** (or move to a "Recovered" section).

---

## Category A: GE scanner variants with non-standard private-tag encodings

These patients are GE but their flow-encoding private tag (`(0x0043, 0x1030)`) takes values outside our recognized 4D-flow range `{2, 3, 4, 5}`. The current `dicom_catalog_4d_flow` tag filter therefore drops all of their 4D-flow rows, yielding "No 4D Flow files found".

Marked: 2026-04-29 / 04-30.

| Patient | Scanner / model | Tag values seen | Notes & recovery path |
|---|---|---|---|
| **Ebimqued** | GE Signa HDxt | `(0x0043, 0x1030)` ∈ `{8, 9, 10, 11}` | Older GE firmware. Need to extend the 4D-flow tag filter to recognize 8–11 as `{vx, vy, vz, magV}` for HDxt. **Verify mapping against a known-good HDxt dataset before lifting the skip.** |
| **Pibefu** | GE Signa HDxt | Same as Ebimqued | Same fix path. |
| **Usuegtug** | GE Discovery MR750 | `(0x0043, 0x1030)` ∈ `{0, 6, 50}` | Tag values do not look like 4D-flow at all (no obvious vx/vy/vz/mag pattern). Likely the patient genuinely lacks 4D-flow data on this study. **Verify by inspecting series descriptions and checking if a different study has 4D-flow.** |

---

## Category B: External ECC-correction pipeline produced malformed corrected-velocity data

Upstream `corrected_velocities/<pid>.npy` exists but the resulting corrected per-timepoint NIfTIs are wrong shape / partially garbage. The build proceeds through Phases 1–2 and then dies in Phase 3 or downstream.

Marked: 2026-05-04 / 05-05.

| Patient | Symptom | Recovery path |
|---|---|---|
| **Bipayu** | Corrected NIfTI is `(X, Y, Z, 1)` (only 1 timepoint instead of 20) | Re-run external ECC correction; suspect the upstream input was already 3D (single timepoint). |
| **Bipolscol** | Same as Bipayu — corrected NIfTI shape `(X, Y, Z, 1)` | Same. |
| **Hetupo** | Corrected NIfTI is correctly 4D (20 timepoints), but `diff_vz` per-timepoint dir contains only 11/20 valid files (frames 11–19 missing/garbage) | Re-run external ECC correction; investigate whether intermediate per-timepoint storage corrupted frames 11–19 specifically. |

---

## Category C: DualVenc-only patients (no single-venc 4D-flow series)

GE GenIQ research dual-venc protocol patients. The pipeline assumes a single-venc reconstruction (one mag + one vx/vy/vz per cardiac phase); dual-venc series store two complete velocity sets per direction (HighVenc + LowVenc) plus an Anatomy and Preview series. The new `DualVenc` filter (database-driven, in `patients.dicom_catalog_4d_flow`) drops these series; for these 18 patients there is no single-venc fallback in the same study, so the catalog ends up empty.

Marked: 2026-05-11.

Identification rule: any series whose `Series Descriptions` entry in `patients_database.csv` contains "DualVenc" (case-insensitive) is dropped.

| # | Patient |
|---|---|
| 1 | Alkubol |
| 2 | Ceruresk |
| 3 | Cradinif |
| 4 | Criheno |
| 5 | Dihoori |
| 6 | Dosebom |
| 7 | Gesoqui |
| 8 | Gojaje |
| 9 | Goulaja |
| 10 | Gusralor |
| 11 | Hekislu |
| 12 | Ifquanig |
| 13 | Lenikey |
| 14 | Megugu |
| 15 | Musloke |
| 16 | Pugelu |
| 17 | Sekepug |
| 18 | Sujapib |

**Recovery path:** would require implementing a dual-venc reconstruction (combine HighVenc + LowVenc into a single phase-unwrapped velocity volume) — non-trivial, separate research effort. For now, treat as out-of-scope.

---

## Category D: Single-venc series listed in database but not transferred from PACS

| Patient | Symptom | Recovery path |
|---|---|---|
| **Strejekast** | `patients_database.csv` lists `Ax 4D FLOW v350 Heart` (series 4130) as present, but unzipped DICOM dir only contains series 8, 9, 2400–2406 (the 7 dual-venc series). Series 4130 is missing on disk. | Re-pull series 4130 from PACS / source archive. If recovered, this patient is salvageable via the existing single-venc pipeline. **Worth re-checking after rsync.** |

Marked: 2026-05-11.

---

## Category E: Incomplete 4D-flow components on disk

Patients whose 4D-flow catalog ends up missing one or more required components (mag, vx, vy, vz) or has incomplete cardiac-phase coverage on at least one component.

Marked: 2026-05-11.

| Patient | Disk inventory (post-tag-filter) | Diagnosis | Recovery path |
|---|---|---|---|
| **Quimafbray** | 1 series (sn=2100): vx-only, 20 phases, 120 slices. **No vy, vz, or mag series at all.** | Upstream PACS export only included the vx-tagged series. | Re-pull the matching vy/vz/mag-tagged series from PACS using the same study/accession. **Worth re-checking after rsync.** |
| **Lestrafi** | 1 series (sn=4130): vx-only, **1 cphase**, 176 files. | Series is a single-volume Preview/Anatomy, not a real 4D-flow acquisition. The actual 4D-flow series (if any) was not transferred. | Re-pull the actual 4D-flow series from PACS. Verify the study has one. |
| **Quebike** | 4 series (sn=4120–4123): vx, vy, vz, magV all present. **vy is missing 2 cardiac phases** (2064 files vs expected 2400). | Same shape as Bipayu/Bipolscol/Hetupo: incomplete cardiac coverage on one component. | Either re-pull the missing vy phases from PACS, or implement a zero-pad / temporal-interpolation workaround for missing cphases (pipeline-side). |
| **Flirego** | After DualVenc filter: only 1 series remains (sn=4130, vx, **1 cphase**, 160 files). Pre-filter had dual-venc series 2300–2305 (real 4D-flow data) plus 4130 (Preview-only). | The single-venc series listed in the database (`Ax Stanford 4D FLOW v250 Chest`) is actually a Preview/Anatomy on disk, not real 4D-flow. The actual data was acquired as dual-venc. | Effectively dual-venc-only — same recovery path as Category C. |

---

## Category G: Sagittal 4D-flow acquisitions (parked until reorientation pipeline is finalized)

These patients were acquired with the slice direction along the LR axis (sagittal slabs). They are NOT marked `skip` — instead they're labeled `sagittal_train` / `sagittal_validation` so they remain visible in the splits file and can be reincorporated without re-tagging when the sagittal integration is fully validated. The current dataset / dataloader pipeline only consumes rows whose `split` ∈ {`train`, `validation`, `test`}, so these `sagittal_*` labels are silently excluded from the active baseline training run.

Reasons for parking:

- The sagittal reorientation path is implemented end-to-end (`build_downsampled_full_fov_per_timepoint(reorient_non_axial=True)`, `padding_support_mask`, axial-aligned canonical grid) but has only been validated on a single test patient (Ackdradum).
- Polynomial-coefficient reconstruction at the resampled grid is known to be incorrect under reorientation (the polynomial is built in normalized `[-1, 1]` coordinates spanning the source FOV; resampling to the canonical axial grid remaps axes and extrapolates into padded regions). A fix is needed — reconstruct on the source grid first, then resample like every other data stream — before sagittals can be reintroduced.
- The training loss must learn to ignore padded regions via `padding_support_mask`, which currently has no implementation in the trainer.

Marked: 2026-05-22.

Counts at marking time: **38** in `sagittal_train` (originally `train`), **10** in `sagittal_validation` (originally `validation`). Identified by reading the corrected per-timepoint vx NIfTI and applying `DicomToNiftiConverter.classify_orientation`.

**Recovery path:** when sagittal integration is fully validated, simply rewrite the labels back to `train` / `validation` in `splits_05-05-26.csv`. No data changes required.

---

## Category F: Coronal 4D-flow acquisitions (out of scope for v1 of axial-aligned reorientation)

These patients were acquired with the slice direction along the AP axis (coronal slabs). The "Sagittal Patient Integration" plan was scoped to axial + sagittal only because:

- Population is tiny (2 patients across 384 non-skip = 0.5%), so the marginal training value is small
- Coronal AP-slab padding is the symmetric-but-mirror-image of the sagittal LR-slab padding case; supporting it adds code paths without proportional learning signal
- Current `build_downsampled_full_fov_per_timepoint(reorient_non_axial=True)` would handle coronals correctly via the same `create_axial_aligned_reference_grid` helper if we lift the skip later — no code change needed beyond removing the skip entry

Marked: 2026-05-15.

| Patient | Notes |
|---|---|
| **Epcayit** | Coronal slab; slice direction = -P (anterior). Test build via `scripts/test_sagital_reorientation.py` succeeded with 48.4% padding mask coverage (186 mm AP slab in 379 mm padded AP axis). Data available, just deferred. |
| **Jecifu** | Coronal slab; same direction matrix as Epcayit. Not yet test-built. Data available, just deferred. |

**Recovery path:** when ready to add coronal support, simply remove the skip entries here and from `splits_05-05-26.csv`. The existing reorientation code already handles coronals.

---

## Workflow for re-checking after rsync

After rsyncing raw data from the source archive:

1. For each patient in **Category D / E**, re-list the unzipped DICOM dir and check whether the previously-missing series numbers are now present.
2. If new data appears, regenerate the master DICOM catalog (`dicom_catalog_<pid>.csv`) and the 4D-flow catalog, then re-run the build pipeline for that patient.
3. If the build succeeds, move the patient out of `skip` in `splits/splits_05-05-26.csv` and remove their entry from this file.

For **Category A / B / C** the skip is more structural (code-level filter, upstream pipeline issue, or unsupported acquisition mode). Recovering those patients requires a code change (Category A) or a new upstream run (Category B) or new functionality (Category C).
